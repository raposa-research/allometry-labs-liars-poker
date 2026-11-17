#!/usr/bin/env python
"""Batch evaluation: pit two saved JAX RNaD agents against each other.

Supports both the new msgpack format (from train_jax.py) and the old pickle
format (from train.py).  Format is auto-detected by file extension.

Usage
-----
  python eval_agents.py --agent0 path/to/agent_a.msgpack --agent1 path/to/agent_b.msgpack
  python eval_agents.py --agent0 old.pickle --agent1 new.msgpack --num-games 1000 --seed 0
"""

import functools
import io
import json
import os
import pickle
import re
import tempfile

import click
import flax
import jax
import jax.core
import jax.numpy as jnp
import numpy as np

from rnad import EnvStep, RNaDConfig, FineTuning
from liars_poker_jax import (
    LiarsPokerConfig,
    new_initial_state,
    apply_action,
    is_terminal,
    current_player_id,
    legal_actions_mask,
    observation_tensor,
    returns as jax_returns,
)

from train_jax import JaxLiarsPokerRNaDSolver, JaxPPOBRAgent


# ---------------------------------------------------------------------------
# S3 helpers (boto3 imported lazily; respects AWS_ENDPOINT_URL for R2)
# ---------------------------------------------------------------------------


def _is_s3(path: str) -> bool:
    return path.startswith("s3://")


def _s3_client():
    import boto3
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
    return boto3.client("s3", endpoint_url=endpoint_url)


def _parse_s3(path: str) -> tuple[str, str]:
    without_scheme = path[len("s3://"):]
    bucket, _, key = without_scheme.partition("/")
    return bucket, key


def _read_bytes(path: str) -> bytes:
    if _is_s3(path):
        bucket, key = _parse_s3(path)
        buf = io.BytesIO()
        _s3_client().download_fileobj(bucket, key, buf)
        return buf.getvalue()
    with open(path, "rb") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Old pickle support (same approach as convert_checkpoint.py)
# ---------------------------------------------------------------------------

# JAX version-compatibility shim: older pickles may include a `named_shape`
# keyword in ShapedArray.__init__ that was later removed.  Only install if JAX
# still has a custom __init__ (newer JAX uses object.__init__ directly).
_orig_shaped_array_init = jax.core.ShapedArray.__init__
if _orig_shaped_array_init is not object.__init__:
    def _patched_shaped_array_init(self, *args, named_shape=None, **kwargs):
        _orig_shaped_array_init(self, *args, **kwargs)
    jax.core.ShapedArray.__init__ = _patched_shaped_array_init


class _OldAgentData:
    """Absorbs the pickled state dict without running any class logic."""
    def __setstate__(self, state):
        self.__dict__.update(state)


class _AgentUnpickler(pickle.Unpickler):
    """Redirects old agent classes to _OldAgentData during unpickling."""
    _REDIRECT = frozenset({
        "JaxLiarsPokerRNaDSolver",
        "RNaDSolver",
        "ParallelRNaDSolver",
    })

    # Solly-A was pickled by OpenSpiel's R-NaD, whose module is not in the
    # open_spiel pip wheel. Its config classes (RNaDConfig, StateRepresentation,
    # AdamConfig, NerdConfig, FineTuning) live on in this repo's rnad.py, which
    # was ported from it, so resolve them there instead.
    _OPENSPIEL_RNAD = "open_spiel.python.algorithms.rnad.rnad"

    def find_class(self, module, name):
        if name in self._REDIRECT:
            return _OldAgentData
        if module == self._OPENSPIEL_RNAD:
            import rnad
            return getattr(rnad, name)
        return super().find_class(module, name)


def _remap_tree(old_tree, new_template):
    """Remap old_tree leaves onto new_template's tree structure."""
    new_treedef = jax.tree_util.tree_structure(new_template)
    old_leaves = jax.tree_util.tree_leaves(old_tree)
    new_leaves = jax.tree_util.tree_leaves(new_template)
    if len(old_leaves) != len(new_leaves):
        raise ValueError(
            f"Leaf count mismatch — old checkpoint has {len(old_leaves)} "
            f"leaves but the new template expects {len(new_leaves)}."
        )
    return jax.tree_util.tree_unflatten(new_treedef, old_leaves)


def _trim_policy_head(tree, old_num_actions: int, new_num_actions: int):
    """Slice policy-head leaves from old_num_actions down to new_num_actions."""
    def trim(leaf):
        if leaf.shape and leaf.shape[-1] == old_num_actions:
            return leaf[..., :new_num_actions]
        return leaf
    return jax.tree_util.tree_map(trim, tree)


def _load_pickle_agent(path: str) -> JaxLiarsPokerRNaDSolver:
    """Load an old pickle checkpoint into a JaxLiarsPokerRNaDSolver."""
    with open(path, "rb") as f:
        old = _AgentUnpickler(f).load()

    # Rebuild the config as a CURRENT RNaDConfig rather than reusing the
    # unpickled instance. These pickles were written by an older class, so they
    # lack any field added since (e.g. warmup_steps, which train_jax.init()
    # reads unconditionally -- reusing the old object raises AttributeError).
    # Copying field-by-field picks up current defaults for anything missing and
    # stays correct as further fields are added.
    import dataclasses as _dc
    _old_cfg = old.config
    rnad_config = RNaDConfig(**{
        f.name: getattr(_old_cfg, f.name)
        for f in _dc.fields(RNaDConfig) if hasattr(_old_cfg, f.name)
    })

    # Reconstruct game config from game_name string. Older checkpoints spell
    # the player count `players=`, newer ones `num_players=`; missing it
    # silently yields a 2-player config for a 3-player agent.
    game_params = dict(re.findall(r'(\w+)=(\d+)', rnad_config.game_name))
    jax_config = LiarsPokerConfig(
        num_players=int(game_params.get("num_players",
                                        game_params.get("players", 2))),
        hand_length=int(game_params["hand_length"]),
        num_digits=int(game_params["num_digits"]),
        # OpenSpiel's python_liars_poker writes RAW hand digits. This tree
        # defaults to a per-digit histogram, and the two have the SAME
        # observation length when hand_length == num_digits, so a mismatch
        # fails silently: the agent simply stops seeing its own hand.
        hand_encoding="raw",
    )

    agent = JaxLiarsPokerRNaDSolver(rnad_config, jax_config)

    # Detect old policy-head size for possible trimming.
    last_hidden = rnad_config.policy_network_layers[-1]
    policy_kernels = [
        l for l in jax.tree_util.tree_leaves(old.params)
        if l.ndim == 2 and l.shape[0] == last_hidden
        and l.shape[1] != 1 and l.shape[0] != l.shape[1]
    ]
    if len(policy_kernels) != 1:
        raise ValueError(
            f"Expected exactly one policy-head kernel, found {len(policy_kernels)}."
        )
    old_num_actions = int(policy_kernels[0].shape[1])
    new_num_actions = jax_config.num_actions
    trimming = old_num_actions != new_num_actions

    def remap_and_trim(old_tree, template):
        remapped = _remap_tree(old_tree, template)
        if trimming:
            return _trim_policy_head(remapped, old_num_actions, new_num_actions)
        return remapped

    agent.params = remap_and_trim(old.params, agent.params)
    agent.params_target = remap_and_trim(old.params_target, agent.params_target)
    agent.params_prev = remap_and_trim(old.params_prev, agent.params_prev)
    agent.params_prev_ = remap_and_trim(old.params_prev_, agent.params_prev_)

    agent.optimizer.state = remap_and_trim(old.optimizer, agent.optimizer.state)
    agent.optimizer_target.state = remap_and_trim(
        old.optimizer_target, agent.optimizer_target.state
    )

    agent.learner_steps = old.learner_steps
    agent.actor_steps = old.actor_steps
    agent._rngkey = old.rngkey  # __getstate__ stored it as 'rngkey'

    return agent


# ---------------------------------------------------------------------------
# Msgpack loaders (unchanged)
# ---------------------------------------------------------------------------


def load_br_agent(path: str) -> JaxPPOBRAgent:
    """Load a JaxPPOBRAgent from a BR msgpack checkpoint (has 'br_player' key)."""
    with open(path, "rb") as f:
        raw = f.read()

    meta = flax.serialization.msgpack_restore(raw)
    jax_cfg = meta["jax_config"]
    pnl = meta["policy_network_layers"]
    if isinstance(pnl, dict):
        layers = [int(pnl[str(i)]) for i in range(len(pnl))]
    else:
        layers = [int(x) for x in pnl]

    jax_config = LiarsPokerConfig(
        num_players=int(jax_cfg["num_players"]),
        hand_length=int(jax_cfg["hand_length"]),
        num_digits=int(jax_cfg["num_digits"]),
    )
    agent = JaxPPOBRAgent(
        jax_config=jax_config,
        br_player=int(meta["br_player"]),
        network_layers=layers,
    )
    template = {"params": agent.params, "opt_state": agent.opt_state,
                "jax_config": meta["jax_config"], "policy_network_layers": meta["policy_network_layers"],
                "br_player": meta["br_player"]}
    restored = flax.serialization.from_bytes(template, raw)
    agent.params = restored["params"]
    agent.params_target = agent.params  # alias for eval compatibility
    return agent


def _load_msgpack_agent(path: str):
    """Load an agent from a msgpack checkpoint (RNaD or BR, auto-detected)."""
    with open(path, "rb") as f:
        raw = f.read()

    meta = flax.serialization.msgpack_restore(raw)

    if "br_player" in meta:
        return load_br_agent(path)
    jax_cfg = meta["jax_config"]
    pnl = meta["policy_network_layers"]
    if isinstance(pnl, dict):
        layers = tuple(pnl[str(i)] for i in range(len(pnl)))
    else:
        layers = tuple(int(x) for x in pnl)

    jax_config = LiarsPokerConfig(
        num_players=int(jax_cfg["num_players"]),
        hand_length=int(jax_cfg["hand_length"]),
        num_digits=int(jax_cfg["num_digits"]),
    )
    rnad_config = RNaDConfig(
        game_name="liars_poker",
        policy_network_layers=layers,
    )
    agent = JaxLiarsPokerRNaDSolver(rnad_config, jax_config)
    agent.restore_state(raw)
    return agent


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------


def load_agent(path: str):
    """Load an agent from a checkpoint file (pickle or msgpack, auto-detected).

    Accepts local paths or s3:// URIs. Format is determined by file extension:
      .pickle  — old Haiku-based RNaDSolver (converted in-memory to Flax)
      .msgpack — new Flax-based JaxLiarsPokerRNaDSolver or JaxPPOBRAgent
    """
    if _is_s3(path):
        suffix = ".pickle" if path.endswith(".pickle") else ".msgpack"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(_read_bytes(path))
            tmp_path = tmp.name
        try:
            return load_agent(tmp_path)
        finally:
            os.unlink(tmp_path)
    if path.endswith(".pickle"):
        return _load_pickle_agent(path)
    return _load_msgpack_agent(path)


# ---------------------------------------------------------------------------
# Vectorized game rollout
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(0, 1, 2, 3, 7))
def _play_games(network_apply0, network_apply1, jax_config, num_games, params0, params1, rng_key, max_steps):
    """Play num_games games in parallel using vmap + lax.scan.

    Returns:
      final_rewards: [num_games, num_players]
      final_done:    [num_games] bool
      action_counts: [num_players, num_actions] int32 – actions taken per player
    """
    config = jax_config

    # Initialise a batch of independent games.
    batch_states = jax.vmap(lambda _: new_initial_state(config))(
        jnp.zeros(num_games, dtype=jnp.int32)
    )

    # Deal phase: assign hands via random chance actions.
    def deal_step(carry, _):
        states, rng = carry
        rng, sub = jax.random.split(rng)
        digits = jax.random.randint(
            sub, (num_games,), 0, config.num_digits, dtype=jnp.int32
        )
        return (
            jax.vmap(lambda s, a: apply_action(s, config, a))(states, digits),
            rng,
        ), None

    rng_key, deal_rng = jax.random.split(rng_key)
    (batch_states, rng_key), _ = jax.lax.scan(
        deal_step, (batch_states, deal_rng), xs=None, length=config.total_cards
    )

    # Play phase: scan over max_steps, tracking terminal rewards and action counts.
    init_done = jax.vmap(is_terminal)(batch_states)
    init_rewards = jnp.zeros((num_games, config.num_players), dtype=jnp.float64)
    init_action_counts = jnp.zeros((config.num_players, config.num_actions), dtype=jnp.int32)

    def traj_step(carry, _):
        states, final_rewards, done, rng, action_counts = carry
        rng, rng0, rng1 = jax.random.split(rng, 3)

        cur_player = jax.vmap(lambda s: current_player_id(s, config))(states)
        obs = jax.vmap(lambda s: observation_tensor(s, config, s.current_player))(states)
        legal = jax.vmap(lambda s: legal_actions_mask(s, config))(states)

        env_step = EnvStep(
            obs=obs.astype(jnp.float64),
            legal=legal.astype(jnp.int8),
            player_id=cur_player.astype(jnp.float64),
            valid=(~done).astype(jnp.float64),
            rewards=jnp.zeros((num_games, config.num_players), dtype=jnp.float64),
        )

        pi0, _, _, _ = jax.vmap(network_apply0, (None, 0), 0)(params0, env_step)
        pi1, _, _, _ = jax.vmap(network_apply1, (None, 0), 0)(params1, env_step)

        gumbels0 = jax.random.gumbel(rng0, pi0.shape, dtype=jnp.float32)
        gumbels1 = jax.random.gumbel(rng1, pi1.shape, dtype=jnp.float32)
        action0 = jnp.argmax(
            jnp.log(pi0.astype(jnp.float32) + 1e-45) + gumbels0, axis=-1
        ).astype(jnp.int32)
        action1 = jnp.argmax(
            jnp.log(pi1.astype(jnp.float32) + 1e-45) + gumbels1, axis=-1
        ).astype(jnp.int32)

        # Select action by current player; done games advance regardless
        # (their results are already captured in final_rewards).
        action = jnp.where(cur_player == 0, action0, action1)

        new_states = jax.vmap(lambda s, a: apply_action(s, config, a))(states, action)

        new_done = jax.vmap(is_terminal)(new_states)
        just_terminated = new_done & ~done
        all_ret = jax.vmap(lambda s: jax_returns(s, config))(new_states).astype(
            jnp.float64
        )
        final_rewards = jnp.where(just_terminated[:, None], all_ret, final_rewards)

        # Accumulate action counts for non-done games.
        # player_onehot: [num_games, num_players], action_onehot: [num_games, num_actions]
        valid = (~done).astype(jnp.int32)
        player_onehot = (jnp.arange(config.num_players)[None, :] == cur_player[:, None]).astype(jnp.int32)
        action_onehot = (jnp.arange(config.num_actions)[None, :] == action[:, None]).astype(jnp.int32)
        # contribution: [num_games, num_players, num_actions] -> sum over games
        contribution = (player_onehot[:, :, None] * action_onehot[:, None, :] * valid[:, None, None])
        action_counts = action_counts + contribution.sum(axis=0)

        return (new_states, final_rewards, new_done, rng, action_counts), None

    (_, final_rewards, final_done, _, action_counts), _ = jax.lax.scan(
        traj_step,
        (batch_states, init_rewards, init_done, rng_key, init_action_counts),
        xs=None,
        length=max_steps,
    )
    return final_rewards, final_done, action_counts


@functools.partial(jax.jit, static_argnums=(0, 1, 2, 3, 7))
def _play_games_with_history(network_apply0, network_apply1, jax_config, num_games, params0, params1, rng_key, max_steps):
    """Like _play_games but also returns per-step move history and policy distributions.

    Extra returns (all shape [max_steps, num_games, ...]):
      step_actions:  [max_steps, num_games] int32 – action taken at each step
      step_policies: [max_steps, num_games, num_actions] float32 – acting player's policy
      step_players:  [max_steps, num_games] int32 – which player acted
      step_valid:    [max_steps, num_games] bool – whether this step was still active
      hands:         [num_games, num_players, hand_length] int32 – dealt hands
    """
    config = jax_config

    batch_states = jax.vmap(lambda _: new_initial_state(config))(
        jnp.zeros(num_games, dtype=jnp.int32)
    )

    def deal_step(carry, _):
        states, rng = carry
        rng, sub = jax.random.split(rng)
        digits = jax.random.randint(
            sub, (num_games,), 0, config.num_digits, dtype=jnp.int32
        )
        return (
            jax.vmap(lambda s, a: apply_action(s, config, a))(states, digits),
            rng,
        ), None

    rng_key, deal_rng = jax.random.split(rng_key)
    (batch_states, rng_key), _ = jax.lax.scan(
        deal_step, (batch_states, deal_rng), xs=None, length=config.total_cards
    )

    # Save dealt hands before play begins.
    hands = batch_states.hands  # [num_games, num_players, hand_length]

    init_done = jax.vmap(is_terminal)(batch_states)
    init_rewards = jnp.zeros((num_games, config.num_players), dtype=jnp.float64)

    def traj_step(carry, _):
        states, final_rewards, done, rng = carry
        rng, rng0, rng1 = jax.random.split(rng, 3)

        cur_player = jax.vmap(lambda s: current_player_id(s, config))(states)
        obs = jax.vmap(lambda s: observation_tensor(s, config, s.current_player))(states)
        legal = jax.vmap(lambda s: legal_actions_mask(s, config))(states)

        env_step = EnvStep(
            obs=obs.astype(jnp.float64),
            legal=legal.astype(jnp.int8),
            player_id=cur_player.astype(jnp.float64),
            valid=(~done).astype(jnp.float64),
            rewards=jnp.zeros((num_games, config.num_players), dtype=jnp.float64),
        )

        pi0, _, _, _ = jax.vmap(network_apply0, (None, 0), 0)(params0, env_step)
        pi1, _, _, _ = jax.vmap(network_apply1, (None, 0), 0)(params1, env_step)

        gumbels0 = jax.random.gumbel(rng0, pi0.shape, dtype=jnp.float32)
        gumbels1 = jax.random.gumbel(rng1, pi1.shape, dtype=jnp.float32)
        action0 = jnp.argmax(
            jnp.log(pi0.astype(jnp.float32) + 1e-45) + gumbels0, axis=-1
        ).astype(jnp.int32)
        action1 = jnp.argmax(
            jnp.log(pi1.astype(jnp.float32) + 1e-45) + gumbels1, axis=-1
        ).astype(jnp.int32)

        action = jnp.where(cur_player == 0, action0, action1)
        # Policy of the acting player
        acting_pi = jnp.where(cur_player[:, None] == 0, pi0, pi1)

        new_states = jax.vmap(lambda s, a: apply_action(s, config, a))(states, action)
        new_done = jax.vmap(is_terminal)(new_states)
        just_terminated = new_done & ~done
        all_ret = jax.vmap(lambda s: jax_returns(s, config))(new_states).astype(jnp.float64)
        final_rewards = jnp.where(just_terminated[:, None], all_ret, final_rewards)

        step_info = {
            "action": action,
            "policy": acting_pi.astype(jnp.float32),
            "player": cur_player,
            "valid": ~done,
        }
        return (new_states, final_rewards, new_done, rng), step_info

    (_, final_rewards, final_done, _), step_data = jax.lax.scan(
        traj_step,
        (batch_states, init_rewards, init_done, rng_key),
        xs=None,
        length=max_steps,
    )
    # step_data keys each have leading dim [max_steps, num_games, ...]
    return final_rewards, final_done, hands, step_data


def _action_label(action_idx: int, config) -> str:
    """Human-readable label for an action index."""
    if action_idx == 0:
        return "challenge"
    bid_id = action_idx - 1
    count = bid_id // config.num_digits + 1
    digit = bid_id % config.num_digits + 1
    return f"{count}x{digit}"


def _games_to_records(hands, step_data, rewards, done, jax_config, agent0_label, agent1_label, agent0_is_p0=True):
    """Convert JAX arrays into a list of JSON-serializable game dicts."""
    num_games = int(hands.shape[0])
    max_steps = int(step_data["action"].shape[0])
    num_actions = jax_config.num_actions

    records = []
    for g in range(num_games):
        if not done[g]:
            continue  # skip timed-out games
        game_hands = {}
        for p in range(jax_config.num_players):
            hand_digits = [int(hands[g, p, i]) for i in range(jax_config.hand_length)]
            game_hands[f"player_{p}"] = hand_digits

        moves = []
        for t in range(max_steps):
            if not step_data["valid"][t, g]:
                break
            player = int(step_data["player"][t, g])
            action = int(step_data["action"][t, g])
            policy = step_data["policy"][t, g]  # [num_actions]

            # Build sparse policy dict (only legal/nonzero entries)
            policy_dict = {}
            for a in range(num_actions):
                p_val = float(policy[a])
                if p_val > 1e-6:
                    policy_dict[_action_label(a, jax_config)] = round(p_val, 6)

            if agent0_is_p0:
                agent_label = agent0_label if player == 0 else agent1_label
            else:
                agent_label = agent1_label if player == 0 else agent0_label

            moves.append({
                "player": player,
                "agent": agent_label,
                "action": _action_label(action, jax_config),
                "policy": policy_dict,
            })

        game_rewards = [float(rewards[g, p]) for p in range(jax_config.num_players)]
        records.append({
            "hands": game_hands,
            "moves": moves,
            "rewards": {f"player_{p}": game_rewards[p] for p in range(jax_config.num_players)},
            "agent_positions": {
                "player_0": agent0_label if agent0_is_p0 else agent1_label,
                "player_1": agent1_label if agent0_is_p0 else agent0_label,
            },
        })
    return records


def _hot_rgb(v: float) -> tuple[int, int, int] | None:
    """Map v in [0, 1] to an RGB tuple (purple→red→orange→yellow). Returns None for v<=0."""
    if v <= 0:
        return None
    stops = [
        (0.00, (30,   0,  80)),
        (0.33, (180,   0, 100)),
        (0.66, (255, 120,   0)),
        (1.00, (255, 255, 100)),
    ]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if v <= t1:
            t = (v - t0) / (t1 - t0)
            return tuple(int(a + (b - a) * t) for a, b in zip(c0, c1))
    return stops[-1][1]


def _color_cell(text: str, v: float) -> str:
    """Wrap text with a 24-bit ANSI background color scaled by v in [0, 1]."""
    rgb = _hot_rgb(v)
    if rgb is None:
        return text
    r, g, b = rgb
    return f"\x1b[48;2;{r};{g};{b}m\x1b[97m{text}\x1b[0m"


def print_action_heatmap(action_counts: np.ndarray, jax_config, agent_label: str):
    """Print a colored count x digit heatmap of actions for each player position."""
    num_digits = jax_config.num_digits
    max_count = jax_config.hand_length * jax_config.num_players
    num_players = jax_config.num_players

    for player_idx in range(num_players):
        counts = action_counts[player_idx]  # [num_actions]
        total = int(counts.sum())
        # Normalize against the max bid count for this player (across both positions).
        bid_max = max(int(counts[1:].max()), 1)

        click.echo(f"\n  {agent_label} as player {player_idx}  (total actions: {total})")

        # Challenge/call row
        call_n = int(counts[0])
        call_pct = 100.0 * call_n / total if total else 0.0
        call_v = call_n / bid_max
        call_text = f"{call_n:5d}  ({call_pct:5.1f}%)"
        click.echo(f"  {'CALL':>6s}: {_color_cell(call_text, call_v)}")

        # Header: digits 1..num_digits
        header = "  " + " " * 7 + "".join(f"  d{d:<2d}" for d in range(1, num_digits + 1))
        click.echo(header)

        # Grid rows: count 1..max_count
        for c in range(1, max_count + 1):
            row_counts = []
            for d in range(1, num_digits + 1):
                bid_id = (c - 1) * num_digits + (d - 1)
                action_idx = bid_id + 1  # BID_ACTION_OFFSET = 1
                val = int(counts[action_idx]) if action_idx < len(counts) else 0
                row_counts.append(val)
            if sum(row_counts) == 0:
                continue  # skip unpopulated count rows
            cells = "".join(_color_cell(f"{v:5d}", v / bid_max) for v in row_counts)
            click.echo(f"  c{c:3d}:  {cells}")


# ---------------------------------------------------------------------------
# Finetuning wrapper
# ---------------------------------------------------------------------------


def _make_finetuned_apply(network_apply, finetune: FineTuning):
    """Return a wrapped apply fn that post-processes policy with FineTuning."""
    def apply(params, env_step):
        pi, v, log_pi, logit = network_apply(params, env_step)
        pi = finetune.post_process_policy(pi, env_step.legal)
        return pi, v, log_pi, logit
    return apply


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--agent0",
    required=True,
    help="Path to checkpoint for player 0 (.msgpack or .pickle); local or s3://.",
)
@click.option(
    "--agent1",
    required=True,
    help="Path to checkpoint for player 1 (.msgpack or .pickle); local or s3://.",
)
@click.option(
    "--num-games",
    default=500,
    show_default=True,
    help="Number of games to play.",
)
@click.option(
    "--max-steps",
    default=50,
    show_default=True,
    help="Maximum game steps per rollout (must cover longest possible game).",
)
@click.option(
    "--seed",
    default=42,
    show_default=True,
    help="Random seed for reproducibility.",
)
@click.option(
    "--verbose/--no-verbose",
    default=False,
    show_default=True,
    help="Print per-game results.",
)
@click.option(
    "--save-games",
    default=None,
    type=click.Path(),
    help="Path to save a JSON file with detailed game logs (moves + policies).",
)
@click.option(
    "--save-num-games",
    default=20,
    show_default=True,
    help="Number of games to record in the JSON file (per position).",
)
@click.option(
    "--finetune0/--no-finetune0",
    default=False,
    show_default=True,
    help="Apply FineTuning post-processing to agent 0's policy.",
)
@click.option(
    "--finetune1/--no-finetune1",
    default=False,
    show_default=True,
    help="Apply FineTuning post-processing to agent 1's policy.",
)
def main(agent0: str, agent1: str, num_games: int, max_steps: int, seed: int, verbose: bool, save_games: str | None, save_num_games: int, finetune0: bool, finetune1: bool):
    """Batch evaluation: two saved JAX RNaD agents play Liar's Poker."""
    click.echo(f"Loading agent 0 from {agent0}")
    a0 = load_agent(agent0)

    click.echo(f"Loading agent 1 from {agent1}")
    a1 = load_agent(agent1)

    # Both agents must share the same jax_config (game parameters).
    jax_config = a0._jax_config

    ft = FineTuning()
    apply0 = _make_finetuned_apply(a0.network.apply, ft) if finetune0 else a0.network.apply
    apply1 = _make_finetuned_apply(a1.network.apply, ft) if finetune1 else a1.network.apply
    if finetune0:
        click.echo("FineTuning enabled for agent 0.")
    if finetune1:
        click.echo("FineTuning enabled for agent 1.")

    rng_key = jax.random.PRNGKey(seed)
    rng_key, rng_games = jax.random.split(rng_key)

    total_games = num_games * 2

    # Both runs use the same rng_games so they play identical game situations,
    # with only the agent roles swapped.  This makes results exactly symmetric:
    # swapping agent0/agent1 produces mirrored stats.
    # --- Run A: agent0 as player 0, agent1 as player 1 ---
    click.echo(f"Running {num_games} games: agent0=p0, agent1=p1 (max_steps={max_steps}) ...")
    rewards_ab, done_ab, counts_ab = _play_games(
        apply0, apply1, jax_config, num_games,
        a0.params_target, a1.params_target, rng_games, max_steps,
    )
    rewards_ab = np.array(rewards_ab)   # [num_games, 2]
    done_ab = np.array(done_ab)
    counts_ab = np.array(counts_ab)     # [2, num_actions]: index 0=a0, 1=a1

    # --- Run B: agent1 as player 0, agent0 as player 1 ---
    click.echo(f"Running {num_games} games: agent1=p0, agent0=p1 (max_steps={max_steps}) ...")
    rewards_ba, done_ba, counts_ba = _play_games(
        apply1, apply0, jax_config, num_games,
        a1.params_target, a0.params_target, rng_games, max_steps,
    )
    rewards_ba = np.array(rewards_ba)   # [num_games, 2]: index 0=a1, 1=a0
    done_ba = np.array(done_ba)
    counts_ba = np.array(counts_ba)     # [2, num_actions]: index 0=a1, 1=a0

    # Combine rewards: agent0 rewards = (as p0 from ab) + (as p1 from ba)
    rewards_a0 = np.concatenate([rewards_ab[:, 0], rewards_ba[:, 1]])
    rewards_a1 = np.concatenate([rewards_ab[:, 1], rewards_ba[:, 0]])

    if verbose:
        click.echo("\n--- Run A (agent0=p0) ---")
        for i, r in enumerate(rewards_ab):
            click.echo(f"Game {i+1:>5d}: a0={r[0]:+.1f}  a1={r[1]:+.1f}")
        click.echo("\n--- Run B (agent1=p0) ---")
        for i, r in enumerate(rewards_ba):
            click.echo(f"Game {i+1:>5d}: a1={r[0]:+.1f}  a0={r[1]:+.1f}")

    n_completed = int(done_ab.sum()) + int(done_ba.sum())

    click.echo(f"\n{'='*50}")
    click.echo(f"Results over {total_games} games ({num_games} per position, "
               f"{n_completed} completed, {total_games - n_completed} timed out)")
    click.echo(f"{'='*50}")

    # Per-position breakdown: rewards_ab = agent0 as p0, agent1 as p1
    #                         rewards_ba = agent1 as p0, agent0 as p1
    for label, r_self, r_opp, n_games in [
        ("as p0", rewards_ab[:, 0], rewards_ab[:, 1], num_games),
        ("as p1", rewards_ba[:, 1], rewards_ba[:, 0], num_games),
    ]:
        wins = int((r_self > r_opp).sum())
        mean = r_self.mean()
        std = r_self.std() / np.sqrt(n_games)
        click.echo(f"  Agent 0 {label}:  mean {mean:+.4f}  (+/- {std:.4f})  wins: {wins}/{n_games} ({100*wins/n_games:.1f}%)")

    rewards_a0 = np.concatenate([rewards_ab[:, 0], rewards_ba[:, 1]])
    rewards_a1 = np.concatenate([rewards_ab[:, 1], rewards_ba[:, 0]])
    wins0 = int((rewards_a0 > rewards_a1).sum())
    wins1 = int((rewards_a1 > rewards_a0).sum())
    draws = total_games - wins0 - wins1
    mean0 = rewards_a0.mean()
    mean1 = rewards_a1.mean()
    std0 = rewards_a0.std() / np.sqrt(total_games)
    std1 = rewards_a1.std() / np.sqrt(total_games)
    click.echo(f"Agent 0  overall mean: {mean0:+.4f}  (+/- {std0:.4f})  wins: {wins0} ({100*wins0/total_games:.1f}%)")
    click.echo(f"Agent 1  overall mean: {mean1:+.4f}  (+/- {std1:.4f})  wins: {wins1} ({100*wins1/total_games:.1f}%)")
    if draws:
        click.echo(f"Draws: {draws} ({100*draws/total_games:.1f}%)")

    # --- Action heatmaps ---
    # counts_ab[0] = agent0 actions as p0; counts_ba[1] = agent0 actions as p1
    # counts_ab[1] = agent1 actions as p1; counts_ba[0] = agent1 actions as p0
    counts_a0 = np.stack([counts_ab[0], counts_ba[1]])  # [2, num_actions]: as p0, as p1
    counts_a1 = np.stack([counts_ba[0], counts_ab[1]])  # [2, num_actions]: as p0, as p1

    click.echo(f"\n{'='*50}")
    click.echo("Action heatmaps (counts per bid: count x digit)")
    click.echo(f"{'='*50}")
    click.echo("\nAgent 0:")
    print_action_heatmap(counts_a0, jax_config, "Agent 0")
    click.echo("\nAgent 1:")
    print_action_heatmap(counts_a1, jax_config, "Agent 1")

    # --- Save detailed game logs ---
    if save_games:
        a0_label = os.path.basename(agent0)
        a1_label = os.path.basename(agent1)

        rng_key, rng_save = jax.random.split(rng_key)
        n = min(save_num_games, num_games)

        click.echo(f"\nRecording {n} games per position for JSON export...")
        # Run A: agent0=p0
        rew_a, done_a, hands_a, steps_a = _play_games_with_history(
            apply0, apply1, jax_config, n,
            a0.params_target, a1.params_target, rng_save, max_steps,
        )
        rew_a, done_a, hands_a = np.array(rew_a), np.array(done_a), np.array(hands_a)
        steps_a = {k: np.array(v) for k, v in steps_a.items()}

        records = _games_to_records(hands_a, steps_a, rew_a, done_a, jax_config,
                                    a0_label, a1_label, agent0_is_p0=True)

        # Run B: agent1=p0
        rew_b, done_b, hands_b, steps_b = _play_games_with_history(
            apply1, apply0, jax_config, n,
            a1.params_target, a0.params_target, rng_save, max_steps,
        )
        rew_b, done_b, hands_b = np.array(rew_b), np.array(done_b), np.array(hands_b)
        steps_b = {k: np.array(v) for k, v in steps_b.items()}

        records += _games_to_records(hands_b, steps_b, rew_b, done_b, jax_config,
                                     a0_label, a1_label, agent0_is_p0=False)

        output = {
            "agent0": a0_label,
            "agent1": a1_label,
            "config": {
                "num_players": jax_config.num_players,
                "hand_length": jax_config.hand_length,
                "num_digits": jax_config.num_digits,
            },
            "games": records,
        }
        with open(save_games, "w") as f:
            json.dump(output, f, indent=2)
        click.echo(f"Saved {len(records)} games to {save_games}")


if __name__ == "__main__":
    main()
