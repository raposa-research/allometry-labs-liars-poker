"""Head-to-head cross-play between saved agents, with explicit seat control.

`eval_agents.py` hardcodes `action = where(cur_player == 0, action0, action1)`, so
at 3 players it can only play "agent0 in SEAT 0 versus two copies of agent1".
Seat identity is not incidental in this game -- several 8x10x3 runs showed one
seat behaving completely differently from the other two -- so comparing "A in
seat 0" with "B in seat 0" would confound agent strength with seat effects. This
takes an explicit seat->agent map and rotates the lone agent through every seat.

Also uses the params-only loader from policy_entropy, because
eval_agents.load_agent restores optimizer state (fails against checkpoints
written with a newer optax) and ignores the checkpoint's network_type.

CAVEAT on interpretation: winning cross-play is NOT being closer to equilibrium.
At 3 players the relation need not be transitive, and an agent can beat another
head-to-head while being MORE exploitable by a best response. This measures who
beats whom, nothing else.

Usage:
  crossplay.py A.msgpack B.msgpack [--games N] [--traj N] [--history compact]
"""
import itertools
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

from liars_poker_jax import (apply_action, current_player_id, is_terminal,
                             legal_actions_mask, new_initial_state,
                             observation_tensor, returns as jax_returns)
from rnad import EnvStep
from policy_entropy import load_agent


def play(agents, seat_map, num_games, seed, max_steps):
    """Play num_games; seat i is controlled by agents[seat_map[i]].

    Returns per-seat mean payoff, shape [num_players].
    """
    config = agents[0]._jax_config
    hist_enc = agents[0]._history_encoding
    applies = [jax.vmap(a.network.apply, (None, 0), 0) for a in agents]
    params = [a.params for a in agents]
    smap = jnp.asarray(seat_map, dtype=jnp.int32)

    rng = jax.random.PRNGKey(seed)
    states = jax.vmap(lambda _: new_initial_state(config))(jnp.arange(num_games))

    # Deal phase: one chance action per card, exactly as eval_agents does.
    def deal_step(carry, _):
        states, rng = carry
        rng, sub = jax.random.split(rng)
        digits = jax.random.randint(sub, (num_games,), 0, config.num_digits,
                                    dtype=jnp.int32)
        return (jax.vmap(lambda s, a: apply_action(s, config, a))(states, digits),
                rng), None

    rng, deal_rng = jax.random.split(rng)
    (states, _), _ = jax.lax.scan(deal_step, (states, deal_rng), xs=None,
                                  length=config.total_cards)

    done = jax.vmap(is_terminal)(states)
    final_rewards = jnp.zeros((num_games, config.num_players), dtype=jnp.float64)

    def traj_step(carry, _):
        states, final_rewards, done, rng = carry
        cur_player = jax.vmap(lambda s: current_player_id(s, config))(states)
        # observation_tensor defaults to history_encoding='sparse'; these agents
        # were trained on 'compact', and passing the wrong one silently produces a
        # differently-sized observation (1455 vs 53 at 8x10x3).
        obs = jax.vmap(lambda s: observation_tensor(
            s, config, s.current_player, history_encoding=hist_enc))(states)
        legal = jax.vmap(lambda s: legal_actions_mask(s, config))(states)
        env_step = EnvStep(
            obs=obs.astype(jnp.float64),
            legal=legal.astype(jnp.int8),
            player_id=cur_player.astype(jnp.float64),
            valid=(~done).astype(jnp.float64),
            rewards=jnp.zeros((num_games, config.num_players), dtype=jnp.float64),
        )

        # Sample every agent's action, then select by the acting seat's owner.
        acts = []
        for i, ap in enumerate(applies):
            rng, sub = jax.random.split(rng)
            pi, _, _, _ = ap(params[i], env_step)
            g = jax.random.gumbel(sub, pi.shape, dtype=jnp.float32)
            acts.append(jnp.argmax(jnp.log(pi.astype(jnp.float32) + 1e-45) + g,
                                   axis=-1).astype(jnp.int32))
        owner = smap[cur_player]                     # [num_games] agent index
        action = acts[0]
        for i in range(1, len(acts)):
            action = jnp.where(owner == i, acts[i], action)

        new_states = jax.vmap(lambda s, a: apply_action(s, config, a))(states, action)
        new_done = jax.vmap(is_terminal)(new_states)
        just_term = new_done & ~done
        all_ret = jax.vmap(lambda s: jax_returns(s, config))(new_states).astype(jnp.float64)
        final_rewards = jnp.where(just_term[:, None], all_ret, final_rewards)
        return (new_states, final_rewards, new_done, rng), None

    (_, final_rewards, done, _), _ = jax.lax.scan(
        traj_step, (states, final_rewards, done, rng), xs=None, length=max_steps)
    unfinished = float(np.asarray(1.0 - done.astype(jnp.float32)).mean())
    return np.asarray(final_rewards).mean(axis=0), unfinished


def main(paths, games, traj, history, seed):
    # Label with the RUN directory, not just the file stem: every run writes
    # "agent_2000000.msgpack", so stems collide and `summary[label]` silently
    # overwrote one agent's result with the other's -- the printed summary then
    # showed a single line whose owner was ambiguous.
    def _label(p):
        q = pathlib.Path(p)
        return f"{q.parent.name}/{q.stem}" if q.parent.name else q.stem
    labels = [_label(p) for p in paths]
    if labels[0] == labels[1]:
        labels = [f"{l}#{i}" for i, l in enumerate(labels)]
    agents = [load_agent(p, batch_size=games, trajectory_max=traj,
                         history_encoding=history) for p in paths]
    n = agents[0]._jax_config.num_players
    print(f"{n}-player cross-play, {games} games/rotation, seed {seed}")
    for i, lab in enumerate(labels):
        print(f"  agent {i}: {lab}")

    summary = {}
    for lone in range(len(agents)):
        other = 1 - lone
        per_rot = []
        for seat in range(n):
            smap = [other] * n
            smap[seat] = lone
            mean, unfinished = play(agents, smap, games, seed + seat, traj)
            per_rot.append(mean[seat])
            print(f"  [{labels[lone]} alone in seat {seat}] "
                  + "  ".join(f"seat{j}={mean[j]:+.4f}" for j in range(n))
                  + f"   (lone={mean[seat]:+.4f}"
                  + (f", {unfinished:.1%} unfinished)" if unfinished > 0.001 else ")"))
        summary[labels[lone]] = float(np.mean(per_rot))
    print("\nlone-agent payoff averaged over all seat rotations "
          "(zero-sum: >0 means the lone agent beats the pair):")
    for lab, v in summary.items():
        print(f"  {lab:38s} {v:+.4f}")


if __name__ == "__main__":
    args, kw = [], {"games": 2048, "traj": 100, "history": None, "seed": 0}
    it = iter(sys.argv[1:])
    for a in it:
        if a == "--games":
            kw["games"] = int(next(it))
        elif a == "--traj":
            kw["traj"] = int(next(it))
        elif a == "--history":
            kw["history"] = next(it)
        elif a == "--seed":
            kw["seed"] = int(next(it))
        else:
            args.append(a)
    if len(args) != 2:
        print(__doc__)
        raise SystemExit(2)
    main(args, kw["games"], kw["traj"], kw["history"], kw["seed"])
