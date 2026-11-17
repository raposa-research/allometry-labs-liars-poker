"""lpeas vs the sibling repo's JAX game with dummy actions (dummy_mult > 1).

The sibling `liars_poker_jax` game (branch augmented-game-dummy-actions) appends
(dummy_mult - 1) * num_real_actions synthetic actions. In dummy_mode="lose" a
dummy ends the game with the taker losing; in "alias" it plays the real action
it duplicates. `scripts/eval_liars_poker_checkpoint.py` maps a network policy
over the full action space onto what lpeas solves (a forfeit column, resp.
summed aliases); this test checks that mapping plus the lpeas forfeit
machinery against a brute-force public-tree evaluation that uses the JAX game's
own legal masks, transitions, observations and payoffs.

Needs the sibling checkout on a branch that has `LiarsPokerConfig.dummy_mult`;
set LP_SIBLING_SRC to override; defaults to this repo's src/.
"""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SIBLING_SRC = os.environ.get("LP_SIBLING_SRC", os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.abspath(SIBLING_SRC))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))

jx = pytest.importorskip("liars_poker_jax")
if not hasattr(jx.LiarsPokerConfig, "dummy_mult"):
    pytest.skip("sibling checkout lacks dummy actions", allow_module_level=True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import lpeas  # noqa: E402
from eval_liars_poker_checkpoint import collapse_dummy_probs, network_legal_mask  # noqa: E402

jax.config.update("jax_platform_name", "cpu")

CASES = [(2, 2, 2, "lose"), (2, 2, 3, "lose"), (3, 1, 4, "lose"), (2, 2, 2, "alias"), (3, 1, 3, "alias")]


def make_game(D, H, mult, mode):
    cfg = jx.LiarsPokerConfig(num_players=2, hand_length=H, num_digits=D, dummy_mult=mult, dummy_mode=mode,
                              hand_encoding="raw")
    NH = D**H

    def deal(h0, h1):
        s = jx.new_initial_state(cfg)
        for step in range(2 * H):
            h = (h0, h1)[step % 2]
            digit = (h // D ** (step // 2)) % D + 1  # lpeas hand_digit convention
            s = jx.apply_action(s, cfg, jnp.int32(digit - 1))
        return s

    pairs = [(h0, h1) for h0 in range(NH) for h1 in range(NH)]
    roots = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[deal(h0, h1) for h0, h1 in pairs])
    fns = {
        "apply": jax.jit(jax.vmap(lambda s, a: jx.apply_action(s, cfg, a), (0, None))),
        "legal": jax.jit(jax.vmap(lambda s: jx.legal_actions_mask(s, cfg))),
        "terminal": jax.jit(jax.vmap(jx.is_terminal)),
        "returns": jax.jit(jax.vmap(lambda s: jx.returns(s, cfg))),
        "obs": jax.jit(jax.vmap(lambda s, p: jx.observation_tensor(s, cfg, p, include_history=True,
                                                                     history_encoding="sparse"), (0, None))),
    }
    return cfg, NH, roots, fns


class Reference:
    """Brute-force evaluation over the public tree (hand pairs vectorised)."""

    def __init__(self, cfg, NH, fns, rng):
        self.cfg, self.NH, self.fns, self.rng = cfg, NH, fns, rng
        self.table = {}  # (player, obs bytes) -> probs over the FULL action space
        self.legal_by_key = {}

    def policy(self, player, obs_rows, legal):
        pi = np.zeros((self.NH, self.cfg.num_actions))
        for h in range(self.NH):
            key = (player, obs_rows[h].tobytes())
            if key not in self.table:
                pr = self.rng.random(self.cfg.num_actions) * legal
                self.table[key] = pr / pr.sum()
                self.legal_by_key[key] = legal.copy()
            assert np.array_equal(self.legal_by_key[key], legal)
            pi[h] = self.table[key]
        return pi

    def solve(self, states, reach0, reach1):
        """(E, V0, V1): E = reach-weighted P0 utility; Vp[h] = best-response
        value of player p holding h against the opponent's reach."""
        NH = self.NH
        term = np.asarray(self.fns["terminal"](states))
        if term.all():
            u = np.asarray(self.fns["returns"](states), dtype=np.float64).reshape(NH, NH, 2)
            E = reach0 @ u[:, :, 0] @ reach1
            return E, u[:, :, 0] @ reach1, reach0 @ u[:, :, 1]
        assert not term.any()
        p = int(np.asarray(states.current_player)[0])
        assert (np.asarray(states.current_player) == p).all()
        legal = np.asarray(self.fns["legal"](states))
        assert (legal == legal[0]).all()
        legal = legal[0]
        obs = np.asarray(self.fns["obs"](states, jnp.int32(p))).reshape(NH, NH, -1)
        obs = obs[:, 0] if p == 0 else obs[0, :]  # own-hand rows
        pi = self.policy(p, obs, legal)
        # Actions whose child states coincide (alias dummies) are merged: the
        # JAX observation does not record which of them was played, so the
        # opponent cannot condition on it and only their total mass matters.
        # (Keying the recursion on the action instead lets the best responder
        # read the alias id, which overstates its value.)
        children, mass = {}, {}
        for a in np.flatnonzero(legal):
            child = self.fns["apply"](states, jnp.int32(a))
            key = b"".join(np.asarray(v).tobytes() for v in jax.tree_util.tree_leaves(child))
            if key not in children:
                children[key], mass[key] = child, np.zeros(NH)
            mass[key] += pi[:, a]
        E, V = 0.0, [np.zeros(NH), np.zeros(NH)]
        V[p][:] = -np.inf
        for key, child in children.items():
            r = [reach0, reach1]
            r[p] = r[p] * mass[key]
            Ea, V0a, V1a = self.solve(child, r[0], r[1])
            E += Ea
            Va = [V0a, V1a]
            V[p] = np.maximum(V[p], Va[p])
            V[1 - p] += Va[1 - p]
        return E, V[0], V[1]


@pytest.mark.parametrize("D,H,mult,mode", CASES)
def test_dummy_actions_match_jax_game(D, H, mult, mode):
    cfg, NH, roots, fns = make_game(D, H, mult, mode)
    ref = Reference(cfg, NH, fns, np.random.default_rng(D * 100 + H * 10 + mult))
    E, V0, V1 = ref.solve(roots, np.full(NH, 1.0 / NH), np.full(NH, 1.0 / NH))
    ref_ev0, ref_br = E, (V0.sum() / NH, V1.sum() / NH)

    t = lpeas.LiarsPokerTraverser(D, H)
    assert t.num_actions * mult == cfg.num_actions
    seen = set()

    def policy_fn(player, infostates, legal_real):
        legal_net = network_legal_mask(legal_real, cfg)
        pi = np.zeros((infostates.shape[0], cfg.num_actions))
        for i in range(infostates.shape[0]):
            key = (player, infostates[i].astype(np.float32).tobytes())
            assert key in ref.table, "lpeas infostate tensor not produced by the JAX game"
            assert np.array_equal(ref.legal_by_key[key], legal_net[i])
            seen.add(key)
            pi[i] = ref.table[key]
        return collapse_dummy_probs(pi, legal_real, cfg)

    out = lpeas.ev_and_exploitability(t, policy_fn, chunk_nodes=500, verbose=False)
    assert seen == set(ref.table), "lpeas did not visit every JAX infoset"
    assert out["ev0"] == pytest.approx(ref_ev0, abs=1e-9)
    assert out["br_value"][0] == pytest.approx(ref_br[0], abs=1e-9)
    assert out["br_value"][1] == pytest.approx(ref_br[1], abs=1e-9)


def test_collapse_helpers():
    cfg = jx.LiarsPokerConfig(num_players=2, hand_length=2, num_digits=2, dummy_mult=3, dummy_mode="lose")
    legal = np.array([[False, True, True, False, False, False, False, False, False],
                      [True, False, False, False, False, False, False, False, True]])
    net = network_legal_mask(legal, cfg)
    assert net.shape == (2, 27) and np.array_equal(net, np.tile(legal, (1, 3)))
    pi = np.random.default_rng(0).random((2, 27)) * net
    pi /= pi.sum(axis=1, keepdims=True)
    lose = collapse_dummy_probs(pi, legal, cfg)
    assert lose.shape == (2, 10)
    assert np.allclose(lose[:, :9], pi[:, :9]) and np.allclose(lose[:, 9], pi[:, 9:].sum(axis=1))
    alias = collapse_dummy_probs(pi, legal, jx.LiarsPokerConfig(num_players=2, hand_length=2, num_digits=2,
                                                                  dummy_mult=3, dummy_mode="alias"))
    assert alias.shape == (2, 9) and np.allclose(alias, pi[:, :9] + pi[:, 9:18] + pi[:, 18:])
    rnd_cfg = jx.LiarsPokerConfig(num_players=2, hand_length=2, num_digits=2, dummy_mult=3, dummy_mode="random")
    assert np.array_equal(network_legal_mask(legal, rnd_cfg)[:, 9:], np.ones((2, 18), bool))
    rnd = collapse_dummy_probs(pi, legal, rnd_cfg)
    assert rnd.shape == (2, 9) and np.allclose(rnd.sum(axis=1), 1.0) and (rnd[~legal] == 0).all()
