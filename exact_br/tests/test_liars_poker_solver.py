"""Validates the lpeas Liar's Poker traverser against OpenSpiel on small games.

Run with: .venv/bin/python -m pytest tests/test_liars_poker_solver.py -v
"""

import numpy as np
import pyspiel
import pytest
from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import best_response, expected_game_score, exploitability
import open_spiel.python.games  # noqa: F401  (registers python_liars_poker)

import lpeas

# (num_digits, hand_length); covers hand lengths 1..3 (terminal payoff buckets)
# while staying small enough for OpenSpiel's python best response.
GAMES = [(2, 2), (3, 1), (5, 1), (2, 3)]


def load_game(num_digits, hand_length):
    return pyspiel.load_game(
        f"python_liars_poker(num_digits={num_digits},hand_length={hand_length},players=2)"
    )


def enumerate_infosets(game):
    """Maps information_state_tensor bytes -> (player, legal actions) over the whole tree."""
    out = {}
    stack = [game.new_initial_state()]
    while stack:
        s = stack.pop()
        if s.is_terminal():
            continue
        if s.is_chance_node():
            for a, _ in s.chance_outcomes():
                stack.append(s.child(a))
            continue
        p = s.current_player()
        key = np.asarray(s.information_state_tensor(p)).astype(np.uint8).tobytes()
        legal = s.legal_actions()
        if key in out:
            assert out[key] == (p, legal, s.information_state_string(p))
        else:
            out[key] = (p, legal, s.information_state_string(p))
        for a in legal:
            stack.append(s.child(a))
    return out


def random_tabular_policy(game, seed):
    rng = np.random.RandomState(seed)
    pol = policy_lib.TabularPolicy(game)
    for i in range(pol.action_probability_array.shape[0]):
        mask = pol.legal_actions_mask[i].astype(bool)
        probs = np.zeros_like(pol.action_probability_array[i])
        probs[mask] = rng.dirichlet(np.ones(mask.sum()) * 0.5)
        pol.action_probability_array[i] = probs
    return pol


def policy_fn_from_tabular(game, pol, num_actions):
    """policy_fn for lpeas that looks up the OpenSpiel tabular policy by infostate tensor."""
    table = {}
    for key, (p, legal, info_str) in enumerate_infosets(game).items():
        probs = pol.action_probability_array[pol.state_lookup[info_str]]
        table[key] = np.asarray(probs[:num_actions], dtype=np.float64)

    def policy_fn(player, infostates, legal):
        del player, legal
        return np.stack([table[row.tobytes()] for row in infostates])

    return policy_fn


@pytest.mark.parametrize("num_digits,hand_length", GAMES)
def test_infostates_and_legal_masks_match_openspiel(num_digits, hand_length):
    game = load_game(num_digits, hand_length)
    t = lpeas.LiarsPokerTraverser(num_digits, hand_length)
    assert t.infostate_size == game.information_state_tensor_size()
    expected = enumerate_infosets(game)
    seen = set()
    for level in range(t.num_levels):
        for idx in range(t.level_size(level)):
            for hand in range(t.num_hands):
                tensor, legal = t.node_infostate(level, idx, hand)
                key = tensor.tobytes()
                assert key in expected, t.infostate_string(level, idx, hand)
                p, legal_os, info_str = expected[key]
                assert p == t.level_player(level)
                assert np.flatnonzero(legal).tolist() == legal_os
                assert t.infostate_string(level, idx, hand) == info_str
                seen.add(key)
    assert len(seen) == len(expected)  # every OpenSpiel infoset is visited exactly once
    assert len(seen) == t.num_infosets(0) + t.num_infosets(1)


@pytest.mark.parametrize("num_digits,hand_length", GAMES)
@pytest.mark.parametrize("seed", [0, 1])
def test_ev_and_exploitability_match_openspiel(num_digits, hand_length, seed):
    game = load_game(num_digits, hand_length)
    t = lpeas.LiarsPokerTraverser(num_digits, hand_length)
    pol = random_tabular_policy(game, seed)
    out = lpeas.ev_and_exploitability(
        t, policy_fn_from_tabular(game, pol, t.num_actions), chunk_nodes=1000, verbose=False
    )

    root = game.new_initial_state()
    ev = expected_game_score.policy_value(root, [pol, pol])
    br = [best_response.BestResponsePolicy(game, p, pol).value(root) for p in range(2)]
    nash_conv = exploitability.nash_conv(game, pol)

    assert out["ev0"] == pytest.approx(ev[0], abs=1e-9)
    assert out["br_value"][0] == pytest.approx(br[0], abs=1e-9)
    assert out["br_value"][1] == pytest.approx(br[1], abs=1e-9)
    assert out["expl"][0] == pytest.approx(ev[0] + br[1], abs=1e-9)
    assert out["expl"][1] == pytest.approx(-ev[0] + br[0], abs=1e-9)
    assert out["expl"][0] + out["expl"][1] == pytest.approx(nash_conv, abs=1e-9)


def test_uniform_policy_counts_3x3():
    t = lpeas.LiarsPokerTraverser(3, 3)
    assert t.num_decision_nodes == 39_088_169
    assert t.num_terminals == 24_157_816
    assert t.num_infosets(0) + t.num_infosets(1) == 39_088_169 * 27
    assert t.infostate_size == 79
    assert t.num_actions == 19


def test_br_policy_simulation_2x2():
    """The stored best-response policy, played in OpenSpiel, earns br_value."""
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from verify_br_simulation import simulate

    num_digits, hand_length = 2, 2
    game = load_game(num_digits, hand_length)
    t = lpeas.LiarsPokerTraverser(num_digits, hand_length)
    pol = random_tabular_policy(game, seed=3)
    policy_fn = policy_fn_from_tabular(game, pol, t.num_actions)
    out = lpeas.ev_and_exploitability(t, policy_fn, chunk_nodes=1000, verbose=False, store_br=True)

    class Cfg:
        pass

    cfg = Cfg()
    cfg.num_digits, cfg.hand_length = num_digits, hand_length
    rng = np.random.default_rng(0)
    for br_player in range(2):
        mean, se = simulate(game, t, cfg, policy_fn, br_player, num_games=20_000, batch=5_000, rng=rng)
        assert abs(mean - out["br_value"][br_player]) < 4 * se, (br_player, mean, se, out["br_value"])
