"""Tests for the dummy-action space-scaling probe (config.dummy_mult).

The probe exists to vary |A| while holding the GAME fixed, so that
action-space size can be shown to be what hurts NeuRD (whose v-trace target
carries inv_mu = 1/mu(a_t), which scales with |A| at a flat policy) and not
some other difference. These tests pin the properties the experiment relies
on; if any fails, the experiment measures something other than |A|.

Run: PYTHONPATH=src python src/test_dummy_actions.py
"""
import jax
import jax.numpy as jnp
import numpy as np

from liars_poker_jax import (CHALLENGE_ACTION, LiarsPokerConfig, apply_action,
                             is_chance_node, is_terminal, legal_actions_mask,
                             new_initial_state, observation_tensor,
                             returns as jax_returns, _resolve_dummy)

BASE = LiarsPokerConfig(num_players=2, hand_length=3, num_digits=3)


def _dealt(config, seed=0):
    """A state just past the deal phase."""
    s = new_initial_state(config)
    rng = np.random.default_rng(seed)
    for _ in range(config.total_cards):
        s = apply_action(s, config, jnp.int32(rng.integers(config.num_digits)))
    return s


def test_action_count_scales_exactly():
    assert BASE.num_actions == BASE.num_real_actions == 19, BASE.num_actions
    for m in (1, 2, 4, 8, 16):
        c = LiarsPokerConfig(num_players=2, hand_length=3, num_digits=3,
                             dummy_mult=m)
        assert c.num_real_actions == 19
        assert c.num_actions == 19 * m, (m, c.num_actions)
    print("  |A| = 19, 38, 76, 152, 304 for mult 1,2,4,8,16")


def test_mult1_is_byte_identical():
    """dummy_mult=1 must not perturb the game at all."""
    c1 = LiarsPokerConfig(num_players=2, hand_length=3, num_digits=3)
    cm = LiarsPokerConfig(num_players=2, hand_length=3, num_digits=3,
                          dummy_mult=1)
    s1, sm = _dealt(c1), _dealt(cm)
    assert np.array_equal(np.asarray(legal_actions_mask(s1, c1)),
                          np.asarray(legal_actions_mask(sm, cm)))
    for a in range(c1.num_real_actions):
        if not np.asarray(legal_actions_mask(s1, c1))[a]:
            continue
        n1 = apply_action(s1, c1, jnp.int32(a))
        nm = apply_action(sm, cm, jnp.int32(a))
        assert np.array_equal(np.asarray(n1.bid_history),
                              np.asarray(nm.bid_history))
        assert int(n1.current_bid_action) == int(nm.current_bid_action)
    print("  dummy_mult=1 reproduces the original mask and transitions")


def test_real_action_prefix_unchanged():
    """The first num_real_actions entries of the mask must not move."""
    for m in (2, 4, 16):
        c = LiarsPokerConfig(num_players=2, hand_length=3, num_digits=3,
                             dummy_mult=m)
        s, s1 = _dealt(c), _dealt(BASE)
        full = np.asarray(legal_actions_mask(s, c))
        base = np.asarray(legal_actions_mask(s1, BASE))
        assert np.array_equal(full[:BASE.num_real_actions], base), m
    print("  real-action prefix of the mask is identical at every multiplier")


def test_dummies_legal_exactly_when_a_real_action_is():
    """Dummy legality must track real legality at EVERY node type.

    NOTE on chance nodes: `legal_actions_mask` does NOT guard them -- in the
    unmodified game it already reports 18/19 actions legal during the deal
    (`bid_originator` is -1, so `can_bid` is true), and callers are expected to
    check `is_chance_node` themselves. So the invariant to assert is NOT "no
    dummy is legal during the deal" but "dummies are legal exactly when some
    real action is", which is what the implementation guarantees and what
    keeps the probe consistent with the base game.
    """
    c = LiarsPokerConfig(num_players=2, hand_length=3, num_digits=3,
                         dummy_mult=4)
    # chance, decision, and terminal -- the invariant must hold at all three.
    states = [(new_initial_state(c), "chance"), (_dealt(c), "decision")]
    s = _dealt(c)
    guard = 0
    while not bool(is_terminal(s)) and guard < 200:
        legal = np.flatnonzero(np.asarray(legal_actions_mask(s, c)))
        s = apply_action(s, c, jnp.int32(legal[0]))
        guard += 1
    assert bool(is_terminal(s))
    states.append((s, "terminal"))
    for st, label in states:
        m = np.asarray(legal_actions_mask(st, c))
        any_real = bool(m[:c.num_real_actions].any())
        assert bool(m[c.num_real_actions:].all()) == any_real, (label, m.sum())
        assert bool(m[c.num_real_actions:].any()) == any_real, (label, m.sum())
    print("  dummy legality tracks real legality at chance/decision/terminal")


def test_dummy_resolves_to_a_legal_real_action():
    c = LiarsPokerConfig(num_players=2, hand_length=3, num_digits=3,
                         dummy_mult=8)
    for seed in range(6):
        s = _dealt(c, seed)
        real = np.asarray(legal_actions_mask(s, c))[:c.num_real_actions]
        for a in range(c.num_real_actions, c.num_actions):
            r = int(_resolve_dummy(s, c, jnp.int32(a)))
            assert 0 <= r < c.num_real_actions, (a, r)
            assert real[r], f"dummy {a} resolved to ILLEGAL real action {r}"
    print("  every dummy resolves to a LEGAL real action (8x, 6 states)")


def test_resolution_is_spread_not_degenerate():
    """The map must actually spread over the legal reals, not pick one.

    A hash that collapsed to a single action would leave the EFFECTIVE action
    space unchanged and the whole experiment would measure nothing.
    """
    c = LiarsPokerConfig(num_players=2, hand_length=3, num_digits=3,
                         dummy_mult=16)
    hits = []
    for seed in range(40):
        s = _dealt(c, seed)
        real = np.asarray(legal_actions_mask(s, c))[:c.num_real_actions]
        n_legal = int(real.sum())
        got = {int(_resolve_dummy(s, c, jnp.int32(a)))
               for a in range(c.num_real_actions, c.num_actions)}
        hits.append(len(got) / min(n_legal, c.num_actions - c.num_real_actions))
    frac = float(np.mean(hits))
    assert frac > 0.4, f"dummies collapse onto few real actions (coverage {frac:.2f})"
    print(f"  15 dummies cover {frac:.0%} of the distinct legal reals available")


def test_game_value_and_dynamics_unchanged():
    """Playing only REAL actions must give the same trajectory at any mult.

    Dummies add strategies but must not alter the underlying game: the same
    real-action sequence from the same deal must produce the same returns.
    """
    seqs = []
    for m in (1, 2, 4, 8, 16):
        c = LiarsPokerConfig(num_players=2, hand_length=3, num_digits=3,
                             dummy_mult=m)
        s = _dealt(c, seed=3)
        guard = 0
        while not bool(is_terminal(s)) and guard < 200:
            real = np.asarray(legal_actions_mask(s, c))[:c.num_real_actions]
            a = int(np.flatnonzero(real)[-1])      # deterministic real choice
            s = apply_action(s, c, jnp.int32(a))
            guard += 1
        seqs.append((int(s.winner), int(s.loser),
                     np.asarray(jax_returns(s, c)).tolist()))
    assert all(x == seqs[0] for x in seqs), seqs
    print(f"  identical real-action play -> identical returns at every mult: {seqs[0][2]}")


def test_observation_size_is_unchanged():
    """|A| grows but the OBSERVATION must not: otherwise the network input
    changes too and the experiment confounds action count with input size."""
    sizes = []
    for m in (1, 2, 4, 16):
        c = LiarsPokerConfig(num_players=2, hand_length=3, num_digits=3,
                             dummy_mult=m)
        s = _dealt(c)
        o = observation_tensor(s, c, s.current_player, history_encoding="compact")
        sizes.append(int(np.asarray(o).size))
    assert len(set(sizes)) == 1, sizes
    print(f"  observation size constant at {sizes[0]} across multipliers")


def test_jit_and_vmap_work():
    c = LiarsPokerConfig(num_players=2, hand_length=3, num_digits=3,
                         dummy_mult=4)
    s = _dealt(c)
    batch = jax.tree_util.tree_map(lambda x: jnp.stack([x] * 5), s)
    acts = jnp.arange(c.num_real_actions, c.num_real_actions + 5, dtype=jnp.int32)
    out = jax.jit(jax.vmap(lambda st, a: apply_action(st, c, a)))(batch, acts)
    assert out.current_bid_action.shape == (5,)
    print("  apply_action with dummies is jit+vmap safe")


# ---------------------------------------------------------------------------
# dummy_mode="alias": deterministic duplicates (the isolating variant)
# ---------------------------------------------------------------------------

def _alias_cfg(mult):
    return LiarsPokerConfig(num_players=2, hand_length=3, num_digits=3,
                            dummy_mult=mult, dummy_mode="alias")


def test_alias_legal_set_scales_exactly():
    """|A_legal| must scale by EXACTLY the multiplier.

    In "random" mode dummies are always legal, so the legal set inflates
    disproportionately (7.0 -> 30.9 at nominal 2x) and the x-axis of the
    experiment does not mean what it says. Alias dummies inherit their
    target's legality, so the scaling is exact.
    """
    base = _dealt(BASE, seed=1)
    n_base = int(np.asarray(legal_actions_mask(base, BASE)).sum())
    for m in (1, 2, 4, 8, 16):
        c = _alias_cfg(m)
        s = _dealt(c, seed=1)
        n = int(np.asarray(legal_actions_mask(s, c)).sum())
        assert n == m * n_base, (m, n, m * n_base)
    print(f"  |A_legal| = {n_base} * mult exactly, for mult 1,2,4,8,16")


def test_alias_is_deterministic_and_matches_its_target():
    """Applying a dummy must be IDENTICAL to applying the action it aliases."""
    from liars_poker_jax import dummy_alias_target
    c = _alias_cfg(8)
    for seed in range(5):
        s = _dealt(c, seed)
        legal = np.asarray(legal_actions_mask(s, c))
        for a in range(c.num_real_actions, c.num_actions):
            if not legal[a]:
                continue
            t = int(dummy_alias_target(jnp.int32(a), c))
            sa = apply_action(s, c, jnp.int32(a))
            st = apply_action(s, c, jnp.int32(t))
            assert int(sa.current_bid_action) == int(st.current_bid_action)
            assert np.array_equal(np.asarray(sa.bid_history),
                                  np.asarray(st.bid_history))
            assert np.array_equal(np.asarray(sa.challenge_history),
                                  np.asarray(st.challenge_history))
            assert int(sa.num_challenges) == int(st.num_challenges)
    print("  dummy transition == its target's transition, bit-for-bit")


def test_alias_target_is_state_independent():
    """The alias map must not depend on the state, or it is not a fixed game."""
    from liars_poker_jax import dummy_alias_target
    c = _alias_cfg(4)
    tgts = []
    for seed in range(4):
        s = _dealt(c, seed)                      # state is irrelevant by design
        tgts.append([int(dummy_alias_target(jnp.int32(a), c))
                     for a in range(c.num_real_actions, c.num_actions)])
    assert all(t == tgts[0] for t in tgts), "alias map varies with state"
    # each real action duplicated exactly (mult-1) times
    from collections import Counter
    cnt = Counter(tgts[0])
    assert set(cnt.values()) == {c.dummy_mult - 1}, cnt
    print(f"  alias map is state-independent; every real action has exactly "
          f"{c.dummy_mult - 1} duplicates")


def test_alias_preserves_returns():
    """Real-action-only play must give identical returns at every multiplier."""
    seqs = []
    for m in (1, 2, 4, 16):
        c = _alias_cfg(m)
        s = _dealt(c, seed=3)
        g = 0
        while not bool(is_terminal(s)) and g < 200:
            real = np.asarray(legal_actions_mask(s, c))[:c.num_real_actions]
            s = apply_action(s, c, jnp.int32(np.flatnonzero(real)[-1])); g += 1
        seqs.append(np.asarray(jax_returns(s, c)).tolist())
    assert all(x == seqs[0] for x in seqs), seqs
    print(f"  identical real play -> identical returns at every mult: {seqs[0]}")


def test_alias_adds_no_stochasticity():
    """Repeated application of the same dummy must give the SAME next state.

    This is the property "random" mode lacks and the whole reason alias exists.
    """
    c = _alias_cfg(8)
    s = _dealt(c, seed=2)
    legal = np.asarray(legal_actions_mask(s, c))
    d = int(np.flatnonzero(legal[c.num_real_actions:])[0]) + c.num_real_actions
    outs = {int(apply_action(s, c, jnp.int32(d)).current_bid_action)
            for _ in range(8)}
    assert len(outs) == 1, outs
    # and contrast: "random" mode is state-hashed, so it is also deterministic
    # per (state, action) -- the stochasticity there is ACROSS states, which is
    # what injects return noise. Documented, not asserted.
    print(f"  alias transition is deterministic (single outcome {outs})")


# ---------------------------------------------------------------------------
# dummy_mode="lose": taking a dummy instantly loses (the costly-dummy probe)
# ---------------------------------------------------------------------------

def _lose_cfg(mult, players=2):
    return LiarsPokerConfig(num_players=players, hand_length=3, num_digits=3,
                            dummy_mult=mult, dummy_mode="lose")


def test_lose_terminates_with_taker_as_loser():
    c = _lose_cfg(4)
    for seed in range(5):
        s = _dealt(c, seed)
        taker = int(s.current_player)
        legal = np.asarray(legal_actions_mask(s, c))
        d = int(np.flatnonzero(legal[c.num_real_actions:])[0]) + c.num_real_actions
        ns = apply_action(s, c, jnp.int32(d))
        assert bool(is_terminal(ns)), "taking a dummy must end the game"
        r = np.asarray(jax_returns(ns, c))
        assert r[taker] == -(c.num_players - 1), (taker, r)
        assert abs(float(r.sum())) < 1e-6, f"not zero-sum: {r}"
        assert all(r[p] == 1.0 for p in range(c.num_players) if p != taker), r
    print("  dummy -> immediate terminal; taker gets -(N-1), others +1, sum 0")


def test_lose_zero_sum_at_three_players():
    c = _lose_cfg(4, players=3)
    s = _dealt(c, 1)
    taker = int(s.current_player)
    legal = np.asarray(legal_actions_mask(s, c))
    d = int(np.flatnonzero(legal[c.num_real_actions:])[0]) + c.num_real_actions
    r = np.asarray(jax_returns(apply_action(s, c, jnp.int32(d)), c))
    assert r[taker] == -2.0 and abs(float(r.sum())) < 1e-6, r
    print(f"  3 players: taker {r[taker]:+.0f}, others +1 each, sum {r.sum():+.0f}")


def test_lose_real_play_is_untouched():
    """Real actions must behave exactly as in the unmodified game."""
    seqs = []
    for m in (1, 2, 4, 16):
        c = _lose_cfg(m)
        s = _dealt(c, seed=3)
        g = 0
        while not bool(is_terminal(s)) and g < 200:
            real = np.asarray(legal_actions_mask(s, c))[:c.num_real_actions]
            s = apply_action(s, c, jnp.int32(np.flatnonzero(real)[-1])); g += 1
        seqs.append(np.asarray(jax_returns(s, c)).tolist())
    assert all(x == seqs[0] for x in seqs), seqs
    print(f"  real-action-only play identical at every mult: {seqs[0]}")


def test_lose_legal_set_scales_exactly():
    base = int(np.asarray(legal_actions_mask(_dealt(BASE, 1), BASE)).sum())
    for m in (1, 2, 4, 8, 16):
        c = _lose_cfg(m)
        n = int(np.asarray(legal_actions_mask(_dealt(c, 1), c)).sum())
        assert n == m * base, (m, n, m * base)
    print(f"  |A_legal| = {base} * mult exactly (same rule as alias)")


def test_lose_is_deterministic():
    c = _lose_cfg(8)
    s = _dealt(c, 2)
    legal = np.asarray(legal_actions_mask(s, c))
    d = int(np.flatnonzero(legal[c.num_real_actions:])[0]) + c.num_real_actions
    outs = {tuple(np.asarray(jax_returns(apply_action(s, c, jnp.int32(d)), c)))
            for _ in range(8)}
    assert len(outs) == 1, outs
    print(f"  deterministic outcome {outs}")


def test_lose_is_strictly_dominated():
    """Every dummy must be no better than the WORST real outcome.

    -(N-1) is the minimum possible payoff in this game, so a dummy can never be
    part of a best response and the equilibrium set is preserved.
    """
    c = _lose_cfg(4)
    s = _dealt(c, 0)
    taker = int(s.current_player)
    legal = np.asarray(legal_actions_mask(s, c))
    dummy_r = []
    for a in range(c.num_real_actions, c.num_actions):
        if legal[a]:
            dummy_r.append(float(np.asarray(
                jax_returns(apply_action(s, c, jnp.int32(a)), c))[taker]))
    assert dummy_r and all(r == -(c.num_players - 1) for r in dummy_r), set(dummy_r)
    print(f"  all {len(dummy_r)} legal dummies pay the taker "
          f"{-(c.num_players-1)} (the game minimum)")


def test_cap_to_game_map_is_identity_with_dummies():
    """REGRESSION: the capacity map must not remap dummy actions.

    `cap_to_game_action_map` decodes each action index as a bid and sends
    anything unreachable to 0 (CHALLENGE), on the assumption that unreachable
    actions are always illegal and never sampled. dummy_mult>1 breaks that:
    dummies sit above the real bid range, are LEGAL, and ARE sampled -- so every
    one of them was silently played as CHALLENGE, and every dummy-probe
    experiment measured extra outputs that were aliases of CHALLENGE.

    The rollout applies `self._cap_to_game[action]` before `apply_action`
    (train_jax `_env_actions`), so this is the join point where the probe was
    being erased.
    """
    from liars_poker_jax import cap_to_game_action_map
    for mult in (1, 2, 4, 8, 16):
        for mode in ("random", "alias", "lose"):
            c = LiarsPokerConfig(num_players=2, hand_length=3, num_digits=3,
                                 dummy_mult=mult, dummy_mode=mode)
            mp = np.asarray(cap_to_game_action_map(c, c))
            assert np.array_equal(mp, np.arange(c.num_actions)), (
                mult, mode, "dummy actions are being remapped",
                sorted(set(mp[c.num_real_actions:].tolist()))[:5])
    # and the genuine capacity path must STILL remap (net wider than the game)
    g = LiarsPokerConfig(num_players=2, hand_length=3, num_digits=3)
    nc = LiarsPokerConfig(num_players=2, hand_length=3, num_digits=10)
    mp = np.asarray(cap_to_game_action_map(g, nc))
    assert len(mp) == nc.num_actions and not np.array_equal(
        mp, np.arange(len(mp))), "capacity remapping was broken by the fix"
    print("  cap_to_game is identity for every dummy_mult/mode; "
          "capacity remapping preserved")


if __name__ == "__main__":
    for fn in (test_action_count_scales_exactly,
               test_mult1_is_byte_identical,
               test_real_action_prefix_unchanged,
               test_dummies_legal_exactly_when_a_real_action_is,
               test_dummy_resolves_to_a_legal_real_action,
               test_resolution_is_spread_not_degenerate,
               test_game_value_and_dynamics_unchanged,
               test_observation_size_is_unchanged,
               test_jit_and_vmap_work,
               test_alias_legal_set_scales_exactly,
               test_alias_is_deterministic_and_matches_its_target,
               test_alias_target_is_state_independent,
               test_alias_preserves_returns,
               test_alias_adds_no_stochasticity,
               test_lose_terminates_with_taker_as_loser,
               test_lose_zero_sum_at_three_players,
               test_lose_real_play_is_untouched,
               test_lose_legal_set_scales_exactly,
               test_lose_is_deterministic,
               test_lose_is_strictly_dominated,
               test_cap_to_game_map_is_identity_with_dummies):
        print(fn.__name__); fn()
    print("\nALL DUMMY-ACTION TESTS PASSED")
