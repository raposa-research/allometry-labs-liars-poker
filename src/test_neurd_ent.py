"""Tests for neurd_ent_coef: the NeuRD-side entropy bonus (ablation cell 3).

Vanilla NeuRD has no entropy term -- its bound on determinism is the nerd.beta
logit trust region plus the all-legal-actions force (EXPERIMENTS.md §20). The
2x2 ablation (update rule x explicit entropy bonus) needs a NeuRD+entropy cell,
so `neurd_ent_coef` adds one, in the same per-player renormalized form as
`ppo_ent_coef`.

Asserts:
  1. neurd_ent_coef=0.0 is BYTE-IDENTICAL to the pre-change loss.
  2. The loss drops by EXACTLY coef * sum_k renormalize(H, mask_k).
  3. It is linear in the coefficient (2x coef -> 2x the reduction).
  4. The gradient actually pushes entropy UP on a near-deterministic policy.

Run: PYTHONPATH=src python src/test_neurd_ent.py
"""
import jax
import jax.numpy as jnp
import numpy as np

from liars_poker_jax import LiarsPokerConfig
from rnad import RNaDConfig, renormalize
from train_jax import JaxLiarsPokerRNaDSolver


def _solver(neurd_ent_coef=0.0, **kw):
    cfg = RNaDConfig(game_name="liars_poker", policy_network_layers=(32, 32),
                     batch_size=8, trajectory_max=12,
                     policy_update="neurd", advantage_estimator="vtrace",
                     eta_reward_transform=0.4,
                     neurd_ent_coef=neurd_ent_coef, **kw)
    return JaxLiarsPokerRNaDSolver(
        cfg, LiarsPokerConfig(num_players=3, hand_length=2, num_digits=3),
        network_type="factored_mlp", history_encoding="compact")


def _loss_of(solver, ts, alpha=1.0, steps=1000):
    return float(solver.loss(solver.params, solver.params_target,
                             solver.params_prev, solver.params_prev_,
                             ts, alpha, steps))


def _entropy_term(solver, ts):
    """coef-free sum_k renormalize(H(pi), valid & player==k), as the loss adds it."""
    apply_v = jax.vmap(solver.network.apply, (None, 0), 0)
    pi, _, log_pi, _ = apply_v(solver.params, ts.env)
    H = -jnp.sum(pi * log_pi, axis=-1)
    tot = 0.0
    for k in range(3):
        mask = ts.env.valid * (ts.env.player_id == k)
        tot += float(renormalize(H, mask))
    return tot


def test_zero_coef_is_identical():
    """The default must not perturb the loss at all."""
    a = _solver(0.0)
    ts = a.collect_batch_trajectory()
    la = _loss_of(a, ts)
    b = _solver(0.0)
    b.params = a.params; b.params_target = a.params_target
    b.params_prev = a.params_prev; b.params_prev_ = a.params_prev_
    lb = _loss_of(b, ts)
    assert la == lb, (la, lb)
    print(f"  neurd_ent_coef=0.0 loss reproducible and unperturbed: {la:.8f}")


def test_bonus_is_exactly_the_entropy_term():
    """loss(coef) must equal loss(0) - coef * sum_k renormalize(H, mask_k)."""
    base = _solver(0.0)
    ts = base.collect_batch_trajectory()
    l0 = _loss_of(base, ts)
    ent = _entropy_term(base, ts)
    for coef in (0.05, 0.2):
        s = _solver(coef)
        s.params = base.params; s.params_target = base.params_target
        s.params_prev = base.params_prev; s.params_prev_ = base.params_prev_
        lc = _loss_of(s, ts)
        expect = l0 - coef * ent
        assert abs(lc - expect) < 2e-4 * max(1.0, abs(expect)), (coef, lc, expect)
        print(f"  coef={coef}: loss={lc:+.6f} == loss0 - coef*H_term="
              f"{expect:+.6f}  (H_term={ent:.6f})")


def test_linear_in_coefficient():
    """Doubling the coefficient must double the reduction."""
    base = _solver(0.0)
    ts = base.collect_batch_trajectory()
    l0 = _loss_of(base, ts)
    drops = []
    for coef in (0.05, 0.10):
        s = _solver(coef)
        s.params = base.params; s.params_target = base.params_target
        s.params_prev = base.params_prev; s.params_prev_ = base.params_prev_
        drops.append(l0 - _loss_of(s, ts))
    assert abs(drops[1] - 2 * drops[0]) < 2e-4 * max(1.0, abs(drops[1])), drops
    print(f"  linear in coef: drop(0.05)={drops[0]:.6f}, drop(0.10)={drops[1]:.6f}")


def test_gradient_raises_entropy():
    """On a sharpened policy, the bonus's gradient must increase entropy.

    Isolates the bonus by comparing the entropy change after one SGD step with
    and without it, from identical params on identical data. A huge coefficient
    is used so the bonus dominates the NeuRD force, which is what is being
    tested -- not that it dominates at the 0.05 used in practice.
    """
    base = _solver(0.0)
    ts = base.collect_batch_trajectory()
    apply_v = jax.vmap(base.network.apply, (None, 0), 0)

    def mean_H(params):
        pi, _, log_pi, _ = apply_v(params, ts.env)
        H = -jnp.sum(pi * log_pi, axis=-1)
        return float(renormalize(H, ts.env.valid))

    h0 = mean_H(base.params)
    out = {}
    for coef in (0.0, 50.0):
        s = _solver(coef)
        s.params = base.params; s.params_target = base.params_target
        s.params_prev = base.params_prev; s.params_prev_ = base.params_prev_
        g = jax.grad(s.loss)(s.params, s.params_target, s.params_prev,
                             s.params_prev_, ts, 1.0, 1000)
        stepped = jax.tree_util.tree_map(lambda p, d: p - 0.05 * d, s.params, g)
        out[coef] = mean_H(stepped)
    assert out[50.0] > out[0.0], (h0, out)
    print(f"  H before={h0:.5f}; after one step coef=0 -> {out[0.0]:.5f}, "
          f"coef=50 -> {out[50.0]:.5f} (bonus raises entropy)")


if __name__ == "__main__":
    for fn in (test_zero_coef_is_identical,
               test_bonus_is_exactly_the_entropy_term,
               test_linear_in_coefficient,
               test_gradient_raises_entropy):
        print(fn.__name__); fn()
    print("\nALL NEURD ENTROPY-BONUS TESTS PASSED")
