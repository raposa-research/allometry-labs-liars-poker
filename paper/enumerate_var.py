"""EXACT enumeration of the action-draw randomness in the NeuRD and PPO updates.

The Monte-Carlo probe estimates tr(Cov) from 24 sampled batches, so it inherits
sampling error in exactly the dimension that creates the 1/mu tail -- which
action got drawn. That dimension is finite: |A_legal| outcomes with known
probabilities pi_a. Here it is SUMMED instead of sampled, giving the exact
first and second moment of the per-step update.

Algebra (on-policy, so mu == pi). Write u_a = v + eta_a with
eta_a = -eta*log(pi_a/pi_reg_a), and delta_a = (G~_a - v) the return residual
if a were the drawn action. v-trace scatters delta only into the drawn slot:

    q_vr(a) = u_a + 1{a=t} * delta_t / pi_t          (t = drawn action)
    baseline = sum_b pi_b q_vr(b) = ubar + delta_t
    adv(a|t) = (u_a - ubar) - delta_t + 1{a=t} * delta_t / pi_t

so the NeuRD per-logit update vector is an explicit function of t. PPO's is

    A_t = eta_t + delta_t ,   c(a|t) = A_t * (1{a=t} - pi_a)

Exact moments over t ~ pi:  E[x] = sum_t pi_t x(.|t),
                            tr(Cov) = sum_t pi_t ||x(.|t)||^2 - ||E[x]||^2.

delta_a:
  * DUMMY actions in `lose` mode -- EXACT. The action terminates the episode
    with the taker losing, so G~ = -1 and delta_a = -1 - v, no estimation.
  * REAL actions -- estimated per (state, action) from the visits where that
    action was actually drawn, recovered from the observed q_vr via
    delta = (q_vr(a_t) - u_{a_t}) * pi_{a_t}. Never-drawn real actions get 0.
    Real actions carry little of the tail (pi ~ 0.08 vs ~0.003), so this
    approximation is far less load-bearing than the exact dummy branch.

Steps are independent given the states, so per-state moments are summed.
"""
import collections, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import jax.numpy as jnp, numpy as np
from update_variance import _build, _read_meta
import rnad as R
from ckpt_root import require  # checkpoint location; see REPRODUCE.md section 3

ALPHA = 1.0


def gather(run, step, batches=24, bs=256, traj=30):
    path = require(run, step)
    s = _build(_read_meta(path), "neurd", bs, traj, "sparse")
    n_real = s._jax_config.num_real_actions
    eta = s.config.eta_reward_transform
    G = collections.defaultdict(list)
    for _ in range(batches):
        ts = s.collect_batch_trajectory()
        (pi, v, log_pi, logit, v_t, lpp, lpp_) = s._loss_rollouts(
            s.params, s.params_target, s.params_prev, s.params_prev_, ts)
        pproc = s.config.finetune(pi, ts.env.legal, 10**9)
        lpr = log_pi - (ALPHA * lpp + (1 - ALPHA) * lpp_)
        for player in range(s._game.num_players()):
            _, _, q = R.v_trace(
                v_t, ts.env.valid, ts.env.player_id, ts.actor.policy, pproc, lpr,
                R._player_others(ts.env.player_id, ts.env.valid, player,
                                 s._opp_reg_weight()),
                ts.actor.action_oh, ts.actor.rewards[:, :, player], player,
                lambda_=1.0, c=s.config.c_vtrace, rho=np.inf, eta=eta,
                opp_reg_weight=s._opp_reg_weight())
            cen = logit - jnp.mean(logit * ts.env.legal, axis=-1, keepdims=True)
            m = np.asarray(ts.env.valid & (ts.env.player_id == player))
            if not m.any():
                continue
            obs = np.asarray(ts.env.obs)[m]
            P = np.asarray(pi)[m]; Q = np.asarray(q)[m]
            V = np.asarray(v)[m].reshape(-1)
            E = -eta * np.asarray(lpr)[m]              # eta_a
            C = np.asarray(cen)[m]; L = np.asarray(ts.env.legal)[m]
            S = np.asarray(ts.actor.action_oh)[m].astype(bool)
            for i in range(obs.shape[0]):
                G[obs[i].tobytes()].append((P[i], Q[i], V[i], E[i], C[i], L[i], S[i]))
    return G, n_real, eta


def exact_moments(G, n_real, beta, min_visits, label):
    KEYS = ("neurd_unclipped", "neurd_clipped", "ppo")
    per_state = {k: [] for k in KEYS}      # per-state (trcov, ||E||^2, n_legal)
    n_states = 0
    for _, vs in G.items():
        if len(vs) < min_visits:
            continue
        n_states += 1
        P = np.stack([x[0] for x in vs]).mean(0)
        Q = np.stack([x[1] for x in vs])
        V = float(np.mean([x[2] for x in vs]))
        Eta = np.stack([x[3] for x in vs]).mean(0)
        Cen = np.stack([x[4] for x in vs]).mean(0)
        L = vs[0][5].astype(bool)
        Sm = np.stack([x[6] for x in vs])

        idx = np.where(L)[0]
        pi = P[idx]; pi = pi / max(pi.sum(), 1e-12)
        u = V + Eta[idx]
        cen = Cen[idx]
        is_d = idx >= n_real

        # delta_a
        delta = np.zeros(len(idx))
        delta[is_d] = -1.0 - V                              # EXACT for lose dummies
        for j, a in enumerate(idx):
            if is_d[j]:
                continue
            hit = Sm[:, a]
            if hit.any():
                delta[j] = float(np.mean((Q[hit, a] - u[j]) * max(P[a], 1e-12)))

        ubar = float(pi @ u)
        cu = u - ubar
        # enumerate the drawn action t
        Xn = np.empty((len(idx), len(idx)))   # [t, a] neurd unclipped
        Xp = np.empty((len(idx), len(idx)))   # [t, a] ppo
        for t in range(len(idx)):
            Xn[t] = cu - delta[t]
            Xn[t, t] += delta[t] / max(pi[t], 1e-12)
            A_t = Eta[idx][t] + delta[t]
            Xp[t] = -A_t * pi
            Xp[t, t] += A_t
        Xc = np.where(cen[None, :] > -beta, np.minimum(Xn, 0.0), 0.0) + \
             np.where(cen[None, :] < beta, np.maximum(Xn, 0.0), 0.0)

        for key, X in (("neurd_unclipped", Xn), ("neurd_clipped", Xc), ("ppo", Xp)):
            Ex = pi @ X                                   # [a]
            second = float(pi @ (X * X).sum(axis=1))      # E||x||^2
            nE2 = float(Ex @ Ex)
            per_state[key].append((second - nE2, nE2, len(idx)))

    print(f"\n{'='*86}\n{label}   ({n_states} infostates with >={min_visits} visits, "
          f"exact over the action draw)\n{'='*86}")
    hdr = (f"{'update':18s}{'mean ||E||^2':>14s}{'mean tr(Cov)':>14s}"
           f"{'pooled relvar':>15s}{'median relvar':>15s}{'vs PPO':>9s}")
    print(hdr); print("-" * len(hdr))
    out, pooled, med = {}, {}, {}
    for k in KEYS:
        tc = np.array([x[0] for x in per_state[k]])
        n2 = np.array([x[1] for x in per_state[k]])
        pooled[k] = float(tc.sum() / max(n2.sum(), 1e-300))   # sum tr(Cov)/sum||E||^2
        med[k] = float(np.median(tc / np.maximum(n2, 1e-300)))
        out[k] = {"mean_normE2": float(n2.mean()), "mean_trcov": float(tc.mean()),
                  "pooled_rel_variance": pooled[k], "median_rel_variance": med[k]}
    for k in KEYS:
        print(f"{k:18s}{out[k]['mean_normE2']:>14.5g}{out[k]['mean_trcov']:>14.5g}"
              f"{pooled[k]:>15.5g}{med[k]:>15.5g}"
              f"{pooled[k]/max(pooled['ppo'],1e-300):>8.2f}x")
    out["n_states"] = n_states
    return out


if __name__ == "__main__":
    res = {}
    for run, step, beta, label in [
            ("lose-fix-3x3x2-neurd-x8", 2000000, 2.0, "lose-8x, NeuRD arm @2M"),
            ("dummy-3x3x2-neurd-x1", 2000000, 2.0, "1x, NeuRD arm @2M")]:
        try:
            G, n_real, eta = gather(run, step)
            res[label] = exact_moments(G, n_real, beta, 12, label)
        except Exception as e:
            import traceback; traceback.print_exc(); res[label] = {"error": str(e)}
    pathlib.Path(__file__).with_name("enumerate_var.json").write_text(json.dumps(res, indent=1))
    print("\nwrote enumerate_var.json")
