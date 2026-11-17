# Reproducing the experiments

Commands to regenerate the measured results in *Liar's Poker*. Each section
covers one figure or table and states the value you should get.

Everything needed is in this repository; there is no second checkout to set up.

## 1. What this covers

| Paper item | Section | Cost |
|---|---|---|
| Table 3 — game sizes | §4.1 | seconds, CPU |
| Exact exploitability, 3x3 2-player (`$0.169` / `$0.046`) | §4.2 | ~90 min, CPU |
| Figures 2a and 5 — dummy-action ladder | §4.3 | 12 training runs + exact BRs |
| Figure 2b — update variance | §4.4 | ~20 min, CPU |
| Figure 3 — advantage vs probability | §4.5 | ~40 min, CPU |
| Table 1 — agents and learned best responses | §4.6 | hours–days, GPU |
| 8x10 3-player ablation | §4.7 | as §4.6 |

## 2. Setup

```bash
python3.12 -m venv .venv                          # tested with Python 3.12
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install 'jax[cuda12]==0.9.2'       # GPU; omit for CPU
.venv/bin/pip install exact_br/liars_poker_eas   # exact-BR extension
```

`exact_br/liars_poker_eas` is a pybind11 + OpenMP C++ extension and needs CMake
and a C++17 toolchain. See `exact_br/README.md`.

Check the install:

```bash
PYTHONPATH=src .venv/bin/python -m pytest src/test_dummy_actions.py src/test_neurd_ent.py -q
.venv/bin/python -m pytest exact_br/tests/test_liars_poker_solver.py -v   # needs OpenSpiel
.venv/bin/python -m pytest exact_br/tests/test_liars_poker_dummy.py -v
```

Expect 25 passed from the first command, including a check that `dummy_mult=1`
is byte-identical to the unmodified game.

## 3. Training the checkpoints

**Trained checkpoints are not distributed.** The one exception is
`experiments/3x3_v3/agent_1650000.pickle` (Solly-A, 3x3 2-player), which is in
the repository — so §4.1 and §4.2 need nothing else.

Everything else must be trained first, using the recipes below. Train into
`checkpoints/<run-name>/`, which is where the analysis scripts look by default;
set `LP_CKPT_ROOT` to use another directory. Each script fails with the run
name and a pointer back here if a checkpoint is missing.

All agents below run for 2M steps (`num_phases=20000 rnad_steps_per_phase=100`)
and share `advantage_estimator=vtrace`, `batch_size=256`.

### 3.1 3x3 2-player agents (Figures 2a, 2b, 3, 5)

```bash
PYTHONPATH=src .venv/bin/python -u src/train_jax.py \
  num_digits=3 hand_length=3 num_players=2 \
  network_type=mlp 'rnad_config.policy_network_layers=[256,256,128]' \
  rnad_config.advantage_estimator=vtrace rnad_config.eta_reward_transform=0.4 \
  rnad_config.batch_size=256 \
  num_phases=20000 rnad_steps_per_phase=100 \
  <VARIANT> \
  save_dir=checkpoints/<run-name>/
```

`<VARIANT>` per run. Multiplier-1 runs take no `dummy_*` flags at all;
every other run adds `dummy_mult=<M> dummy_mode=lose`.

| run name | `<VARIANT>` |
|---|---|
| `dummy-3x3x2-neurd-x1` | `rnad_config.policy_update=neurd` |
| `dummy-3x3x2-ppoent-x1` | `rnad_config.policy_update=ppo rnad_config.ppo_ent_coef=0.05` |
| `x1-pg-noent` | `rnad_config.policy_update=ppo rnad_config.ppo_ent_coef=0.0` |
| `lose-s-neurd-b2-x2` | `dummy_mult=2 dummy_mode=lose rnad_config.policy_update=neurd` |
| `lose-s-neurd-b2-x4` | `dummy_mult=4 dummy_mode=lose rnad_config.policy_update=neurd` |
| `lose-fix-3x3x2-neurd-x8` | `dummy_mult=8 dummy_mode=lose rnad_config.policy_update=neurd` |
| `lose-s-ppoent-x2` | `dummy_mult=2 dummy_mode=lose rnad_config.policy_update=ppo rnad_config.ppo_ent_coef=0.05` |
| `lose-s-ppoent-x4` | `dummy_mult=4 dummy_mode=lose rnad_config.policy_update=ppo rnad_config.ppo_ent_coef=0.05` |
| `lose-fix-3x3x2-ppoent-x8` | `dummy_mult=8 dummy_mode=lose rnad_config.policy_update=ppo rnad_config.ppo_ent_coef=0.05` |
| `lose2-pg-noent` | `dummy_mult=2 dummy_mode=lose rnad_config.policy_update=ppo rnad_config.ppo_ent_coef=0.0` |
| `lose4-pg-noent` | `dummy_mult=4 dummy_mode=lose rnad_config.policy_update=ppo rnad_config.ppo_ent_coef=0.0` |
| `lose8-pg-noent` | `dummy_mult=8 dummy_mode=lose rnad_config.policy_update=ppo rnad_config.ppo_ent_coef=0.0` |

NeuRD uses the default `nerd_beta` of 2.0. `paper/enumerate_all.py` and
`paper/total_var.py` additionally reference β-adjusted variants, which hold the
retained tail mass constant as the action space grows
(`β_m = 2.0 + 0.5·ln((n_m−1)/(n_1−1))`):

| run name | `<VARIANT>` |
|---|---|
| `lose-s-neurd-badj-x2` | `dummy_mult=2 dummy_mode=lose rnad_config.policy_update=neurd rnad_config.nerd_beta=2.385` |
| `lose-s-neurd-badj-x4` | `dummy_mult=4 dummy_mode=lose rnad_config.policy_update=neurd rnad_config.nerd_beta=2.749` |
| `lose-s-neurd-badj-x8` | `dummy_mult=8 dummy_mode=lose rnad_config.policy_update=neurd rnad_config.nerd_beta=3.105` |

### 3.2 8x10 agents (Table 1, §4.7 ablation)

```bash
PYTHONPATH=src .venv/bin/python -u src/train_jax.py \
  num_digits=10 hand_length=8 num_players=<N> \
  network_type=factored_mlp history_encoding=compact \
  'rnad_config.policy_network_layers=[512,512,512,256,256]' \
  rnad_config.advantage_estimator=vtrace \
  rnad_config.batch_size=256 rnad_config.warmup_steps=2000 \
  rnad_config.trajectory_max=<T> \
  num_phases=20000 rnad_steps_per_phase=100 \
  <VARIANT> \
  save_dir=checkpoints/<run-name>/
```

`<T>` is 100 for 3-player and 170 for 5-player.

| run name | N | `<VARIANT>` |
|---|---|---|
| `rnad-ppo-vtrace-8x10x3-ent-eta0` (Solly-B 3p) | 3 | `rnad_config.policy_update=ppo rnad_config.ppo_ent_coef=0.05 rnad_config.eta_reward_transform=0.0` |
| `rnad-ppo-vtrace-8x10x5-ent-eta0` (Solly-B 5p) | 5 | `rnad_config.policy_update=ppo rnad_config.ppo_ent_coef=0.05 rnad_config.eta_reward_transform=0.0` |
| `rnad-neurd-vtrace-8x10x3` (vanilla R-NaD) | 3 | `rnad_config.policy_update=neurd rnad_config.eta_reward_transform=0.4 rnad_config.zero_sum_reg_scaling=true` |
| `abl-8x10x3-ppo-ent-eta04` | 3 | `rnad_config.policy_update=ppo rnad_config.ppo_ent_coef=0.05 rnad_config.eta_reward_transform=0.4` |
| `abl-8x10x3-ppo-noent-eta04` | 3 | `rnad_config.policy_update=ppo rnad_config.ppo_ent_coef=0.0 rnad_config.eta_reward_transform=0.4` |

About 15 hours each on one A100-80GB.

### 3.3 Solly-A (reduced formats)

Solly-A was trained with OpenSpiel `python_liars_poker` via `src/train.py`, not
with `train_jax.py`:

```bash
PYTHONPATH=src .venv/bin/python src/train.py \
  --num_digits 3 --hand_length 3 \
  --training_steps 5000000 --checkpoint_frequency 10000 \
  --save_dir checkpoints-3x3_v3
```

Use `--num_digits 5 --hand_length 5` for 5x5 2-player and `--num_players 3` for
3x3 3-player. The published agents are the 1.65M, 3.2M and 9.3M step
checkpoints; remaining settings are `rnad.py` defaults. The 3x3 2-player
checkpoint ships with the repository, so only the other two need training.

`src/train.py` writes `.pickle`, which must be converted before use with the
JAX tooling:

```bash
PYTHONPATH=src .venv/bin/python src/convert_checkpoint.py agent_3x3x2_1650000.pickle
```

This sets `hand_encoding="raw"`, which matters: OpenSpiel writes raw hand
digits while this codebase defaults to a per-digit histogram, and the two have
the same observation length when `hand_length == num_digits`. A mismatch does
not raise — the agent simply stops seeing its own hand.

### 3.4 Best responses

Trained per §4.6 from the agents above, into `checkpoints/<br-name>/`.

## 4. Experiments

### 4.1 Table 3 — game sizes

```bash
.venv/bin/python paper/game_size.py
```

Prints `B`, hands, decision histories, information sets and game states for all
five configurations. Reproduces the published table exactly, including
`1.7e306` at 8x10 5-player.

### 4.2 Exact exploitability, 3x3 2-player

```bash
.venv/bin/python exact_br/eval_liars_poker_checkpoint.py \
  experiments/3x3_v3/agent_1650000.pickle --finetune
```

Expect `avg_score_response = 0.16955`, i.e. **$0.169 per hand** for Solly-A.
Runtime ~86 minutes on CPU; faster with `jax[cuda12]` and a larger
`--batch-size` (default 262,144 rows).

For the Solly-B recipe on the same game, point it at `dummy-3x3x2-ppoent-x1`
and **drop `--finetune`** (expect **$0.046**):

```bash
.venv/bin/python exact_br/eval_liars_poker_checkpoint.py \
  checkpoints/dummy-3x3x2-ppoent-x1/agent_2000000.msgpack
```

**`--finetune` applies to Solly-A only.** It imposes the 3% probability floor
and 32-value grid that Solly-A uses at inference; Solly-B plays unfiltered.

### 4.3 Figures 2a and 5 — dummy-action ladder

Twelve agents: three recipes × multipliers {1, 2, 4, 8}, all 3x3 2-player with
`dummy_mode=lose`.

| multiplier | NeuRD | Softmax PG | Softmax PG + entropy |
|---|---|---|---|
| 1 | Solly-A (`experiments/3x3_v3/agent_1650000.pickle`) | `x1-pg-noent` | `dummy-3x3x2-ppoent-x1` |
| 2 | `lose-s-neurd-b2-x2` | `lose2-pg-noent` | `lose-s-ppoent-x2` |
| 4 | `lose-s-neurd-b2-x4` | `lose4-pg-noent` | `lose-s-ppoent-x4` |
| 8 | `lose-fix-3x3x2-neurd-x8` | `lose8-pg-noent` | `lose-fix-3x3x2-ppoent-x8` |

The 1× NeuRD point is Solly-A, used in place of a multiplier-1 training run.

Train these with the recipes in §3.1.

Use `network_type=mlp`. The factored head derives per-action features from
`decode_bid`, which has no meaning for dummy actions.

Figure 2a is the exact exploitability of each final checkpoint (§4.2). Figure 5
adds the learned best response for each (§4.6) on the same axes.

### 4.4 Figure 2b — update variance

```bash
.venv/bin/python paper/mass_sweep.py        # writes mass_sweep.csv
.venv/bin/python paper/plot_mass_sweep.py
```

Embeds the 1× NeuRD policy in a mult-× `lose` action space and moves a
controlled fraction of probability mass onto dominated actions, holding the
real-action strategy, state distribution and per-action returns fixed. Reports
relative variance `tr(Cov)/‖E[Δ]‖²` of the logit-space update for NeuRD and
softmax PG at multipliers 2, 4 and 8.

The supporting scripts `paper/enumerate_var.py`, `paper/enumerate_all.py` and
`paper/total_var.py` compute the action-draw moments in closed form and split
the variance into within-state and between-state terms.

### 4.5 Figure 3 — advantage vs probability

```bash
.venv/bin/python paper/forced_opening_adv.py   # writes forced_opening_adv_se.json
.venv/bin/python paper/sig_advantage.py        # significance, BH-corrected
.venv/bin/python paper/plot_a_vs_pi.py         # the figure
```

Uses `dummy-3x3x2-neurd-x1`, `x1-pg-noent` and `dummy-3x3x2-ppoent-x1`. Each
opening bid is forced and self-play is rolled out from there, because the bids
of interest have π ≈ 0 and are never sampled on-policy.

Expect six bids with π between 1e−7 and 1e−4 to have significantly positive
advantage for softmax PG without the entropy bonus, and none for the other two
recipes.

### 4.6 Table 1 — learned best responses

Train the agents first: Solly-A per §3.3, Solly-B per §3.2.

**Learned best response**:

```bash
PYTHONPATH=src .venv/bin/python -u src/train_br.py \
  checkpoints/<AGENT>/agent_2000000.msgpack \
  --br-network-type factored_mlp --br-layers 512 512 512 256 256 \
  --br-init-from checkpoints/<AGENT>/agent_2000000.msgpack \
  --opp-threshold 0 --opp-discretization 0 \
  --lr 3e-5 --ent-coef 0.01 --n-epochs 1 --clip-epsilon 0.2 \
  --batch-size 256 --trajectory-max 100 \
  --n-cycles 4000 --eval-every 25 --eval-episodes 8192 \
  --checkpoint-every 500 --no-wandb \
  --save-dir checkpoints/<BR>/
```

For the reduced-format games use `--br-network-type mlp --br-layers 512 512 256
--lr 1e-4 --n-epochs 2 --batch-size 512 --trajectory-max 60 --n-cycles 3000`
and omit `--br-init-from`.

**Scoring** (Table 1 uses 50,000 hands per seat):

```bash
PYTHONPATH=src .venv/bin/python -u src/eval_br_vs_agent.py \
  --agent checkpoints/<AGENT>/agent_2000000.msgpack \
  --br-dir checkpoints/<BR>/ --step 4000 \
  --episodes 50000 --batch-size 1024 --trajectory-max 100 \
  --br-network-type factored_mlp --br-layers 512 512 512 256 256 \
  --opp-threshold 0 --opp-discretization 0 \
  --outcome-stats
```

See §5 for reading the output.

### 4.7 8x10 3-player ablation

| cell | agent | published |
|---|---|---|
| vanilla R-NaD | `rnad-neurd-vtrace-8x10x3` | −$0.50 |
| softmax PG, no bonus | `abl-8x10x3-ppo-noent-eta04` | −$0.418 |
| softmax PG + bonus | `abl-8x10x3-ppo-ent-eta04` | −$0.053 |

The last two are post-hoc best responses as in §4.6. The first is the
in-training interleaved best response, which reaches `reward_p0/p1/p2 = +1.000`
each, giving `agent_score = −1.000/(N−1) = −0.500`.

## 5. Reading best-response output

`eval_br_vs_agent.py` prints a per-seat reward and a pooled `agent_score`.

- **Check every seat is positive.** A best response is only a valid
  exploitability bound if it beats copying the agent in each seat; a negative
  seat means it underperforms imitation there and the pooled figure is not a
  bound.
- **A positive `agent_score` is a failure**, not an unexploitable agent — it can
  only happen when the best response is net worse than its target.
- **Pass `--br-layers` explicitly.** Checkpoints record the frozen agent's layer
  widths, not the best response's, so the default rebuilds the wrong network.
- **Pass `--opp-threshold 0 --opp-discretization 0` for Solly-B**, at both
  training and scoring time. The defaults are `0.03` and `32`, which apply to
  Solly-A only.

Learned best responses are approximate and give lower bounds on exploitability;
the exact values in §4.2 are available for 3x3 2-player only.
