# Liar's Poker

Code for *Liar's Poker*: training Liar's Poker agents with R-NaD and with the
softmax policy gradient plus an entropy bonus, measuring their exploitability
with learned and exact best responses.

**To reproduce the paper's experiments, see [REPRODUCE.md](REPRODUCE.md)** —
one section per figure and table, with the commands and the expected values.
The rest of this file is reference documentation for the individual tools.

## Layout

| path | contents |
|---|---|
| `src/` | the game, R-NaD / softmax-PG training, best-response training and evaluation |
| `paper/` | analysis scripts for Figures 2b and 3 and Table 3 |
| `exact_br/` | exact exploitability: a C++ public-tree traverser and its driver (see `exact_br/README.md`) |
| `experiments/` | the released Solly-A 3x3 2-player checkpoint |
| `skypilot/` | launch configurations for the 8x10 training runs |

## Installing Dependencies

``` shell
pip install -r requirements.txt
pip install 'jax[cuda12]'            # for GPU training
pip install exact_br/liars_poker_eas # exact best response; needs CMake + C++17
```

## train_jax.py — RNaD + PPO best-response training

Trains an RNaD agent on Liar's Poker using a fully JAX/GPU-based trajectory
rollout, interleaved with PPO best-response (BR) training to measure
exploitability.

**Loop structure:** each phase runs `rnad_steps_per_phase` RNaD gradient steps,
then `br_collect_per_phase` PPO collect+update cycles (one per player position)
against a frozen copy of the current RNaD policy.  Every `eval_every_phases`
phases the BR agents are evaluated against the main agent.

Configuration is managed by [Hydra](https://hydra.cc/). The default config is in
`src/conf/train_jax.yaml`.

``` shell
cd src/

# train with defaults (3-digit, 3-card, 2-player, MLP network)
python train_jax.py

# override game or training params on the command line
python train_jax.py num_digits=5 hand_length=5 num_phases=2000

# factored policy head, as used for the 8x10 agents
python train_jax.py network_type=factored_mlp

# resume from a checkpoint
python train_jax.py load_checkpoint=checkpoints/agent_1000.msgpack
```

Key config options (all overridable via CLI):

| Key | Default | Description |
|-----|---------|-------------|
| `num_digits` | 3 | Number of distinct digit values |
| `hand_length` | 3 | Cards per player |
| `num_players` | 2 | Number of players |
| `num_phases` | 1000 | Total training phases |
| `rnad_steps_per_phase` | 100 | RNaD gradient steps per phase |
| `br_collect_per_phase` | 10 | PPO collect+update cycles per phase |
| `eval_every_phases` | 5 | Evaluate BR vs main every N phases |
| `checkpoint_every_phases` | 20 | Save checkpoint every N phases |
| `network_type` | `mlp` | `mlp`, `factored_mlp` (the 8x10 agents) or `transformer` |
| `save_dir` | `checkpoints` | Output directory for checkpoints |
| `load_checkpoint` | null | Path to resume from |

Checkpoints are saved as `<save_dir>/agent_<step>.msgpack` and are
self-contained (embed game config and network architecture).

Metrics logged to wandb:
- `rnad/loss` — RNaD training loss
- `br/reward_p{i}` — mean reward of BR agent as player *i* vs frozen main
- `br/agent_score` — mean reward of main agent across all BR configurations (lower = more exploitable)

## train_br.py — standalone best-response training

Trains a PPO best-response agent against a single frozen checkpoint, without
running RNaD.  Useful for evaluating exploitability of an existing checkpoint
after training.

Game config and network architecture are read directly from the checkpoint, so
no separate flags are needed for those.

``` shell
cd src/

python train_br.py path/to/agent.msgpack

# train BR for a specific player position only
python train_br.py agent.msgpack --players 0

# more cycles, larger batch
python train_br.py agent.msgpack --n-cycles 500 --batch-size 1024

# save intermediate checkpoints
python train_br.py agent.msgpack --checkpoint-every 50 --save-dir checkpoints/br
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `checkpoint` | (required) | Path to the frozen main-agent `.msgpack` checkpoint |
| `--players` | all | Player positions to train BR for |
| `--n-cycles` | 200 | Number of collect+update cycles |
| `--eval-every` | 20 | Evaluate every N cycles |
| `--eval-episodes` | 2048 | Episodes per evaluation |
| `--batch-size` | 512 | Parallel environments per rollout |
| `--lr` | 3e-4 | PPO learning rate |
| `--save-dir` | `checkpoints/br` | Output directory for BR checkpoints |
| `--checkpoint-every` | 0 | Save every N cycles (0 = end only) |
| `--no-wandb` | — | Disable wandb logging |

BR checkpoints are saved as `<save_dir>/br_p<player>_<cycle>.msgpack`.

The defaults here are not the paper's settings; see
[REPRODUCE.md](REPRODUCE.md) §4.6 for those, and §5 for how to read the
per-seat output.

The final output reports `br/agent_score` (mean reward of the main agent vs
trained BR).  For a zero-sum game, 0.0 = Nash; negative values indicate the
main agent is exploitable.

## eval_agents.py — head-to-head evaluation

Pits two saved agents against each other and reports win rates and mean rewards.
Supports both new `.msgpack` checkpoints (from `train_jax.py` or `train_br.py`)
and old `.pickle` checkpoints.  Format is auto-detected by file extension.

``` shell
cd src/

# basic evaluation (500 games per position = 1000 total)
python eval_agents.py --agent0 checkpoints/agent_a.msgpack --agent1 checkpoints/agent_b.msgpack

# more games, fixed seed, save game logs
python eval_agents.py \
    --agent0 agent_a.msgpack \
    --agent1 agent_b.msgpack \
    --num-games 2000 \
    --seed 0 \
    --save-games games.json \
    --save-num-games 50
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--agent0` | (required) | Checkpoint for player 0 |
| `--agent1` | (required) | Checkpoint for player 1 |
| `--num-games` | 500 | Games per position (total = 2×) |
| `--max-steps` | 50 | Max game steps per rollout |
| `--seed` | 42 | Random seed |
| `--verbose` | false | Print per-game results |
| `--save-games` | — | Path for JSON game log |
| `--save-num-games` | 20 | Games to record per position in JSON |

Each agent plays both positions (to cancel first-mover advantage).  Output
includes per-position and overall mean reward, win rates, and a color-coded
action heatmap showing bid frequency by count × digit.

The JSON game log records hands, move sequence, per-move policy distributions,
and final rewards for each game.


## License

Copyright Allometry Labs. Released under the
[PolyForm Noncommercial License 1.0.0](LICENSE): free for research, teaching,
personal and other noncommercial use; commercial use requires a separate
license from Allometry Labs.

**Exception — two files remain under the Apache License 2.0**
([LICENSE-APACHE](LICENSE-APACHE)), because they are derived from
[OpenSpiel](https://github.com/google-deepmind/open_spiel):

- `src/rnad.py`, derived from OpenSpiel's R-NaD implementation (v1.5,
  `open_spiel/python/algorithms/rnad/rnad.py`);
- `src/liars_poker_jax.py`, a JAX reimplementation of OpenSpiel's
  `python_liars_poker` game.

Both keep DeepMind's original Apache-2.0 header plus a note describing the
changes.
