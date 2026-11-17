# exact_br — exact exploitability for Liar's Poker

Vendored here so that reproducing the paper needs **one** repository. This code
computes the exact best-response values behind Table 1's "Exact Best Response"
row, the solid curves of Figures 2a and 5, and the $0.169 / $0.046 figures
quoted in Section 4.2.

It previously lived, uncommitted, inside a local clone of
`nathanlct/IIG-RL-Benchmark`. It does not depend on that benchmark: the only
requirements are `lpeas` (built from `liars_poker_eas/` here), this repo's
`src/`, and — for the OpenSpiel cross-check only — `pyspiel`.

## Contents

| path | role |
|---|---|
| `liars_poker_eas/` | the `lpeas` package: a pybind11 + OpenMP C++ public-tree traverser (`csrc/lp_tree.cpp`). The tree is far too large to materialise a policy table (3x3 2-player is ~1.06e9 information states), so policies are streamed through it chunk by chunk. |
| `eval_liars_poker_checkpoint.py` | drives the traverser against a trained checkpoint (`.pickle` or `.msgpack`), including the dummy-action mapping (a forfeit column for `dummy_mode=lose`, summed aliases for `alias`). |
| `verify_br_simulation.py` | simulates the computed best response to confirm its value empirically. |
| `tests/test_liars_poker_solver.py` | validates the traverser against OpenSpiel's own `best_response` / `exploitability` on small games. |
| `tests/test_liars_poker_dummy.py` | validates the dummy mapping against this repo's JAX game, using its legal masks, transitions, observations and payoffs. |

## Build and test

```bash
.venv/bin/pip install exact_br/liars_poker_eas        # needs CMake + a C++17 toolchain
.venv/bin/python -m pytest exact_br/tests/test_liars_poker_solver.py -v
.venv/bin/python -m pytest exact_br/tests/test_liars_poker_dummy.py -v
```

`test_liars_poker_solver.py` additionally needs OpenSpiel with
`python_liars_poker` registered. `test_liars_poker_dummy.py` reads this repo's
`src/` by default; override with `LP_SIBLING_SRC`.

## Use

```bash
.venv/bin/python exact_br/eval_liars_poker_checkpoint.py \
  experiments/3x3_v3/agent_1650000.pickle --finetune
```

Verified to return `avg_score_response = 0.16955`, i.e. the paper's $0.169 for
Solly-A. See `REPRODUCE.md` §4.1b, including when `--finetune` applies.

A pure-Python OpenSpiel traversal is not a practical substitute — measured at
~5.7k nodes/s against a 39M-node public tree.
