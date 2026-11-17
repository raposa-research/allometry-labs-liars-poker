#!/usr/bin/env python
"""Convert old haiku-based pickle checkpoints to the new flax msgpack format.

Old checkpoints were created by pickling JaxLiarsPokerRNaDSolver when the
network used Haiku. The new format uses Flax and stores state via
flax.serialization.to_bytes.

The remapping works because both Haiku and Flax produce the same ordered leaf
sequence when the network architecture is identical:

  Haiku leaves (sorted-key DFS):
    mlp.~/linear_0.{b, w},  mlp.~/linear_1.{b, w},
    mlp_1.~/linear_0.{b, w},  mlp_2.~/linear_0.{b, w}

  Flax leaves (sorted-key DFS):
    params.Dense_0.{bias, kernel},  params.Dense_1.{bias, kernel},
    params.Dense_2.{bias, kernel},  params.Dense_3.{bias, kernel}

Within each layer {b, w} and {bias, kernel} are in the same alphabetical
order, so jax.tree_util.tree_unflatten(flax_treedef, haiku_leaves) produces
the correct parameter mapping.  The same logic applies to the Adam optimizer
state, which stores mu and nu in the same param-tree shape.

Usage
-----
  python convert_checkpoint.py agent_100.pickle [agent_200.pickle ...]
  python convert_checkpoint.py checkpoints/*.pickle --output-dir converted/
"""

import argparse
import os
import pickle
import sys

import jax
import jax.core

# Allow running from any working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# JAX version-compatibility shim
# ---------------------------------------------------------------------------
# Older JAX versions included a `named_shape` keyword in ShapedArray.__init__
# that was later removed.  Pickles created with the old JAX cannot be loaded
# without this shim.

_orig_shaped_array_init = jax.core.ShapedArray.__init__

def _patched_shaped_array_init(self, *args, named_shape=None, **kwargs):
    _orig_shaped_array_init(self, *args, **kwargs)

jax.core.ShapedArray.__init__ = _patched_shaped_array_init

from rnad import RNaDConfig
from liars_poker_jax import LiarsPokerConfig
from train_jax import JaxLiarsPokerRNaDSolver


# ---------------------------------------------------------------------------
# Loading old pickles without triggering the new __init__ / __setstate__
# ---------------------------------------------------------------------------


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


def _load_old_checkpoint(path: str) -> _OldAgentData:
    with open(path, "rb") as f:
        return _AgentUnpickler(f).load()


# ---------------------------------------------------------------------------
# Param / optimizer-state remapping and policy-head trimming
# ---------------------------------------------------------------------------


def _trim_policy_head(tree, old_num_actions: int, new_num_actions: int):
    """Slice policy-head leaves from old_num_actions down to new_num_actions.

    Both the bias (shape [old_num_actions]) and kernel (shape [H, old_num_actions])
    are handled by slicing the last axis.
    """
    def trim(leaf):
        if leaf.shape and leaf.shape[-1] == old_num_actions:
            return leaf[..., :new_num_actions]
        return leaf
    return jax.tree_util.tree_map(trim, tree)


def _remap_tree(old_tree, new_template):
    """Remap old_tree leaves onto new_template's tree structure.

    Asserts that leaf counts match so mismatched architectures are caught
    early with a clear error rather than silently producing wrong values.
    """
    new_treedef = jax.tree_util.tree_structure(new_template)
    old_leaves = jax.tree_util.tree_leaves(old_tree)
    new_leaves = jax.tree_util.tree_leaves(new_template)

    if len(old_leaves) != len(new_leaves):
        raise ValueError(
            f"Leaf count mismatch — old checkpoint has {len(old_leaves)} "
            f"leaves but the new template expects {len(new_leaves)}. "
            "Check that policy_network_layers matches the saved checkpoint."
        )

    return jax.tree_util.tree_unflatten(new_treedef, old_leaves)


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def convert_checkpoint(input_path: str, output_path: str) -> None:
    print(f"Loading  {input_path}")
    old = _load_old_checkpoint(input_path)

    rnad_config: RNaDConfig = old.config

    # LiarsPokerConfig was not saved separately; reconstruct from game_name.
    # e.g. "python_liars_poker(num_digits=3,hand_length=3,num_players=2)"
    import re
    game_params = dict(re.findall(r'(\w+)=(\d+)', rnad_config.game_name))
    jax_config = LiarsPokerConfig(
        num_players=int(game_params.get("num_players", 2)),
        hand_length=int(game_params["hand_length"]),
        num_digits=int(game_params["num_digits"]),
    )

    print(
        f"  players={jax_config.num_players}  hand={jax_config.hand_length}"
        f"  digits={jax_config.num_digits}"
        f"  layers={list(rnad_config.policy_network_layers)}"
        f"  learner_steps={old.learner_steps}"
    )

    # Build a fresh flax agent to get correct template trees.
    agent = JaxLiarsPokerRNaDSolver(rnad_config, jax_config)

    # Detect the old policy-head size from the policy head kernel.
    # The policy kernel is the unique 2-D leaf where:
    #   rows == last hidden layer size, cols != 1, cols != rows.
    last_hidden = rnad_config.policy_network_layers[-1]
    policy_kernels = [
        l for l in jax.tree_util.tree_leaves(old.params)
        if l.ndim == 2 and l.shape[0] == last_hidden and l.shape[1] != 1 and l.shape[0] != l.shape[1]
    ]
    if len(policy_kernels) != 1:
        raise ValueError(
            f"Expected exactly one policy-head kernel, found {len(policy_kernels)}. "
            "Cannot auto-detect old action count."
        )
    old_num_actions = int(policy_kernels[0].shape[1])
    new_num_actions = jax_config.num_actions
    trimming = old_num_actions != new_num_actions
    if trimming:
        print(f"  trimming policy head: {old_num_actions} → {new_num_actions} actions")

    def remap_and_trim(old_tree, template):
        remapped = _remap_tree(old_tree, template)
        if trimming:
            return _trim_policy_head(remapped, old_num_actions, new_num_actions)
        return remapped

    # Remap network params.
    agent.params = remap_and_trim(old.params, agent.params)
    agent.params_target = remap_and_trim(old.params_target, agent.params_target)
    agent.params_prev = remap_and_trim(old.params_prev, agent.params_prev)
    agent.params_prev_ = remap_and_trim(old.params_prev_, agent.params_prev_)

    # Remap optimizer states.  old.optimizer / old.optimizer_target hold the
    # raw optax states (self.optimizer.state from __getstate__), which contain
    # param-shaped tensors (Adam mu/nu, SGD trace) in the haiku tree layout.
    agent.optimizer.state = remap_and_trim(old.optimizer, agent.optimizer.state)
    agent.optimizer_target.state = remap_and_trim(
        old.optimizer_target, agent.optimizer_target.state
    )

    # Scalar training state.
    agent.learner_steps = old.learner_steps
    agent.actor_steps = old.actor_steps
    agent._rngkey = old.rngkey  # __getstate__ stored it as 'rngkey'

    print(f"Saving   {output_path}")
    with open(output_path, "wb") as f:
        f.write(agent.state_bytes())
    print("  OK")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="CHECKPOINT",
        help="Pickle checkpoint file(s) to convert.",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        help="Write converted files here (default: same directory as input).",
    )
    args = parser.parse_args()

    for path in args.inputs:
        stem = os.path.splitext(path)[0]
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            stem = os.path.join(args.output_dir, os.path.basename(stem))
        convert_checkpoint(path, stem + ".msgpack")


if __name__ == "__main__":
    main()
