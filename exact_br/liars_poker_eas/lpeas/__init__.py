"""Exact exploitability for 2-player Liar's Poker (OpenSpiel rules).

The game tree is far too large to materialise a policy table (3 digits x 3
cards: ~1.06B information states), so policies are streamed through the public
tree chunk by chunk. `ev_and_exploitability` drives that loop for a policy
function; see `LiarsPokerTraverser` for the low-level API.
"""

import time

import numpy as np

from .pylpeas import LiarsPokerTraverser  # noqa: F401


def ev_and_exploitability(traverser, policy_fn, chunk_nodes=20_000, verbose=True, store_br=False):
    """Computes ev0 / exploitabilities for the profile given by `policy_fn`.

    Args:
      traverser: a `LiarsPokerTraverser`.
      policy_fn: `(player, infostates uint8 [n, IS], legal bool [n, NA]) ->
        probs float [n, NA]`; rows must sum to one over legal actions
        (entries on illegal actions are ignored). Alternatively `[n, NA + 1]`
        whose last column is the probability of forfeiting: the mass on
        "dummy" actions that end the game at once with the acting player
        losing (the sibling repo's `dummy_mode="lose"` action-space probe;
        all such actions are equivalent, so only their total mass matters).
        Rows then sum to one including that column.
      chunk_nodes: number of public nodes per chunk (times `num_hands` rows).
      store_br: keep the best-response policy in the traverser (one byte per
        infoset; see `LiarsPokerTraverser.br_actions`).

    Returns:
      dict with `ev0`, `expl=(expl0, expl1)`, `br_value=(br0, br1)`
      (same conventions as the `eas` PTTT/Dark Hex traversers: `expl[p]` is the
      exploitability of player p's strategy; `(expl0 + expl1) / 2` is what
      `main.py` logs as `avg_score_response`).
    """
    t0 = time.time()
    total = traverser.num_infosets(0) + traverser.num_infosets(1)
    done = 0
    next_report = 0.0
    traverser.begin(chunk_nodes)
    while (chunk := traverser.next_chunk()) is not None:
        player, infostates, legal = chunk
        probs = np.ascontiguousarray(policy_fn(player, infostates, legal), dtype=np.float64)
        traverser.push_probs(probs)
        done += infostates.shape[0]
        if verbose and done / total >= next_report:
            print(f"[lpeas] {done / total * 100:5.1f}% of infosets evaluated ({time.time() - t0:.0f}s)")
            next_report += 0.1
    out = traverser.finish(store_br)
    if verbose:
        print(f"[lpeas] done in {time.time() - t0:.0f}s: ev0={out['ev0']:.6f} expl={out['expl']}")
    return out


def uniform_policy_fn(player, infostates, legal):
    """Uniform random policy over legal actions (for tests)."""
    del player, infostates
    legal = legal.astype(np.float64)
    return legal / legal.sum(axis=1, keepdims=True)
