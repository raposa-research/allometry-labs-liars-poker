# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by Allometry Labs, 2026. A JAX reimplementation of
# the game logic in OpenSpiel's open_spiel/python/games/liars_poker.py, rewritten
# as pure functions so it can be jit-compiled and vmapped. Main additions:
# N-player configurations, a per-digit histogram hand encoding, dummy actions
# (dummy_mult / dummy_mode), and compact and transformer observation encodings.
# This file, including these modifications, is licensed under the Apache
# License 2.0 (see LICENSE-APACHE), not the repository's PolyForm Noncommercial
# license.

"""GPU-friendly JAX implementation of Liar's Poker.

This module implements the same game logic as liars_poker.py but using
pure JAX operations so it can be JIT-compiled and vmapped over batches
of environments.

Design principles:
  - All state is stored in a `LiarsPokerState` chex dataclass (a JAX pytree).
  - All functions are pure: they take a state and return a new state.
  - No Python control flow (if/else) in hot paths — boolean masks are used.
  - Static game configuration lives in `LiarsPokerConfig` (a plain dataclass,
    treated as a compile-time constant by JAX).

Typical usage::

    config = LiarsPokerConfig(num_players=2, hand_length=5, num_digits=5)
    state = new_initial_state(config)

    # JIT-compile apply_action for speed:
    apply = jax.jit(lambda s, a: apply_action(s, config, a))

    # vmap over a batch of environments:
    batched_apply = jax.vmap(lambda s, a: apply_action(s, config, a))
"""

import dataclasses

import chex
import jax.numpy as jnp

from transformer_seq import LPInputs

CHALLENGE_ACTION = 0
BID_ACTION_OFFSET = 1

# Sentinel values for current_player_id(), matching pyspiel conventions.
CHANCE_PLAYER_ID = -1
TERMINAL_PLAYER_ID = -4


# ---------------------------------------------------------------------------
# Static configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LiarsPokerConfig:
  """Static game configuration (Python scalars, not traced by JAX).

  Because JAX traces through dynamic values, all game parameters that define
  array shapes must be plain Python ints here, not JAX arrays.
  """

  num_players: int = 2
  hand_length: int = 10
  num_digits: int = 10
  # Optional action-space abstraction. When set, the next bid's COUNT is
  # restricted to be at most `max_jump` above the current bid's count
  # (base_count for the opening bid). None (default) = unrestricted, and the
  # legal-action mask is byte-identical to the original game. Restricts COUNT
  # only, never the digit; challenge and same-count higher-number bids remain
  # legal.
  max_jump: "int | None" = None
  # Private-hand encoding in observation_tensor.
  #   "histogram" (default) -- per-digit counts, length num_digits. Collapses
  #       the order-irrelevant hand, shrinking the effective hand space.
  #   "raw"                 -- the hand's digit values in dealt order, length
  #       hand_length. This is what OpenSpiel's LiarsPokerObserver emits
  #       (`np.asarray(state.hands[player])`), so checkpoints trained against
  #       `python_liars_poker` REQUIRE it.
  # NOTE the two have the same LENGTH whenever hand_length == num_digits, so a
  # mismatch does not change the observation size and fails silently -- the
  # agent simply stops being able to read its own hand.
  hand_encoding: str = "histogram"
  # ACTION-SPACE SCALING PROBE (dummy_mult > 1). Appends
  # (dummy_mult - 1) * num_real_actions "dummy" actions to the action space.
  # A dummy is legal exactly when some real action is legal, and applying one
  # resolves it to a UNIFORMLY RANDOM legal REAL action (see _resolve_dummy).
  # A dummy is therefore a mixture over existing pure strategies, so it can
  # never beat the best real action: the game's value and equilibrium set are
  # unchanged and only |A| grows. That isolates action-space size, which is
  # what NeuRD's inv_mu = 1/mu(a_t) scales with. dummy_mult=1 (default) is
  # byte-identical to the original game.
  dummy_mult: int = 1
  # HOW a dummy resolves. This choice decides what the probe actually measures.
  #   "random"  -- uniformly random legal REAL action (the original). Inflates
  #                |A| but ALSO makes the transition stochastic, which is a
  #                variance source independent of |A|: measured, PPO's relative
  #                gradient variance is 21x higher in the dummy game (1.300 at
  #                |A_legal|=71) than in a REAL game of comparable size (0.061
  #                at |A_legal|=44). So it does not isolate action-space size.
  #   "alias"   -- DETERMINISTIC duplicate of one specific real action: legal
  #                exactly when its target is legal, and applying it applies the
  #                target. |A_legal| scales by exactly dummy_mult, dummy values
  #                are as distinct as the reals they alias, and the transition
  #                stays deterministic. Mass on a dummy is behaviourally
  #                harmless (you play the aliased action), so performance can
  #                only degrade through LEARNING difficulty -- action-space size
  #                alone, which is the point of the probe.
  #   "lose"    -- taking a dummy INSTANTLY LOSES the game for the taker.
  #                Deterministic, and NOT a duplicate of any real action, so
  #                each real action keeps its own logit and PPO's single-action
  #                signal is NOT split (the flaw that made "alias" measure the
  #                wrong thing). Strictly dominated, so no equilibrium puts mass
  #                on a dummy and the game's value is preserved. Makes dummy
  #                mass COSTLY, which turns NeuRD's beta trust region into a
  #                prediction: apply_force_with_threshold cannot push a centered
  #                logit below -beta, so NeuRD must retain roughly
  #                D/(D + R*exp(4)) of its mass on instantly-losing actions
  #                (D dummies, R real legal actions) -- a floor that GROWS with
  #                the action count.
  dummy_mode: str = "random"

  @property
  def total_cards(self) -> int:
    """Total number of cards dealt across all players."""
    return self.num_players * self.hand_length

  @property
  def max_bids(self) -> int:
    """Total number of distinct bid IDs (0-indexed bid action space)."""
    return self.hand_length * self.num_digits * self.num_players

  @property
  def num_real_actions(self) -> int:
    """Real action space: 0 = challenge, 1..max_bids = bids."""
    return self.max_bids + 1

  @property
  def num_actions(self) -> int:
    """Full action space, including dummy actions when dummy_mult > 1."""
    return self.num_real_actions * self.dummy_mult


# ---------------------------------------------------------------------------
# Game state (JAX pytree via chex.dataclass)
# ---------------------------------------------------------------------------


@chex.dataclass(frozen=True)
class LiarsPokerState:
  """Full game state stored as JAX arrays.

  This is a `chex.dataclass` with `frozen=True`, which means:
    - All annotated fields are JAX array leaves in the pytree.
    - Instances are immutable; use `.replace(field=new_value)` to create
      updated copies.
    - `jax.vmap`, `jax.jit`, and `jax.lax.scan` all work out of the box.

  Fields:
    hands:              [num_players, hand_length] int32 – digit value at each
                        position. Undealt positions hold 0 (never counted).
    deal_step:          scalar int32 – number of cards dealt so far.
    bid_history:        [max_bids, num_players] int32 – 1 where player bid.
    challenge_history:  [max_bids, num_players] int32 – 1 where player
                        challenged.
    current_player:     scalar int32 in [0, num_players).
    bid_originator:     scalar int32; -1 if no bid has been made.
    current_bid_action: scalar int32; -1 if no bid has been made.
    num_challenges:     scalar int32 – challenges since last bid.
    is_rebid:           scalar bool – True when the most recent bid was a
                        rebid by the bid originator.
    winner:             scalar int32; -1 if game is not over.
    loser:              scalar int32; -1 if game is not over.
  """

  hands: chex.Array            # [num_players, hand_length], int32
  deal_step: chex.Array        # scalar int32

  bid_history: chex.Array       # [max_bids, num_players], int32
  challenge_history: chex.Array  # [max_bids, num_players], int32

  current_player: chex.Array    # scalar int32
  bid_originator: chex.Array    # scalar int32, -1 = no bid
  current_bid_action: chex.Array  # scalar int32, -1 = no bid
  num_challenges: chex.Array    # scalar int32
  is_rebid: chex.Array          # scalar bool

  winner: chex.Array  # scalar int32, -1 = not terminal
  loser: chex.Array   # scalar int32, -1 = not terminal


# ---------------------------------------------------------------------------
# State construction
# ---------------------------------------------------------------------------


def new_initial_state(config: LiarsPokerConfig) -> LiarsPokerState:
  """Returns the initial game state (before any cards are dealt)."""
  return LiarsPokerState(
      hands=jnp.zeros(
          (config.num_players, config.hand_length), dtype=jnp.int32
      ),
      deal_step=jnp.zeros((), dtype=jnp.int32),
      bid_history=jnp.zeros(
          (config.max_bids, config.num_players), dtype=jnp.int32
      ),
      challenge_history=jnp.zeros(
          (config.max_bids, config.num_players), dtype=jnp.int32
      ),
      current_player=jnp.zeros((), dtype=jnp.int32),
      bid_originator=jnp.full((), -1, dtype=jnp.int32),
      current_bid_action=jnp.full((), -1, dtype=jnp.int32),
      num_challenges=jnp.zeros((), dtype=jnp.int32),
      is_rebid=jnp.zeros((), dtype=jnp.bool_),
      winner=jnp.full((), -1, dtype=jnp.int32),
      loser=jnp.full((), -1, dtype=jnp.int32),
  )


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------


def is_terminal(state: LiarsPokerState) -> chex.Array:
  """Returns True (scalar bool) if the game is over."""
  return (state.winner >= 0) | (state.loser >= 0)


def is_chance_node(
    state: LiarsPokerState, config: LiarsPokerConfig
) -> chex.Array:
  """Returns True (scalar bool) if we are still in the deal phase."""
  return state.deal_step < config.total_cards


def current_player_id(
    state: LiarsPokerState, config: LiarsPokerConfig
) -> chex.Array:
  """Returns the ID of the player to act.

  Returns:
    TERMINAL_PLAYER_ID (-4) if the game is over.
    CHANCE_PLAYER_ID (-1)   during the deal phase.
    0 .. num_players-1      otherwise.
  """
  return jnp.where(
      is_terminal(state),
      jnp.int32(TERMINAL_PLAYER_ID),
      jnp.where(
          is_chance_node(state, config),
          jnp.int32(CHANCE_PLAYER_ID),
          state.current_player,
      ),
  )


def chance_probabilities(config: LiarsPokerConfig) -> chex.Array:
  """Returns the uniform probability over the num_digits chance outcomes.

  The digit outcomes match the original deck order [1, 2, ..., num_digits-1, 0],
  where the action/outcome value equals the digit face value.
  """
  return jnp.ones(config.num_digits, dtype=jnp.float32) / config.num_digits


def legal_actions_mask(
    state: LiarsPokerState, config: LiarsPokerConfig
) -> chex.Array:
  """Returns a boolean mask of shape [num_actions] (True = legal action).

  Action 0 is the CHALLENGE action; actions 1..max_bids are bid actions.
  Masks out all actions when in the chance/terminal phases.
  """
  action_indices = jnp.arange(config.num_actions, dtype=jnp.int32)

  # Challenge is legal once at least one bid has been made.
  challenge_legal = state.current_bid_action >= 0

  # A rebid is possible when all non-originator players have challenged
  # (and we have not yet executed a rebid this round).
  is_rebid_possible = (~state.is_rebid) & (
      state.num_challenges == config.num_players - 1
  )

  # A player may bid if they are not the current bid originator
  # OR if a rebid is possible (the originator may rebid).
  can_bid = (state.current_player != state.bid_originator) | is_rebid_possible

  # Bids must be strictly higher than the current bid.
  min_bid_action = jnp.where(
      state.current_bid_action < 0,
      jnp.int32(BID_ACTION_OFFSET),
      state.current_bid_action + 1,
  )

  challenge_mask = (action_indices == 0) & challenge_legal
  bid_mask = (
      (action_indices >= min_bid_action)
      & (action_indices <= config.max_bids)
      & can_bid
  )

  # Optional action-space abstraction (max_jump): restrict the next bid's
  # COUNT to be within `max_jump` of the current bid's count. base_count is
  # the current bid's count, or 1 for the opening bid (no bid placed yet).
  # This shrinks the legal action space and is applied identically to EVERY
  # consumer of this mask (the RNaD agent's rollouts and the frozen-agent
  # opponent inside the decoupled BR). When max_jump is None (default) this
  # branch is skipped and the mask is byte-identical to the unrestricted game.
  # Challenge and same-count higher-number bids remain legal (count ==
  # base_count <= base_count + max_jump). Restricts COUNT only, not the digit.
  if config.max_jump is not None:
    D = config.num_digits
    has_bid = state.current_bid_action >= 0
    current_bid_count = (
        state.current_bid_action - BID_ACTION_OFFSET
    ) // D + 1
    base_count = jnp.where(has_bid, current_bid_count, jnp.int32(1))
    # Count of the bid at each action index a (>= 1): ((a - 1) // D) + 1.
    action_bid_count = (action_indices - BID_ACTION_OFFSET) // D + 1
    count_within_jump = action_bid_count <= (base_count + config.max_jump)
    bid_mask = bid_mask & count_within_jump

  real_mask = challenge_mask | bid_mask
  if config.dummy_mult == 1:
    return real_mask            # byte-identical to the original game

  # A dummy resolves to a uniformly random LEGAL REAL action, so it is legal
  # exactly when at least one real action is legal. That inherits every guard
  # above (chance node, terminal, can_bid, max_jump) for free.
  is_dummy = action_indices >= config.num_real_actions
  if config.dummy_mode in ("alias", "lose"):
    # Each dummy is legal EXACTLY when the real action it duplicates is legal.
    # So |A_legal| scales by exactly dummy_mult, unlike "random" where dummies
    # are always legal and inflate the legal set disproportionately (7.0 -> 30.9
    # at nominal 2x).
    tgt = jnp.mod(action_indices - config.num_real_actions,
                  config.num_real_actions)
    return jnp.where(is_dummy, real_mask[tgt], real_mask)
  any_real = jnp.any(real_mask)
  return real_mask | (is_dummy & any_real)


def dummy_alias_target(action: chex.Array, config: LiarsPokerConfig) -> chex.Array:
  """For dummy_mode="alias": the REAL action that dummy `action` duplicates.

  A function of the action index ONLY -- never of the state -- so the alias map
  is fixed for the whole game. Dummy n_real + k duplicates real action
  k % n_real, i.e. each real action gets exactly (dummy_mult - 1) duplicates
  and the multiplier applies uniformly across the action space.
  """
  n_real = config.num_real_actions
  k = jnp.int32(action) - n_real
  return jnp.mod(k, jnp.int32(n_real))


def real_actions_mask(
    state: LiarsPokerState, config: LiarsPokerConfig
) -> chex.Array:
  """The legal mask restricted to REAL actions, shape [num_real_actions]."""
  return legal_actions_mask(state, config)[: config.num_real_actions]


def _resolve_dummy(
    state: LiarsPokerState, config: LiarsPokerConfig, action: chex.Array
) -> chex.Array:
  """Map a dummy action to a uniformly random LEGAL REAL action.

  `apply_action` is pure and carries no PRNG key, and threading one through
  would touch every call site in the codebase (rollouts, crossplay, pretrial,
  BR). Instead the draw is a deterministic hash of the full state plus the
  action index -- including the private hands, so it is unpredictable from any
  single player's information set and cannot be inverted into a useful signal.
  Deterministic given (state, action), pseudorandom across them.

  Uses a splitmix32-style avalanche so neighbouring states/actions do not map
  to neighbouring outcomes (a plain sum would make dummy k resolve almost
  monotonically in k, which is a pattern a network could exploit).
  """
  def _mix(h):
    h = jnp.uint32(h)
    h ^= h >> jnp.uint32(16)
    h *= jnp.uint32(0x7FEB352D)
    h ^= h >> jnp.uint32(15)
    h *= jnp.uint32(0x846CA68B)
    h ^= h >> jnp.uint32(16)
    return h

  h = jnp.uint32(0x9E3779B9)
  for field in (state.deal_step, state.current_player, state.bid_originator,
                state.current_bid_action, state.num_challenges,
                state.is_rebid.astype(jnp.int32), jnp.int32(action)):
    h = _mix(h ^ jnp.uint32(jnp.int32(field) & jnp.int32(0x7FFFFFFF)))
  h = _mix(h ^ jnp.uint32(jnp.sum(state.hands.astype(jnp.int32)
                                  * (jnp.arange(state.hands.size) + 1)
                                  .reshape(state.hands.shape))
                          & jnp.int32(0x7FFFFFFF)))

  real_mask = legal_actions_mask(state, config)[: config.num_real_actions]
  n_legal = jnp.sum(real_mask.astype(jnp.int32))
  k = jnp.int32(h % jnp.uint32(jnp.maximum(n_legal, 1)))
  # index of the k-th legal real action
  rank = jnp.cumsum(real_mask.astype(jnp.int32)) - 1
  hit = real_mask & (rank == k)
  # argmax of a bool picks the first True; fall back to 0 if nothing is legal
  return jnp.where(n_legal > 0, jnp.argmax(hit).astype(jnp.int32),
                   jnp.int32(CHALLENGE_ACTION))


# ---------------------------------------------------------------------------
# Bid encoding helpers (match the original _decode_bid / encode_bid exactly)
# ---------------------------------------------------------------------------


def decode_bid(
    bid_id: chex.Array, config: LiarsPokerConfig
) -> tuple[chex.Array, chex.Array]:
  """Decodes a 0-indexed bid_id into (count, number).

  Matches LiarsPokerState._decode_bid() exactly.
  """
  number = bid_id % config.num_digits + 1
  count = bid_id // config.num_digits + 1
  return count, number


def encode_bid(
    count: chex.Array, number: chex.Array, config: LiarsPokerConfig
) -> chex.Array:
  """Encodes (count, number) into a 0-indexed bid_id.

  The bid action value is encode_bid(...) + BID_ACTION_OFFSET.
  Matches LiarsPokerState.encode_bid() exactly.
  """
  return (count - 1) * config.num_digits + (number - 1)


# ---------------------------------------------------------------------------
# State transition
# ---------------------------------------------------------------------------


def apply_action(
    state: LiarsPokerState,
    config: LiarsPokerConfig,
    action: chex.Array,
) -> LiarsPokerState:
  """Applies an action and returns the resulting state.

  This function is pure (no side effects), uses only JAX operations, and is
  compatible with `jax.jit` and `jax.vmap`.

  Boolean masks select the correct update path for each of the three action
  types (chance deal, challenge, bid) without any Python branching.

  Args:
    state:   The current game state.
    config:  Static game configuration.
    action:  The action to apply as a scalar int32 JAX array.

  Returns:
    A new LiarsPokerState reflecting the action.
  """
  action = jnp.int32(action)

  # DUMMY ACTIONS (dummy_mult > 1): resolve to a uniformly random legal REAL
  # action BEFORE anything else, so every downstream branch -- including the
  # bid/challenge histories and the observation -- sees only real actions and
  # is byte-identical to the unscaled game. dummy_mult == 1 skips this
  # entirely (Python-level branch on a static config field, so it is compiled
  # out and the original path is untouched).
  is_lose_dummy = jnp.bool_(False)
  if config.dummy_mult > 1 and config.dummy_mode == "lose":
    # Remember who took it, then resolve to the alias target ONLY so the normal
    # machinery below has a valid action to process; the resulting state is
    # overridden at the end to a terminal loss for the taker. Guarded to real
    # decision nodes so a dummy index during the deal/terminal is inert.
    is_lose_dummy = ((jnp.int32(action) >= config.num_real_actions)
                     & (~is_terminal(state))
                     & (~is_chance_node(state, config)))
    loser_player = state.current_player
    action = jnp.where(jnp.int32(action) >= config.num_real_actions,
                       dummy_alias_target(action, config), action)
  if config.dummy_mult > 1 and config.dummy_mode != "lose":
    if config.dummy_mode == "alias":
      # DETERMINISTIC: no PRNG, no added environment stochasticity. This is the
      # difference that matters -- "random" mode randomises the transition,
      # which is a variance source independent of |A| (measured: PPO relative
      # variance 1.300 in the dummy game at |A_legal|=71 vs 0.061 in a REAL
      # game at |A_legal|=44, a 21x gap that |A| cannot explain).
      resolved = dummy_alias_target(action, config)
    else:
      resolved = _resolve_dummy(state, config, action)
    action = jnp.where(action >= config.num_real_actions, resolved, action)

  # Determine which of the three action types applies.  All three flags are
  # mutually exclusive; at most one is True.
  terminal = is_terminal(state)
  chance = is_chance_node(state, config) & ~terminal
  not_chance = ~chance & ~terminal
  is_challenge = not_chance & (action == CHALLENGE_ACTION)
  is_bid = not_chance & (action != CHALLENGE_ACTION)

  # ------------------------------------------------------------------
  # CHANCE: deal a digit into the current player's hand.
  # ------------------------------------------------------------------
  # deal_step % num_players gives the receiving player; // gives the slot.
  deal_player = state.deal_step % config.num_players
  deal_slot = state.deal_step // config.num_players

  # Compute a candidate updated hands array (safe even when out of range,
  # because JAX clamps OOB scatter indices; jnp.where selects the original
  # when chance=False).
  # Store action + 1 so dealt digits are 1..num_digits, matching the 1-indexed
  # bid_number returned by decode_bid.  Undealt slots stay 0 (never counted).
  updated_hands = state.hands.at[deal_player, deal_slot].set(action + 1)
  new_hands = jnp.where(chance, updated_hands, state.hands)
  new_deal_step = state.deal_step + chance.astype(jnp.int32)

  # ------------------------------------------------------------------
  # CHALLENGE: record and check whether counting should occur.
  # ------------------------------------------------------------------
  # Use bid_id=0 as a safe fallback when no bid has been placed yet
  # (is_challenge will be False in that case, so the write is harmless).
  safe_chall_bid_idx = jnp.where(
      is_challenge,
      state.current_bid_action - BID_ACTION_OFFSET,
      jnp.int32(0),
  )
  new_challenge_history = state.challenge_history.at[
      safe_chall_bid_idx, state.current_player
  ].add(is_challenge.astype(jnp.int32))

  new_num_challenges = state.num_challenges + is_challenge.astype(jnp.int32)

  # Counting occurs when all required players have challenged:
  #   - Non-rebid: all num_players players must challenge.
  #   - Rebid:     all but the originator (num_players - 1) must challenge.
  should_count_non_rebid = (~state.is_rebid) & (
      new_num_challenges == config.num_players
  )
  should_count_rebid = state.is_rebid & (
      new_num_challenges == config.num_players - 1
  )
  should_count = is_challenge & (should_count_non_rebid | should_count_rebid)

  # Decode the current bid to check the count.
  bid_id = jnp.where(
      state.current_bid_action >= 0,
      state.current_bid_action - BID_ACTION_OFFSET,
      jnp.int32(0),
  )
  bid_count, bid_number = decode_bid(bid_id, config)

  # Count matching digits across all hands.  Undealt slots hold 0 and are
  # never counted since bid_number >= 1; dealt digits are stored as 1..num_digits.
  matches = jnp.sum(state.hands == bid_number)
  bidder_wins = matches >= bid_count

  new_winner = jnp.where(
      should_count & bidder_wins, state.bid_originator, state.winner
  )
  new_loser = jnp.where(
      should_count & ~bidder_wins, state.bid_originator, state.loser
  )

  # ------------------------------------------------------------------
  # BID: record the bid and update bid originator / rebid state.
  # ------------------------------------------------------------------
  # Use bid_id=0 as a safe fallback when action=0 (challenge); the write
  # value will be 0 (is_bid=False), making it a no-op.
  safe_bid_idx = jnp.where(is_bid, action - BID_ACTION_OFFSET, jnp.int32(0))
  new_bid_history = state.bid_history.at[
      safe_bid_idx, state.current_player
  ].add(is_bid.astype(jnp.int32))

  new_current_bid_action = jnp.where(
      is_bid, action, state.current_bid_action
  )
  new_bid_originator = jnp.where(
      is_bid, state.current_player, state.bid_originator
  )

  # is_rebid is True when the bid originator bids again (checked *before*
  # updating bid_originator so we compare against the previous originator).
  is_rebid_update = state.current_player == state.bid_originator
  new_is_rebid = jnp.where(is_bid, is_rebid_update, state.is_rebid)

  # Reset challenge counter whenever a new bid is placed.
  new_num_challenges_final = jnp.where(
      is_bid, jnp.int32(0), new_num_challenges
  )

  # ------------------------------------------------------------------
  # Advance current_player (always, for all action types).
  # ------------------------------------------------------------------
  new_current_player = (state.current_player + 1) % config.num_players

  # dummy_mode="lose": terminate immediately with the TAKER as the loser.
  # `returns` pays relative to bid_originator (originator gets
  # -(num_players-1) when loser >= 0, everyone else +1), so setting
  # bid_originator := taker and loser := taker makes the taker eat the whole
  # loss and keeps the game zero-sum at any player count.
  if config.dummy_mult > 1 and config.dummy_mode == "lose":
    new_bid_originator = jnp.where(is_lose_dummy, loser_player,
                                   new_bid_originator)
    new_loser = jnp.where(is_lose_dummy, loser_player, new_loser)
    new_winner = jnp.where(is_lose_dummy, jnp.int32(-1), new_winner)

  return state.replace(
      hands=new_hands,
      deal_step=new_deal_step,
      bid_history=new_bid_history,
      challenge_history=new_challenge_history,
      current_player=new_current_player,
      bid_originator=new_bid_originator,
      current_bid_action=new_current_bid_action,
      num_challenges=new_num_challenges_final,
      is_rebid=new_is_rebid,
      winner=new_winner,
      loser=new_loser,
  )


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


def observation_size(
    config: LiarsPokerConfig,
    include_private_hand: bool = True,
    include_public: bool = True,
    include_history: bool = True,
    history_encoding: str = "sparse",
) -> int:
  """Returns the length of the flat observation tensor for the given flags.

  Mirrors the piece list in LiarsPokerObserver.__init__:
    - player one-hot:          num_players          (always)
    - private hand digits:     hand_length          (if include_private_hand)
    - rebid_state:             1                    (if include_public)
    - counts_state:            1                    (if include_public)
    - bid_history (flat):      max_bids*num_players (if include_public & history)
    - challenge_history (flat):max_bids*num_players (if include_public & history)

  When ``history_encoding == "compact"`` the two sparse history matrices are
  replaced by a dense structured summary of length ``3*num_digits +
  num_players + 5`` (see ``_compact_history``).
  """
  size = config.num_players
  if include_private_hand:
    size += config.num_digits  # histogram encoding: one count per digit value
  if include_public:
    size += 2  # rebid_state + counts_state
    if include_history:
      if history_encoding == "compact":
        size += 3 * config.num_digits + config.num_players + 5
      else:
        size += 2 * config.max_bids * config.num_players
  return size


def _compact_history(
    state: LiarsPokerState,
    config: LiarsPokerConfig,
    player: chex.Array,
) -> chex.Array:
  """Dense structured summary of the bid/challenge history (PoV of `player`).

  Length = ``3*num_digits + num_players + 5``.  Pieces (D=num_digits,
  P=num_players, TC=total_cards):
    1. has_bid                              [1]
    2. current-bid count, normalized (/TC)  [1]
    3. current-bid number one-hot           [D]
    4. bid_originator relative-to-me one-hot [P]
    5. num_challenges normalized (/P)       [1]
    6. is_rebid                             [1]
    7. bidding depth (current_bid_action+1)/(max_bids+1) [1]
    8. per-digit highest count ever bid (/TC) [D]
    9. per-digit challenged flag            [D]
  """
  D = config.num_digits
  P = config.num_players
  TC = config.total_cards
  max_bids = config.max_bids

  has_bid = state.current_bid_action >= 0
  has_bid_f = has_bid.astype(jnp.float32)

  # Decode the current bid (safe fallback when no bid yet).
  cur_bid_id = jnp.where(
      has_bid, state.current_bid_action - BID_ACTION_OFFSET, jnp.int32(0)
  )
  cur_count, cur_number = decode_bid(cur_bid_id, config)

  # 1. has_bid
  f_has_bid = jnp.reshape(has_bid_f, (1,))
  # 2. current-bid count normalized
  f_count = jnp.reshape(
      jnp.where(has_bid, cur_count.astype(jnp.float32) / TC, 0.0), (1,)
  )
  # 3. current-bid number one-hot [D]
  number_oh = (
      jnp.arange(1, D + 1, dtype=jnp.int32) == cur_number
  ).astype(jnp.float32) * has_bid_f
  # 4. bid_originator relative-to-me one-hot [P]
  rel = (state.bid_originator - player) % P
  orig_oh = (
      jnp.arange(P, dtype=jnp.int32) == rel
  ).astype(jnp.float32) * has_bid_f
  # 5. num_challenges normalized
  f_nchal = jnp.reshape(state.num_challenges.astype(jnp.float32) / P, (1,))
  # 6. is_rebid
  f_rebid = jnp.reshape(state.is_rebid.astype(jnp.float32), (1,))
  # 7. bidding depth
  depth = jnp.where(
      has_bid,
      (state.current_bid_action + 1).astype(jnp.float32) / (max_bids + 1),
      0.0,
  )
  f_depth = jnp.reshape(depth, (1,))

  # Precompute per-bid-id decode + per-digit membership mask.
  bids = jnp.arange(max_bids, dtype=jnp.int32)
  counts = bids // D + 1                       # [max_bids]
  numbers = bids % D + 1                        # [max_bids]
  digit_mask = (
      numbers[:, None] == jnp.arange(1, D + 1, dtype=jnp.int32)[None, :]
  )                                             # [max_bids, D]

  # 8. per-digit highest count ever bid by ANYONE, normalized by TC.
  placed = state.bid_history.sum(axis=1) > 0    # [max_bids]
  per_digit_max = jnp.max(
      jnp.where(placed[:, None] & digit_mask, counts[:, None], 0), axis=0
  ).astype(jnp.float32) / TC                     # [D]

  # 9. per-digit challenged flag.
  challenged = state.challenge_history.sum(axis=1) > 0  # [max_bids]
  per_digit_chal = jnp.max(
      jnp.where(challenged[:, None] & digit_mask, 1, 0), axis=0
  ).astype(jnp.float32)                          # [D]

  return jnp.concatenate([
      f_has_bid,
      f_count,
      number_oh,
      orig_oh,
      f_nchal,
      f_rebid,
      f_depth,
      per_digit_max,
      per_digit_chal,
  ])


def observation_tensor(
    state: LiarsPokerState,
    config: LiarsPokerConfig,
    player: chex.Array,
    include_private_hand: bool = True,
    include_public: bool = True,
    include_history: bool = True,
    history_encoding: str = "sparse",
) -> chex.Array:
  """Returns a flat float32 observation tensor from the PoV of `player`.

  Matches LiarsPokerObserver.set_from() exactly.  The flags select which
  pieces to include, mirroring the IIGObservationType options:

    - Default (public, no history):
        ``observation_tensor(state, config, player)``
        → player + private_hand + rebid_state + counts_state

    - Perfect-recall (public + history):
        ``observation_tensor(state, config, player, include_history=True)``
        → adds bid_history and challenge_history

  Args:
    state:               The game state.
    config:              Static game configuration.
    player:              Scalar int32 JAX array — the observing player.
    include_private_hand: Include the player's own hand digits.
    include_public:      Include rebid/counts flags (and optionally history).
    include_history:     Include bid/challenge histories (requires
                         include_public=True).

  Returns:
    float32 array of length ``observation_size(config, ...)``.
  """
  pieces = []

  # --- player one-hot ---
  player_onehot = (
      jnp.arange(config.num_players, dtype=jnp.int32) == player
  ).astype(jnp.float32)
  pieces.append(player_onehot)

  # --- private hand ---
  # Only populated once dealing is complete (matches original: hand included
  # only when len(state.hands[player]) == hand_length).
  if include_private_hand:
    dealing_done = state.deal_step >= config.total_cards
    # Build histogram: count occurrences of each digit value 1..num_digits.
    # hands[player] values are in {0, 1, ..., num_digits} where 0 = undealt.
    # The histogram encoding collapses the (order-irrelevant) hand into per-digit
    # counts, shrinking the effective hand space by up to num_digits!.
    digit_ids = jnp.arange(1, config.num_digits + 1, dtype=jnp.int32)  # [D]
    hand_vals = state.hands[player]  # [hand_length], values 1..D (0 if undealt)
    histogram = jnp.sum(
        hand_vals[:, None] == digit_ids[None, :],  # [H, D]
        axis=0,
    ).astype(jnp.float32)  # [D], counts of each digit
    if config.hand_encoding == "raw":
      # OpenSpiel-compatible: the hand's digit values, in order.
      filled = hand_vals.astype(jnp.float32)
      hand = jnp.where(
          dealing_done, filled,
          jnp.zeros(config.hand_length, dtype=jnp.float32))
    else:
      hand = jnp.where(
          dealing_done,
          histogram,
          jnp.zeros(config.num_digits, dtype=jnp.float32),
      )
    pieces.append(hand)

  # --- public pieces ---
  if include_public:
    pieces.append(
        jnp.array([state.is_rebid.astype(jnp.float32)], dtype=jnp.float32)
    )
    pieces.append(
        jnp.array(
            [is_terminal(state).astype(jnp.float32)], dtype=jnp.float32
        )
    )

    if include_history:
      if history_encoding == "compact":
        pieces.append(_compact_history(state, config, player))
      else:
        pieces.append(state.bid_history.astype(jnp.float32).ravel())
        pieces.append(state.challenge_history.astype(jnp.float32).ravel())

  return jnp.concatenate(pieces)


# ---------------------------------------------------------------------------
# Structured observation for the sequence transformer (LPTransformer)
# ---------------------------------------------------------------------------


def structured_observation(
    state: LiarsPokerState,
    config: LiarsPokerConfig,
    player: chex.Array,
    max_seq: int = None,
    net_config: LiarsPokerConfig = None,
) -> LPInputs:
  """Structured (per-token) observation for the LPTransformer, PoV of `player`.

  Unlike ``observation_tensor`` (which flattens everything into one vector),
  this returns the ORDERED sequence of actual bids plus the observer's hand
  histogram and a few global scalars, packaged as an :class:`LPInputs` pytree.
  All ops are pure JAX so the function is jittable and vmappable over a batch
  (leading dim added by ``jax.vmap``).

  Args:
    state:   The game state.
    config:  Static game configuration.
    player:  Scalar int32 JAX array -- the observing/acting player.
    max_seq: Number of bid rows in the output sequence (static Python int).
             Defaults to ``config.max_bids`` (a loose upper bound). A smaller
             value is a valid efficiency knob (see ``_lpt_max_seq``): if a game
             places more than ``max_seq`` bids the OLDEST ones are dropped and
             only the most-recent ``max_seq`` are kept.

  Returns:
    An :class:`LPInputs` with UNBATCHED leaves:
      hand_hist  [D]        int32  -- per-digit histogram (0 until dealt).
      bids       [max_seq,3] int32 -- (count, number, bidder) per placed bid,
                                       in chronological (ascending bid_id) order,
                                       padded with 0 rows.
      bid_valid  [max_seq]  bool   -- True for the first K (real) bid rows.
      observer   []         int32  -- the observing player id.
      globals_   [4]        float32-- [is_rebid, num_challenges/P, has_bid,
                                       bidding_depth].
      legal_mask [num_actions] bool -- legal actions (0=challenge, 1..=bids).

  Capacity ("max game") mode (``net_config`` given and larger than ``config``):
    The OUTPUT is sized to ``net_config`` (a bigger max game, e.g. 8x10) while
    the game logic still uses ``config`` (the PLAYED game, e.g. 6x6).  Then:
      * hand_hist has length ``net_config.num_digits`` (D_cap); real digits
        1..D_game are filled, padded digit slots D_game+1..D_cap stay 0 (the
        network masks them out of attention via its ``game_num_digits`` field).
      * legal_mask has length ``net_config.num_actions`` in the CAPACITY action
        space (action id decodes via D_cap).  Only the played game's legal
        actions are True; capacity actions whose (count, number) fall outside
        the played game (number > D_game or count > total_cards_game) are
        masked out.  The chosen capacity action must be mapped back to a game
        action id before ``apply_action`` (see ``cap_to_game_action_map``).
    When ``net_config`` is None (or equals ``config``) this is byte-identical
    to the played game (no capacity).
  """
  if net_config is None:
    net_config = config
  D = config.num_digits            # played-game digit count (D_game)
  Dcap = net_config.num_digits     # capacity digit count (== D_game if no cap)
  P = config.num_players
  TC = config.total_cards
  max_bids = config.max_bids
  if max_seq is None:
    max_seq = max_bids

  # --- hand histogram (masked to 0 until dealing is complete) --------------
  # Sized to the CAPACITY digit count Dcap.  Real hand values are 1..D_game, so
  # padded slots D_game+1..Dcap are naturally 0 (never counted); the network
  # masks those non-existent slots out of attention.
  dealing_done = state.deal_step >= config.total_cards
  digit_ids = jnp.arange(1, Dcap + 1, dtype=jnp.int32)  # [Dcap]
  hand_vals = state.hands[player]                    # [hand_length], 1..D (0=undealt)
  histogram = jnp.sum(
      hand_vals[:, None] == digit_ids[None, :], axis=0
  ).astype(jnp.int32)                                # [Dcap]
  hand_hist = jnp.where(
      dealing_done, histogram, jnp.zeros(Dcap, dtype=jnp.int32)
  )

  # --- ordered sequence of actual bids -------------------------------------
  # bids strictly increase, so ascending bid_id == chronological order.
  bid_ids = jnp.arange(max_bids, dtype=jnp.int32)
  counts, numbers = decode_bid(bid_ids, config)      # each [max_bids]
  bidders = jnp.argmax(state.bid_history, axis=1).astype(jnp.int32)  # [max_bids]
  placed = state.bid_history.sum(axis=1) > 0         # [max_bids] bool

  # Target output row for each placed bid = #placed bids up to and including it,
  # minus 1 (a compacting prefix-sum). Non-placed rows are routed out of range
  # (max_seq) and dropped by the scatter.
  #
  # DROP-OLDEST on overflow: if a game places more than `max_seq` bids we keep
  # the most RECENT `max_seq` (recency matters most in Liar's Poker). We shift
  # every placed bid's row left by `shift = num_placed - max_seq`, so the oldest
  # bids fall to NEGATIVE rows and are dropped by the scatter alongside the
  # non-placed rows (`.at[].set(mode="drop")` drops both < 0 and >= max_seq).
  # This is rare: at 6x6 games seldom exceed ~15 bids and max_seq defaults to 24.
  # When num_placed <= max_seq the shift is 0, i.e. IDENTICAL to the tight bound
  # (max_seq == max_bids) used before this knob existed -- no behavior change on
  # the non-overflow path. NOTE: the KV-cache rollout path cannot evict, so it
  # assumes no overflow; pick max_seq large enough that overflow never occurs.
  num_placed = jnp.sum(placed.astype(jnp.int32))          # scalar
  shift = jnp.maximum(num_placed - max_seq, 0)
  target_row = jnp.cumsum(placed.astype(jnp.int32)) - 1 - shift  # [max_bids]
  row_idx = jnp.where(placed, target_row, max_seq)        # non-placed -> drop
  vals = jnp.stack([counts, numbers, bidders], axis=1)    # [max_bids, 3]
  bids = jnp.zeros((max_seq, 3), dtype=jnp.int32)
  bids = bids.at[row_idx].set(vals, mode="drop")

  kept = jnp.minimum(num_placed, max_seq)                 # rows actually filled
  bid_valid = jnp.arange(max_seq, dtype=jnp.int32) < kept

  # --- global scalars ------------------------------------------------------
  has_bid = state.current_bid_action >= 0
  depth = jnp.where(
      has_bid,
      (state.current_bid_action + 1).astype(jnp.float32) / (max_bids + 1),
      0.0,
  )
  globals_ = jnp.array([
      state.is_rebid.astype(jnp.float32),
      state.num_challenges.astype(jnp.float32) / P,
      has_bid.astype(jnp.float32),
      depth,
  ], dtype=jnp.float32)

  # --- legal mask (capacity action space when net_config differs) ----------
  game_legal = legal_actions_mask(state, config)     # [num_actions] (played game)
  if net_config.num_actions == config.num_actions and Dcap == D:
    legal_mask = game_legal
  else:
    # Scatter each played-game legal action into its CAPACITY action slot
    # (index remap is static: game action a -> (count, number) via D_game ->
    # capacity id via D_cap).  Unmapped capacity slots stay False (illegal:
    # out-of-game digits/counts), so the network can never pick them.
    g2c = jnp.asarray(_game_to_cap_action_index(config, net_config))
    legal_mask = (
        jnp.zeros(net_config.num_actions, dtype=game_legal.dtype)
        .at[g2c].set(game_legal)
    )

  return LPInputs(
      hand_hist=hand_hist,
      bids=bids,
      bid_valid=bid_valid,
      observer=jnp.asarray(player, dtype=jnp.int32),
      globals_=globals_,
      legal_mask=legal_mask,
  )


def _game_to_cap_action_index(
    config: LiarsPokerConfig, net_config: LiarsPokerConfig
):
  """Static [num_actions_game] int array: played-game action id -> capacity id.

  Action 0 (challenge) maps to 0.  A bid action ``a`` (1..max_bids_game) decodes
  (via D_game) to (count, number) and re-encodes (via D_cap) to the capacity
  action id ``(count-1)*D_cap + number``.  The map is injective, so scattering a
  legal mask through it never collides.
  """
  import numpy as _np
  Dg = config.num_digits
  Dcap = net_config.num_digits
  a = _np.arange(config.num_actions, dtype=_np.int32)
  bid_id = a - 1
  count = bid_id // Dg + 1
  number = bid_id % Dg + 1
  cap = (count - 1) * Dcap + (number - 1) + 1          # capacity action id
  return _np.where(a == 0, 0, cap).astype(_np.int32)


def cap_to_game_action_map(
    config: LiarsPokerConfig, net_config: LiarsPokerConfig = None
):
  """Static [num_actions_cap] int array: capacity action id -> played-game id.

  Inverse of :func:`_game_to_cap_action_index` on the played game's legal
  range.  A capacity action ``a`` decodes (via D_cap) to (count, number); if it
  is reachable in the played game (number <= D_game and count <= total_cards_game)
  it maps to the game action id ``(count-1)*D_game + number``; otherwise it maps
  to 0 (challenge) as a harmless placeholder -- such actions are always illegal
  (masked in the policy) so they are never sampled.  When ``net_config`` is None
  or equals ``config`` this is the identity ``arange(num_actions)``.
  """
  import numpy as _np
  if net_config is None:
    net_config = config
  # IDENTITY when there is no capacity remapping to do. The docstring always
  # promised this, but the code below only *happened* to produce it: it decodes
  # each index as a bid and sends anything unreachable to 0 (CHALLENGE), on the
  # stated assumption that unreachable actions "are always illegal (masked in
  # the policy) so they are never sampled".
  #
  # dummy_mult > 1 breaks that assumption: dummy actions sit ABOVE the real bid
  # range, are LEGAL, and ARE sampled -- and every one of them decoded to a
  # count > total_cards and was silently remapped to CHALLENGE. Every
  # dummy-probe rollout therefore played challenges instead of dummies (100% of
  # indices 19..151 at 3x3x2 x8), so those experiments measured a policy whose
  # extra outputs were aliases of CHALLENGE, not the intended probe.
  if net_config is config or (
      net_config.num_actions == config.num_actions
      and net_config.num_digits == config.num_digits):
    return _np.arange(config.num_actions, dtype=_np.int32)
  Dg = config.num_digits
  Dcap = net_config.num_digits
  a = _np.arange(net_config.num_actions, dtype=_np.int32)
  bid_id = a - 1
  count = bid_id // Dcap + 1
  number = bid_id % Dcap + 1
  reachable = (number <= Dg) & (count <= config.total_cards)
  game_id = (count - 1) * Dg + (number - 1) + 1
  out = _np.where((a != 0) & reachable, game_id, 0)
  out[0] = 0
  return out.astype(_np.int32)


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------


def returns(
    state: LiarsPokerState, config: LiarsPokerConfig
) -> chex.Array:
  """Returns the reward array of shape [num_players].

  Matches LiarsPokerState.returns() exactly:
    - If the bidder wins: bidder gets (num_players - 1), everyone else gets -1.
    - If the bidder loses: bidder gets -(num_players - 1), everyone else gets +1.
    - Otherwise: all zeros (game not yet over).
  """
  player_ids = jnp.arange(config.num_players, dtype=jnp.int32)
  is_originator = player_ids == state.bid_originator

  winner_rewards = jnp.where(
      is_originator,
      jnp.float32(config.num_players - 1),
      jnp.float32(-1.0),
  )
  loser_rewards = jnp.where(
      is_originator,
      jnp.float32(-(config.num_players - 1)),
      jnp.float32(1.0),
  )
  zero_rewards = jnp.zeros(config.num_players, dtype=jnp.float32)

  return jnp.where(
      state.winner >= 0,
      winner_rewards,
      jnp.where(state.loser >= 0, loser_rewards, zero_rewards),
  )
