"""Causal transformer policy/value network for Liar's Poker.

Replaces the MLP-based RNaDNetwork with a causal transformer that
processes the game history as a sequence of action tokens:

  [PLAYER_ID] [hand_1 ... hand_H] [START] [action_1] [action_2] ...

Each action token is one of:
  - Bid:       player_embed(bidder) + count_embed(count) + digit_embed(digit) + type(BID)
  - Challenge: player_embed(challenger) + type(CHALLENGE)

Uses prefix-causal attention:
  - Prefix (player identity + hand): bidirectional within prefix.
  - Action sequence (START + moves): causal, with full attention to prefix.

Policy and value are read from the last valid action position (or START
if no actions have been taken yet).
"""

from typing import Tuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp

from rnad import (
    EnvStep,
    _legal_policy,
    legal_log_policy,
)


class TransformerBlock(nn.Module):
    """Pre-norm transformer encoder block."""

    d_model: int
    n_heads: int
    d_ff: int = 0  # 0 → defaults to 4 * d_model

    @nn.compact
    def __call__(self, x, mask=None):
        d_ff = self.d_ff or 4 * self.d_model

        # Self-attention with pre-norm
        y = nn.LayerNorm()(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            deterministic=True,
        )(y, y, mask=mask)
        x = x + y

        # FFN with pre-norm
        y = nn.LayerNorm()(x)
        y = nn.Dense(d_ff)(y)
        y = nn.gelu(y)
        y = nn.Dense(self.d_model)(y)
        x = x + y

        return x


# Token type IDs
_TT_PLAYER_ID = 0
_TT_HAND = 1
_TT_START = 2
_TT_BID = 3
_TT_CHALLENGE = 4
_NUM_TOKEN_TYPES = 5


class TransformerPolicyNetwork(nn.Module):
    """Causal transformer policy/value network for Liar's Poker.

    Drop-in replacement for RNaDNetwork: same __call__ signature
    (takes EnvStep, returns (pi, v, log_pi, logit)).

    Sequence layout:
      [PLAYER_ID] [hand_1..hand_H] [START] [move_1] [move_2] ...

    For each bid_id (0..max_bids-1), there are (1 + num_players) move slots:
      - 1 bid token (valid if that bid was made)
      - num_players challenge tokens (valid if that player challenged on this bid)

    Attention is prefix-causal: the prefix (player_id + hand) is bidirectional;
    the action part (START + moves) attends causally to itself and fully to prefix.

    Policy/value are read from the embedding at the last valid action position.

    Attributes:
        num_players:  Number of players in the game.
        hand_length:  Cards per player.
        num_digits:   Distinct digit values (1..num_digits).
        num_actions:  Size of the discrete action space.
        d_model:      Transformer hidden dimension.
        n_heads:      Number of attention heads.
        n_layers:     Number of transformer encoder layers.
        d_ff:         Feed-forward inner dimension (0 → 4 * d_model).
    """

    num_players: int
    hand_length: int
    num_digits: int
    num_actions: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 0

    def setup(self):
        P = self.num_players
        H = self.hand_length
        D = self.num_digits
        max_bids = H * D * P
        max_count = H * P
        moves_per_bid = 1 + P  # 1 bid + P potential challenge slots
        max_moves = max_bids * moves_per_bid

        prefix_len = 1 + H  # PLAYER_ID + hand
        seq_len = prefix_len + 1 + max_moves  # + START + moves

        # --- Value embedding tables ---
        # Digit: index 0 = padding, 1..D = digit values
        self.digit_embed = nn.Embed(D + 1, self.d_model)
        # Count: index 0 = padding, 1..max_count = count values
        self.count_embed = nn.Embed(max_count + 1, self.d_model)
        # Player: index 0..P-1 = players, P = padding/unused
        self.player_embed = nn.Embed(P + 1, self.d_model)

        # Token type embedding (role of each position)
        self.type_embed = nn.Embed(_NUM_TOKEN_TYPES, self.d_model)

        # Learned positional embeddings
        self.pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(0.02),
            (seq_len, self.d_model),
        )

        # START token (learned)
        self.start_embed = self.param(
            "start_embed",
            nn.initializers.normal(0.02),
            (self.d_model,),
        )

        # Project scalar flags (rebid, terminal) → d_model, added to START
        self.flag_proj = nn.Dense(self.d_model)

        # Transformer encoder layers
        self.transformer_layers = [
            TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
            )
            for _ in range(self.n_layers)
        ]

        # Final layer norm (pre-norm architecture)
        self.final_norm = nn.LayerNorm()

        # Output heads
        self.policy_head = nn.Dense(self.num_actions)
        self.value_head = nn.Dense(1)

    def __call__(
        self, env_step: EnvStep
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """Forward pass. Handles both unbatched and batched inputs."""
        if env_step.obs.ndim > 1:
            fn = lambda obs, legal, pid, valid, rew: self._forward_single(
                EnvStep(obs=obs, legal=legal, player_id=pid, valid=valid, rewards=rew)
            )
            return jax.vmap(fn)(
                env_step.obs,
                env_step.legal,
                env_step.player_id,
                env_step.valid,
                env_step.rewards,
            )
        return self._forward_single(env_step)

    def _forward_single(
        self, env_step: EnvStep
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        """Process a single (unbatched) observation."""
        P = self.num_players
        H = self.hand_length
        D = self.num_digits
        max_bids = H * D * P
        moves_per_bid = 1 + P
        max_moves = max_bids * moves_per_bid
        prefix_len = 1 + H
        n_action = 1 + max_moves  # START + moves
        seq_len = prefix_len + n_action

        obs = env_step.obs

        # -----------------------------------------------------------------
        # Parse the flat observation
        # Layout: [player_oh(P) | hand(H) | rebid(1) | counts(1)
        #          | bid_hist(max_bids*P) | chall_hist(max_bids*P)]
        # -----------------------------------------------------------------
        player_id = jnp.argmax(obs[:P]).astype(jnp.int32)
        hand = obs[P : P + H].astype(jnp.int32)  # 0..D (0 = undealt)
        rebid_flag = obs[P + H]
        terminal_flag = obs[P + H + 1]

        hist_off = P + H + 2
        bid_hist = obs[hist_off : hist_off + max_bids * P].reshape(max_bids, P)
        chall_hist = obs[hist_off + max_bids * P :].reshape(max_bids, P)

        # -----------------------------------------------------------------
        # Decode bids → (count, digit, player) per bid_id
        # -----------------------------------------------------------------
        bid_ids = jnp.arange(max_bids, dtype=jnp.int32)
        bid_counts = bid_ids // D + 1  # 1-indexed
        bid_digits = bid_ids % D + 1  # 1-indexed
        bid_players = jnp.argmax(bid_hist, axis=1).astype(jnp.int32)
        bid_valid = bid_hist.sum(axis=1) > 0  # [max_bids]

        # -----------------------------------------------------------------
        # Build prefix tokens: [PLAYER_ID] [hand_1..hand_H]
        # -----------------------------------------------------------------
        player_id_tok = (
            self.player_embed(player_id)
            + self.type_embed(jnp.int32(_TT_PLAYER_ID))
        )  # [d]

        hand_tok = (
            self.digit_embed(hand)
            + self.type_embed(jnp.full(H, _TT_HAND, dtype=jnp.int32))
        )  # [H, d]

        # -----------------------------------------------------------------
        # Build START token (carries rebid/terminal flags)
        # -----------------------------------------------------------------
        flags = jnp.stack([rebid_flag, terminal_flag])
        start_tok = (
            self.start_embed
            + self.flag_proj(flags)
            + self.type_embed(jnp.int32(_TT_START))
        )  # [d]

        # -----------------------------------------------------------------
        # Build action tokens
        # -----------------------------------------------------------------
        # Bid tokens: player + count + digit + type(BID)
        bid_tok = (
            self.player_embed(bid_players)
            + self.count_embed(bid_counts)
            + self.digit_embed(bid_digits)
            + self.type_embed(jnp.full(max_bids, _TT_BID, dtype=jnp.int32))
        )  # [max_bids, d]

        # Challenge tokens: player + type(CHALLENGE)
        # Shape: [max_bids, P, d]
        player_indices = jnp.broadcast_to(
            jnp.arange(P, dtype=jnp.int32)[None, :], (max_bids, P)
        )
        chall_tok = (
            self.player_embed(player_indices.reshape(-1)).reshape(
                max_bids, P, self.d_model
            )
            + self.type_embed(
                jnp.full((max_bids, P), _TT_CHALLENGE, dtype=jnp.int32)
            )
        )  # [max_bids, P, d]

        chall_valid = (chall_hist > 0) & bid_valid[:, None]  # [max_bids, P]

        # Interleave: [bid_0, chall_0_p0, chall_0_p1, bid_1, ...]
        # Shape: [max_bids, 1+P, d]
        move_section = jnp.concatenate(
            [bid_tok[:, None, :], chall_tok], axis=1
        )
        move_flat = move_section.reshape(max_moves, self.d_model)

        # Move validity: [max_bids, 1+P]
        move_valid_section = jnp.concatenate(
            [bid_valid[:, None], chall_valid], axis=1
        )
        move_flat_valid = move_valid_section.reshape(max_moves)

        # -----------------------------------------------------------------
        # Assemble full sequence [seq_len, d]
        # -----------------------------------------------------------------
        seq = jnp.concatenate(
            [
                player_id_tok[None, :],  # [1, d]
                hand_tok,                # [H, d]
                start_tok[None, :],      # [1, d]
                move_flat,               # [max_moves, d]
            ],
            axis=0,
        )
        seq = seq + self.pos_embed

        # -----------------------------------------------------------------
        # Validity mask
        # -----------------------------------------------------------------
        hand_valid = hand > 0  # [H]

        valid = jnp.concatenate(
            [
                jnp.ones(1, dtype=jnp.bool_),          # PLAYER_ID: always
                hand_valid,                              # [H]
                jnp.ones(1, dtype=jnp.bool_),          # START: always
                move_flat_valid.astype(jnp.bool_),     # [max_moves]
            ]
        )  # [seq_len]

        # -----------------------------------------------------------------
        # Prefix-causal attention mask
        # -----------------------------------------------------------------
        # Prefix (player_id + hand): bidirectional within prefix
        # Action (START + moves): causal within actions, full to prefix
        mask = jnp.zeros((seq_len, seq_len), dtype=jnp.bool_)

        # Prefix ↔ prefix: bidirectional
        mask = mask.at[:prefix_len, :prefix_len].set(True)
        # Action → prefix: full attention to prefix
        mask = mask.at[prefix_len:, :prefix_len].set(True)
        # Action → action: causal (lower-triangular)
        mask = mask.at[prefix_len:, prefix_len:].set(
            jnp.tril(jnp.ones((n_action, n_action), dtype=jnp.bool_))
        )

        # Apply validity: both source and target must be valid
        mask = mask & valid[:, None] & valid[None, :]

        # -----------------------------------------------------------------
        # Transformer forward pass
        # -----------------------------------------------------------------
        x = seq
        for layer in self.transformer_layers:
            x = layer(x, mask=mask)

        x = self.final_norm(x)

        # -----------------------------------------------------------------
        # Readout from last valid action position
        # -----------------------------------------------------------------
        # Action positions start at prefix_len (START is first, always valid)
        action_valid = valid[prefix_len:]  # [n_action]
        # Find the last valid position (reversed argmax trick)
        last_offset = n_action - 1 - jnp.argmax(jnp.flip(action_valid))
        read_idx = prefix_len + last_offset
        readout = x[read_idx]

        # -----------------------------------------------------------------
        # Policy and value heads
        # -----------------------------------------------------------------
        logit = self.policy_head(readout)
        v = self.value_head(readout)
        pi = _legal_policy(logit, env_step.legal)
        log_pi = legal_log_policy(logit, env_step.legal)

        return pi, v, log_pi, logit
