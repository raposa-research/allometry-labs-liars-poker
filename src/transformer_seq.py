"""Standalone flax transformer for Liar's Poker as sequence modeling.

`LPTransformer` treats a Liar's Poker decision as autoregressive sequence
modeling over the ordered sequence of *actual* bids (not the flat observation
vector).  The token sequence is

    [ D hand tokens ]  [ L bid tokens ]  [ 1 DECISION token ]

and the policy/value are read from the DECISION token, which attends over the
hand tokens and the valid (non-padding) bid tokens.

This module is intentionally STANDALONE: it consumes STRUCTURED inputs
(per-digit hand counts, the ordered list of bids, globals, legal mask) rather
than the flat obs vector used by `rnad.RNaDNetwork` / `FactoredMLPNetwork`.
Wiring it into the training pipeline (EnvStep, R-NaD losses) is a separate,
future task and is deliberately NOT done here.

Design notes (implemented exactly as documented):

  * Weight tying.  A single `NumberEmbed = nn.Embed(D+1, d_model)` and a single
    smooth `CountEncoder` (`count_enc`) instance are reused for BOTH the history
    tokens (hand/bid tokens) AND the per-action decode keys.  A shared
    `TypeEmbed = nn.Embed(3, d_model)` distinguishes {HAND, BID, DECISION/ACTION}.

  * count_enc is a SMOOTH ordinal encoder (not a lookup): it maps an integer
    count `c` to `[c/total_cards, sin(c*w_k), cos(c*w_k) ...]` for a few
    geometric frequencies `w_k`, then a shared `Dense(d_model)`.  This shares
    structure across neighbouring counts (differentiable in c).

  * RoPE is applied to the Q/K vectors inside attention (rotate-half /
    GPT-NeoX style), keyed by an integer position per token.  Hand tokens share
    position 0 (rotation angle 0 == identity == "no RoPE on hands", making the
    hand set order-invariant).  Bid token i (0-based) gets position i+1.  The
    DECISION token gets position (num_valid_bids + 1), i.e. it sits just after
    the last real bid -- this choice is what makes full-forward and incremental
    KV-cache decoding produce identical results (see `decode`/tests).

  * Body: causal pre-norm transformer (MHA + MLP, residual, LayerNorm).  Hand
    tokens attend only among themselves (bidirectional prefix, always present).
    Bid token i attends to all hands + bids 0..i (causal).  The DECISION token
    attends to all hands + all valid bids + itself.  Padding bid tokens are
    masked as keys everywhere.

  * Head: tied dot-product.  Action keys are precomputed from the SHARED
    embeddings -- action 0 (challenge) is a learned `challenge_embed`; action
    a=1..max_bids decodes to (count,number) via count=(a-1)//D+1,
    number=(a-1)%D+1 and its key is TypeEmbed[ACTION] + NumberEmbed[number] +
    count_enc(count).  logits = (Dense(h) @ keys.T)/sqrt(d_model), then illegal
    actions are set to -1e9.  value = MLP(h) -> [B,1].

  * KV cache (decode mode).  A functional cache (a per-layer list of
    (K, V) arrays of shape [B, n_heads, T, d_head], with RoPE already baked into
    the cached K) is threaded through three methods: `prefill` (hands, run once),
    `append_bid` (append ONE bid's K/V through the stack), and `decision_logits`
    (run the transient DECISION token as a length-1 query against the cache).
"""

from typing import Any, List, Optional, Tuple

import chex
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

# Compute dtype for the transformer trunk.  Parameters are ALWAYS stored in
# fp32 (``param_dtype``); only the matmuls / activations run in this dtype
# (bf16 for the training config).  The head outputs (logits, value) and the
# attention softmax are cast back to fp32 so the R-NaD loss / v-trace stay fp32.
Dtype = Any

# Token type ids (shared TypeEmbed indices).
TYPE_HAND = 0
TYPE_BID = 1
TYPE_ACTION = 2  # used for both the DECISION token and the action decode keys.
# TYPE_DIGIT_DECISION is used ONLY by the 'perdigit_tailprob' head (the per-digit
# decision tokens).  It is injected via a SEPARATE learned param
# (``digit_dec_type_embed``) created only in that head, NOT by widening
# ``type_embed`` (which would change the legacy param tree).  Kept here as a
# documentation marker of the token kind.
TYPE_DIGIT_DECISION = 3

NEG_INF = -1e9


# --------------------------------------------------------------------------- #
# Game-size-INVARIANT count / belief tables (exact binomial, closed form).
#
# Used ONLY by the 'perdigit_tailprob' head.  Both tables are derived from the
# STATIC game config (num_players, hand_length, num_digits), so they recompute
# per game size -- there is NO per-count / per-digit learned parameter.  This is
# what makes the head transfer across counts / num_players: a count ``c`` enters
# the network only through the scalar ``q(c) = log P(total_count >= c)`` and a
# per-(digit,count) belief ``log P(others_count_d >= c - h_d | hand)``, both of
# which are just gathers into these precomputed log-survival tables.
# --------------------------------------------------------------------------- #
def _binom_survival_table(pool: int, num_digits: int) -> np.ndarray:
  """Survival table ``S[t] = P(X >= t)`` for ``X ~ Binomial(pool, 1/num_digits)``.

  Length ``pool + 2`` (t = 0 .. pool+1); ``S[0] == 1`` and ``S[pool+1] == 0`` by
  construction.  Built by the exact iterative pmf recurrence (no gammaln/scipy):
  ``pmf[0] = (1-p)^pool``, ``pmf[k+1] = pmf[k] * (pool-k)/(k+1) * p/(1-p)``.
  """
  p = 1.0 / float(num_digits)
  pmf = np.zeros(pool + 1, dtype=np.float64)
  pmf[0] = (1.0 - p) ** pool
  for k in range(pool):
    pmf[k + 1] = pmf[k] * (pool - k) / (k + 1) * (p / (1.0 - p))
  pmf = np.clip(pmf, 0.0, None)
  ge = np.zeros(pool + 2, dtype=np.float64)
  for t in range(pool + 1):
    ge[t] = pmf[t:].sum()
  ge[0] = 1.0
  ge[pool + 1] = 0.0
  return ge


def _log_survival_table(pool: int, num_digits: int, clamp: float) -> np.ndarray:
  """``log P(X >= t)`` for ``X ~ Binomial(pool, 1/num_digits)`` -> [pool+2] fp32.

  ``log S[t]``, clamped to ``[-clamp, 0]`` (the deep tail underflows to 0 in the
  survival table -> ``-inf``; clamping keeps a finite, differentiable logit).
  ``t == 0`` -> ``log 1 == 0`` exactly.
  """
  s = _binom_survival_table(pool, num_digits)
  ls = np.where(s > 0.0, np.log(np.clip(s, 1e-300, None)), -float(clamp))
  ls = np.clip(ls, -float(clamp), 0.0)
  return ls.astype(np.float32)


@flax.struct.dataclass
class LPInputs:
  """Structured, batched (leading dim B) inputs for a full parallel forward.

  Attributes:
    hand_hist: [B, D] int   -- observer's per-digit counts (0..hand_length).
    bids:      [B, L, 3] int -- ordered ACTUAL bids: (count c in 1..max_count,
               number n in 1..D, bidder_player p in 0..P-1).  Padded with 0s.
    bid_valid: [B, L] bool  -- which bid rows are real (not padding).
    observer:  [B] int      -- acting/observing player id.
    globals_:  [B, G] float -- scalar state (is_rebid, num_challenges/P,
               has_bid, bidding_depth, ...).
    legal_mask: [B, num_actions] bool -- legal actions (0=challenge, 1..=bids).
  """
  hand_hist: chex.Array
  bids: chex.Array
  bid_valid: chex.Array
  observer: chex.Array
  globals_: chex.Array
  legal_mask: chex.Array


# --------------------------------------------------------------------------- #
# RoPE helpers.
# --------------------------------------------------------------------------- #
def _rope_cos_sin(pos: chex.Array, d_head: int, base: float = 10000.0):
  """Return (cos, sin) of shape [B, 1, T, d_head] for positions `pos` [B, T]."""
  half = d_head // 2
  inv_freq = 1.0 / (base ** (np.arange(0, half, dtype=np.float32) / half))
  inv_freq = jnp.asarray(inv_freq)  # [half]
  ang = pos.astype(jnp.float32)[..., None] * inv_freq  # [B, T, half]
  cos = jnp.cos(ang)
  sin = jnp.sin(ang)
  cos = jnp.concatenate([cos, cos], axis=-1)  # [B, T, d_head]
  sin = jnp.concatenate([sin, sin], axis=-1)
  return cos[:, None, :, :], sin[:, None, :, :]  # [B, 1, T, d_head]


def _rotate_half(x: chex.Array) -> chex.Array:
  half = x.shape[-1] // 2
  x1, x2 = x[..., :half], x[..., half:]
  return jnp.concatenate([-x2, x1], axis=-1)


def _apply_rope(x: chex.Array, pos: chex.Array) -> chex.Array:
  """Apply RoPE to x [B, H, T, d_head] using integer positions pos [B, T]."""
  cos, sin = _rope_cos_sin(pos, x.shape[-1])
  # cos/sin are computed in fp32; match x's compute dtype so RoPE keeps the
  # q/k tensors in bf16 (no silent upcast of the attention inputs).
  cos = cos.astype(x.dtype)
  sin = sin.astype(x.dtype)
  return x * cos + _rotate_half(x) * sin


def _apply_rope_2d(x: chex.Array, seq_pos: chex.Array, digit_pos: chex.Array,
                   d_seq: int) -> chex.Array:
  """2D RoPE for the 'rope_digit' number encoding (fMHA-compatible).

  Partition each head's dim (``d_head``) into two contiguous rotary sub-blocks
  and rotate each with a DIFFERENT integer position:

    * ``x[..., :d_seq]`` -- the existing SEQUENCE-position rotary (bid order),
      driven by ``seq_pos`` (hands=0, bid i -> i+1, decision -> num_bids+1).
    * ``x[..., d_seq:]`` -- a NEW DIGIT-index rotary, driven by ``digit_pos``
      (hand token for digit d -> d, bid about number n -> n, decision/null -> 0).

  Standard rotate-half RoPE is applied WITHIN each sub-block (each must be even),
  so ``Q.K`` accumulates BOTH a relative-sequence-position factor (first slice)
  AND a relative-DIGIT-position factor (second slice).  The digit factor is
  ``sum_pairs |content|^2 cos((qd-kd)*w)`` -> maximised at digit offset 0, giving
  an exact same-digit attention peak.  Null-digit tokens (``digit_pos==0``) get
  angle 0 == identity rotation on the digit slice, so they carry no digit phase.

  Applied to Q/K ONLY (never V), so the attention stays the plain fused
  ``softmax(QK/scale)`` form (no additive bias, no extra where-select).
  """
  x_seq = _apply_rope(x[..., :d_seq], seq_pos)
  x_dig = _apply_rope(x[..., d_seq:], digit_pos)
  return jnp.concatenate([x_seq, x_dig], axis=-1)


# --------------------------------------------------------------------------- #
# Smooth ordinal count encoder (shared instance, weight-tied).
# --------------------------------------------------------------------------- #
class CountEncoder(nn.Module):
  """Smooth ordinal encoding of an integer count -> [..., d_model].

  feats = [c/total_cards, sin(c*w_k), cos(c*w_k) for k=0..F-1] -> Dense(d_model).
  Frequencies span periods 2 .. 2*max_count geometrically (smooth in c).
  """
  d_model: int
  total_cards: int
  max_count: int
  num_freqs: int = 6

  @nn.compact
  def __call__(self, c: chex.Array) -> chex.Array:
    c = c.astype(jnp.float32)
    ks = np.arange(self.num_freqs, dtype=np.float32)
    denom = max(self.num_freqs - 1, 1)
    # k=0 -> pi (period 2), k=F-1 -> pi/max_count (period 2*max_count).
    w = np.pi * (float(self.max_count) ** (-(ks) / denom))  # [F]
    w = jnp.asarray(w)
    ang = c[..., None] * w  # [..., F]
    feats = jnp.concatenate(
        [(c / self.total_cards)[..., None], jnp.sin(ang), jnp.cos(ang)],
        axis=-1)  # [..., 1 + 2F]
    return nn.Dense(self.d_model)(feats)


class CountMLP(nn.Module):
  """Game-size-INVARIANT count encoder for the 'perdigit_tailprob' head.

  Maps the SCALAR tail-probability feature ``q(c) = log P(total_count >= c)``
  (total_count ~ Binomial(POOL, 1/D)) to ``[..., d_model]`` via a small SHARED
  1-hidden-layer MLP.  The count ``c`` enters ONLY through ``q(c)`` -- a scalar
  that is comparable across games (a count's *rarity* under the deal, not its raw
  integer) -- so the same learned function transfers across counts / num_players
  / hand_length.  Parameter-free in game size (only ``d_model`` and ``hidden``
  set the shapes).
  """
  d_model: int
  hidden: int = 0  # 0 -> d_model.

  @nn.compact
  def __call__(self, q: chex.Array) -> chex.Array:
    # q: [...] scalar tail feature.  -> [..., d_model].
    h = self.hidden or self.d_model
    x = q[..., None]  # [..., 1]
    x = nn.gelu(nn.Dense(h)(x))
    return nn.Dense(self.d_model)(x)


# --------------------------------------------------------------------------- #
# Transformer block (pre-norm), with helpers for full + incremental decode.
# --------------------------------------------------------------------------- #
class TransformerBlock(nn.Module):
  d_model: int
  n_heads: int
  d_ff: int
  dtype: Dtype = jnp.float32  # compute dtype; params stay fp32.

  def setup(self):
    # ``dtype`` = compute dtype (bf16 for training), ``param_dtype`` = fp32 so
    # weights are stored/updated in full precision while matmuls run in bf16.
    kw = dict(dtype=self.dtype, param_dtype=jnp.float32)
    self.ln1 = nn.LayerNorm(**kw)
    self.ln2 = nn.LayerNorm(**kw)
    self.qp = nn.Dense(self.d_model, **kw)
    self.kp = nn.Dense(self.d_model, **kw)
    self.vp = nn.Dense(self.d_model, **kw)
    self.op = nn.Dense(self.d_model, **kw)
    self.f1 = nn.Dense(self.d_ff, **kw)
    self.f2 = nn.Dense(self.d_model, **kw)

  @property
  def d_head(self) -> int:
    return self.d_model // self.n_heads

  def _rope_dims(self) -> Tuple[int, int]:
    """(d_seq, d_digit) partition of d_head for the 2D 'rope_digit' RoPE.

    Half/half, with each sub-block forced EVEN (RoPE rotates dim PAIRS):
    ``d_digit = 2*(d_head//4)`` (~d_head/2, even) and ``d_seq = d_head-d_digit``
    (even since d_head is even).  d_head divisible by 4 -> exactly half/half.
    """
    dh = self.d_head
    d_digit = 2 * (dh // 4)
    return dh - d_digit, d_digit

  def _split(self, t: chex.Array) -> chex.Array:
    # [B, T, d_model] -> [B, H, T, d_head]
    b, tlen, _ = t.shape
    return t.reshape(b, tlen, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

  def _merge(self, t: chex.Array) -> chex.Array:
    # [B, H, T, d_head] -> [B, T, d_model]
    b, h, tlen, dh = t.shape
    return t.transpose(0, 2, 1, 3).reshape(b, tlen, h * dh)

  def _mlp(self, x: chex.Array) -> chex.Array:
    y = self.ln2(x)
    y = self.f2(nn.gelu(self.f1(y)))
    return x + y

  def full(self, x: chex.Array, mask: chex.Array, pos: chex.Array,
           attn_bias: chex.Array = None, combined_bias: chex.Array = None,
           digit_pos: chex.Array = None):
    """Full self-attention over the whole sequence.

    Args:
      x: [B, T, d_model]
      mask: [B, T, T] bool -- mask[b,q,k] True if q may attend to k.  Ignored
        when ``combined_bias`` is supplied (the mask is baked into it).
      pos: [B, T] int RoPE positions.
      attn_bias: optional [B, H, T, T] fp32 additive bias on the attention
        logits (the relative-digit bias; ``None`` in categorical mode -> no-op,
        so the categorical path is byte-identical).  Added BEFORE the structural
        mask, so masked (illegal / padding) keys stay at ``NEG_INF``.
      combined_bias: optional [B, H, T, T] fp32 additive term that ALREADY folds
        BOTH the relative-digit bias AND the structural/validity mask (0 where
        allowed, ``NEG_INF`` where masked).  When supplied, the attention is the
        single-additive-bias form ``softmax(QK/scale + combined_bias)`` with NO
        separate ``where``-select -- this is the shape XLA's fused-attention
        (cuDNN fMHA) rewrite matches on GPU, so the digit-bias path stays fused
        instead of falling back to an unfused, memory-bound attention.  It is
        numerically identical to the ``attn_bias``+``where`` path up to fp
        rounding (masked logits underflow to 0 after softmax either way), so the
        full-vs-cache equivalence is preserved.  Precomputed ONCE per forward
        (mask + bias are both layer-independent) and reused across layers.
    Returns:
      (x_out [B,T,d_model], k [B,H,T,d_head], v [B,H,T,d_head])
      The returned k,v (RoPE already applied to k) are used to seed the cache.
    """
    y = self.ln1(x)
    if digit_pos is not None:
      # 2D 'rope_digit' RoPE: sequence rotary + a second digit-index rotary on
      # a dim-slice.  V is NEVER rotated -> fused softmax(QK/scale) structure.
      d_seq, _ = self._rope_dims()
      q = _apply_rope_2d(self._split(self.qp(y)), pos, digit_pos, d_seq)
      k = _apply_rope_2d(self._split(self.kp(y)), pos, digit_pos, d_seq)
    else:
      q = _apply_rope(self._split(self.qp(y)), pos)
      k = _apply_rope(self._split(self.kp(y)), pos)
    v = self._split(self.vp(y))
    scale = jnp.sqrt(jnp.asarray(self.d_head, jnp.float32))
    # QK^T runs in the compute dtype (bf16); softmax is done in fp32 for
    # numerical stability, then attn is cast back to bf16 for the AV matmul.
    scores = (jnp.einsum('bhqd,bhkd->bhqk', q, k) / scale).astype(jnp.float32)
    if combined_bias is not None:
      # Single fused additive bias (digit bias + additive mask): no separate
      # select, so the GPU fused-attention kernel keeps matching.
      scores = scores + combined_bias
    else:
      if attn_bias is not None:
        scores = scores + attn_bias.astype(jnp.float32)
      scores = jnp.where(mask[:, None, :, :], scores, NEG_INF)
    attn = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
    o = jnp.einsum('bhqk,bhkd->bhqd', attn, v)
    x = x + self.op(self._merge(o))
    x = self._mlp(x)
    return x, k, v

  def step(self, x: chex.Array, pos: chex.Array,
           cache_k: chex.Array, cache_v: chex.Array,
           key_valid: chex.Array = None, attn_bias: chex.Array = None,
           digit_pos: chex.Array = None, self_mask: chex.Array = None):
    """Incremental step for a small block of new query tokens (T_q>=1).

    The new tokens attend to concat(cache, own) -- i.e. all cached (earlier,
    valid) tokens plus themselves, causally among themselves.

    Args:
      x: [B, T_q, d_model] new token(s).
      pos: [B, T_q] positions of the new token(s).
      cache_k, cache_v: [B, H, T_c, d_head] cached keys/values (RoPE in k).
      key_valid: optional [B, T_c] bool mask -- which CACHED key slots are real.
        ``None`` (the default, used by the growing-cache `append_bid` /
        `decision_logits` path) means every cached slot is valid.  A mask is
        supplied by the FIXED-SIZE preallocated-cache path (`*_fixed` methods,
        Stage 2 rollout), where the cache is padded out to a static length and
        only the first ``counter`` bid slots are real.
      attn_bias: optional [B, H, T_q, T_c+T_q] fp32 additive bias on the
        attention logits over keys ``concat(cache, own)`` (the relative-digit
        bias).  ``None`` in categorical mode -> byte-identical.  Added BEFORE the
        validity/causal mask so masked keys stay at ``NEG_INF``.
      digit_pos: optional [B, T_q] int DIGIT index of the NEW token(s) for the
        2D 'rope_digit' RoPE (baked into own_k/q).  ``None`` (categorical /
        relative_bias) -> the original single-axis sequence RoPE (byte-
        identical).  Cached keys already carry their own baked digit phase, so
        no per-cached-key digit id is needed here.
    Returns:
      (x_out [B,T_q,d_model], own_k [B,H,T_q,d_head], own_v [B,H,T_q,d_head])
    """
    b = x.shape[0]
    y = self.ln1(x)
    if digit_pos is not None:
      # 2D 'rope_digit' RoPE on the NEW token(s).  The digit phase is BAKED into
      # own_k here, so cached keys already carry their own digit phase -> no
      # per-key digit id needs to be threaded at attention time (unlike the
      # additive relative_bias).  V is never rotated.
      d_seq, _ = self._rope_dims()
      q = _apply_rope_2d(self._split(self.qp(y)), pos, digit_pos, d_seq)
      own_k = _apply_rope_2d(self._split(self.kp(y)), pos, digit_pos, d_seq)
    else:
      q = _apply_rope(self._split(self.qp(y)), pos)
      own_k = _apply_rope(self._split(self.kp(y)), pos)
    own_v = self._split(self.vp(y))
    k = jnp.concatenate([cache_k, own_k], axis=2)
    v = jnp.concatenate([cache_v, own_v], axis=2)
    tq = q.shape[2]
    tc = cache_k.shape[2]
    scale = jnp.sqrt(jnp.asarray(self.d_head, jnp.float32))
    # QK^T in compute dtype (bf16); softmax in fp32; AV matmul back in bf16.
    scores = (jnp.einsum('bhqd,bhkd->bhqk', q, k) / scale).astype(jnp.float32)
    if attn_bias is not None:
      scores = scores + attn_bias.astype(jnp.float32)
    # Mask among the new tokens; cached tokens are visible if valid.  The default
    # is CAUSAL (``ki <= qi``); the 'perdigit_tailprob' decision block passes an
    # IDENTITY ``self_mask`` so its D+1 query tokens (global + per-digit) attend
    # ONLY to the cache + themselves (never to each other), matching the
    # full-forward decision-token mask (each dec token sees itself only).
    qi = np.arange(tq)[:, None]
    ki = np.arange(tq)[None, :]
    if self_mask is None:
      self_mask = jnp.asarray(ki <= qi)  # [Tq, Tq]
    if key_valid is None:
      cache_mask = jnp.ones((tq, tc), dtype=bool)
      full_mask = jnp.concatenate([cache_mask, self_mask], axis=1)  # [Tq,Tc+Tq]
      scores = jnp.where(full_mask[None, None, :, :], scores, NEG_INF)
    else:
      # Per-env cache validity: [B, T_c] -> broadcast over the T_q query axis.
      cache_mask = jnp.broadcast_to(key_valid[:, None, :], (b, tq, tc))
      self_mask_b = jnp.broadcast_to(self_mask[None, :, :], (b, tq, tq))
      full_mask = jnp.concatenate([cache_mask, self_mask_b], axis=2)  # [B,Tq,Tc+Tq]
      scores = jnp.where(full_mask[:, None, :, :], scores, NEG_INF)
    attn = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
    o = jnp.einsum('bhqk,bhkd->bhqd', attn, v)
    x = x + self.op(self._merge(o))
    x = self._mlp(x)
    return x, own_k, own_v


# --------------------------------------------------------------------------- #
# Main module.
# --------------------------------------------------------------------------- #
class LPTransformer(nn.Module):
  """Liar's Poker sequence transformer (policy logits + value)."""
  d_model: int
  n_layers: int
  n_heads: int
  num_digits: int      # D
  max_count: int
  num_players: int     # P
  total_cards: int
  num_actions: int     # 1 (challenge) + max_bids
  max_seq: int         # L (max number of bids)
  d_ff: int = 0        # 0 -> 4*d_model
  count_freqs: int = 6
  # 'categorical' | 'relative_bias' | 'rope_digit'
  number_encoding: str = 'categorical'
  # Digit-rope ORIGIN for 'rope_digit' (ignored otherwise):
  #  * 'zero': digit n -> rope position n (from-zero / absolute; legacy behaviour).
  #  * 'max' : digit n -> rope position (num_digits - n) so the TOP digit anchors
  #    at phase 0 and lower digits count UP from the top.  Null/pad tokens keep
  #    phase 0 in BOTH origins.  'max' keeps the count-boundary / top-digit phase
  #    STABLE as D grows -> transfers to a larger game's new top digit.
  digit_rope_origin: str = 'max'
  # Action-decode HEAD digit handling:
  #  * False (default/control): the CATEGORICAL head -- a learned per-digit
  #    `number_embed` selector in the tied key (byte-identical to before).
  #  * True: the DIGIT-EQUIVARIANT head -- the action key drops `number_embed`
  #    (key base = type_ACTION + count_enc(count), shared across digits) and the
  #    digit is injected ONLY via the SAME from-max digit rope (phase num_digits-n)
  #    applied to the key on a digit-rotary slice, exactly mirroring how the null
  #    decision query (phase 0) rope-selects hand/bid-n in the trunk.  This makes
  #    the ENTIRE digit path parameter-free -> a smaller-game agent transfers to a
  #    larger one (new digits inherit the shared head function).  challenge (a=0)
  #    keeps its learned `challenge_embed` (digit 0 -> phase 0 -> identity rope).
  head_digit_equivariant: bool = False
  # PLAYED-game digit count for the FROM-MAX phase + capacity hand masking.
  #  * <= 0 (default/sentinel): equals ``num_digits`` -> byte-identical to the
  #    pre-capacity behaviour (the network's sizing == the played game).
  #  * > 0: the FIXED-SIZE ("max game") mode.  ``num_digits`` (and the derived
  #    ``num_actions`` / hand length / count normalization) are the CAPACITY
  #    dims of a larger max game, while ``game_num_digits`` = D_game is the
  #    ACTUAL played game's digit count.  It drives (a) the from-max digit rope
  #    origin (top played digit -> phase 0 in EVERY game) and (b) the hand-slot
  #    validity mask: hand digit slots for n > D_game are NON-EXISTENT in the
  #    played game and are masked OUT of attention (never valid keys), so the
  #    zero-count padding slots never pollute the (capacity-sized) hand encoding.
  game_num_digits: int = 0
  # ---- 'perdigit_tailprob' action head (config-gated; default OFF) --------- #
  # head_mode:
  #  * 'legacy' (default): the tied dot-product head above -- BYTE-IDENTICAL to
  #    the historical network (single DECISION token, categorical/equivariant
  #    action keys).  Nothing below is created or run.
  #  * 'perdigit_tailprob': the NEW head.  D per-digit decision tokens (one shared
  #    per-digit function via the from-max digit rope) + a game-size-INVARIANT
  #    outer-product bid head ``logit(d,c) = <q_digit[d], CountMLP(q(c))> +
  #    alpha*belief(d,c)`` with q(c)/belief closed-form binomial tail log-probs,
  #    and the CHALLENGE (action 0) + value read from the RETAINED global DECISION
  #    token.  REQUIRES number_encoding='rope_digit' (the digit routing is the
  #    2D digit rope); the per-digit path is fully parameter-free -> transfers
  #    across counts / num_players / num_digits.
  head_mode: str = 'legacy'
  # PLAYED-game hand length for the count/belief pools.  <=0 (default): derive
  # from ``total_cards // num_players`` (exact when the net is sized to the played
  # game).  In CAPACITY mode ``total_cards`` is the capacity total, so the played
  # hand length must be supplied so the binomial pools match the played game.
  game_hand_length: int = 0
  belief_clamp: float = 30.0   # log-prob clamp for the q(c)/belief tail features.
  count_mlp_hidden: int = 0    # CountMLP hidden width (0 -> d_model).
  dtype: Dtype = jnp.float32  # trunk compute dtype (bf16 for training).
  # Gradient-checkpoint (nn.remat) the heavy full self-attention blocks.
  #  * None (default): platform-conditional -- remat ONLY on the XLA CPU backend
  #    (where the un-rematted fused R-NaD loss overruns the compiler); on GPU the
  #    blocks run un-rematted (remat would force a wasteful forward recompute in
  #    the backward pass).
  #  * True / False: explicit override wins regardless of backend.
  # remat is value-transparent, so this NEVER changes numerics.
  remat: Optional[bool] = None

  @property
  def _game_D(self) -> int:
    """Played-game digit count (``num_digits`` when ``game_num_digits`` unset)."""
    return self.num_digits if self.game_num_digits <= 0 else self.game_num_digits

  @property
  def _new_head(self) -> bool:
    """True for the 'perdigit_tailprob' head (per-digit tokens + tail-prob head)."""
    return self.head_mode == 'perdigit_tailprob'

  @property
  def _game_hand_length(self) -> int:
    """Played-game hand length (for the count/belief binomial pools)."""
    return (self.game_hand_length if self.game_hand_length > 0
            else self.total_cards // self.num_players)

  @property
  def _count_total_pool(self) -> int:
    """POOL for q(c): total cards over ALL players = P * hand_length."""
    return self.num_players * self._game_hand_length

  @property
  def _belief_others_pool(self) -> int:
    """POOL for belief(d,c): the OTHER players' unknown cards = (P-1)*hand_len."""
    return (self.num_players - 1) * self._game_hand_length

  def _hand_valid_mask(self, b: int) -> chex.Array:
    """[b, D] bool -- hand digit slots 1..D_game valid, slots > D_game masked.

    In the (default) non-capacity mode ``_game_D == num_digits`` so every slot
    is valid and this is byte-identical to an all-True mask.  In capacity mode
    slots for non-existent digits (n > D_game) are False, keeping them out of
    attention as KEYS (their zero-count padding tokens never affect the encoding
    of the real digits or the decision read-out).
    """
    valid = jnp.arange(self.num_digits) < self._game_D  # [D]
    return jnp.broadcast_to(valid, (b, self.num_digits))

  @property
  def _relbias(self) -> bool:
    """True in 'relative_bias' mode (digit identity lives in the attn bias)."""
    return self.number_encoding == 'relative_bias'

  @property
  def _rope_digit(self) -> bool:
    """True in 'rope_digit' mode (digit identity lives in a 2nd RoPE axis).

    Like relative_bias, the per-digit ``number_embed`` is DROPPED from the token
    VALUE; digit routing is instead a second, DIGIT-index rotary on Q/K (see
    ``_apply_rope_2d``).  Unlike relative_bias it adds NO attention bias and NO
    extra select, so the attention keeps the fused softmax(QK/scale) op-structure
    (categorical-identical) -> fMHA-friendly on GPU.
    """
    return self.number_encoding == 'rope_digit'

  @property
  def _shared_value_number(self) -> bool:
    """True when the digit is DROPPED from the token value (relbias|rope_digit).

    In both modes digit identity is injected in attention (bias or 2nd RoPE
    axis), not in the token content, so the hand/bid token VALUE is shared
    across digits.  'categorical' keeps the per-digit ``number_embed`` in value.
    """
    return self._relbias or self._rope_digit

  @property
  def _need_number_embed(self) -> bool:
    """True iff the per-digit ``number_embed`` is actually used anywhere.

    It is needed when the token VALUE carries the digit (categorical mode:
    ``not _shared_value_number``) OR the action-decode head is the categorical
    one (``not head_digit_equivariant``).  Only when BOTH the value AND the head
    are digit-shared (rope_digit/relbias + equivariant head) is it fully unused,
    and then it is not created -> the param tree is num_digits-independent.

    The 'perdigit_tailprob' head never uses ``number_embed`` (its trunk is
    rope_digit -> shared value, and its outer-product head has no per-digit
    param), so it is dropped there too -> that head's param tree is fully
    num_digits-independent.
    """
    if self._new_head:
      return False
    return (not self._shared_value_number) or (not self.head_digit_equivariant)

  def _digit_rope_pos(self, ids: chex.Array) -> chex.Array:
    """Raw digit ids (0=null/pad, 1..D) -> 'rope_digit' rotary POSITIONS.

    * ``digit_rope_origin='max'`` (default): real digit n>=1 -> ``num_digits-n``
      (TOP digit -> phase 0, phase-stable as D grows); null/pad (0) stays 0.
    * ``digit_rope_origin='zero'``: identity (n -> n), legacy from-zero absolute.

    Elementwise, so applying it per-token (KV-cache paths) is identical to
    applying it to a whole digit-id vector (full-forward) -> full-vs-cache
    equivalence is preserved.
    """
    ids = jnp.asarray(ids)
    if self.digit_rope_origin == 'max':
      # Keep null/pad (0) at 0; map real digits 1..D_game to D_game-1..0.  Uses
      # the PLAYED-game D (``_game_D``), not the capacity num_digits, so the top
      # played digit anchors at phase 0 in EVERY game (6x6: digit6->0; 6x8:
      # digit8->0).  Non-existent padded digits (n > D_game) get a negative
      # phase, but they are always masked out (illegal actions / masked hand
      # slots), so this never participates in the output.
      return jnp.where(ids == 0, 0, self._game_D - ids).astype(ids.dtype)
    assert self.digit_rope_origin == 'zero', (
        f"digit_rope_origin must be 'max' or 'zero'; got {self.digit_rope_origin!r}")
    return ids

  def _head_rope_dims(self) -> Tuple[int, int]:
    """(d_seq, d_digit) partition of ``d_model`` for the digit-equivariant HEAD.

    The head is a plain dot-product over ``d_model`` (no attention heads), so the
    digit rotary lives on a ``d_model``-level slice.  Half/half with each block
    even (RoPE rotates dim PAIRS): ``d_digit = 2*(d_model//4)``.
    """
    dm = self.d_model
    d_digit = 2 * (dm // 4)
    return dm - d_digit, d_digit

  def _apply_head_digit_rope(self, x: chex.Array,
                             digit_pos: chex.Array) -> chex.Array:
    """Apply the digit rotary to the digit-slice of head vectors ``x`` [A, d].

    Rotates only ``x[..., d_seq:]`` by ``digit_pos`` (the seq-slice is left as-is
    -- the head has no sequence position); ``digit_pos`` are ALREADY from-max
    positions.  digit_pos==0 (challenge / null) -> identity rotation.
    """
    d_seq, _ = self._head_rope_dims()
    x_seq = x[..., :d_seq]
    x_dig = x[..., d_seq:][None, None, :, :]        # [1, 1, A, d_digit]
    x_dig = _apply_rope(x_dig, digit_pos[None, :])[0, 0]  # [A, d_digit]
    return jnp.concatenate([x_seq, x_dig], axis=-1)

  def setup(self):
    d = self.d_model
    assert (d // self.n_heads) % 2 == 0, 'd_head must be even for RoPE'
    d_ff = self.d_ff or 4 * d
    # Digit (`number`) axis handling.
    #  * 'categorical' (default/control): the learned per-digit lookup
    #    `number_embed` is folded into the hand/bid token CONTENT (value).
    #  * 'relative_bias': digit identity is REMOVED from token content and
    #    injected instead as a learned per-head RELATIVE-DIGIT bias on the
    #    attention logits (a sharp same-digit peak => EXACT bid-n<->hand-n
    #    routing) while the token VALUE is shared across digits.
    # `number_embed` is the ONLY per-digit learned parameter.  It feeds the token
    # CONTENT in 'categorical' mode and/or the CATEGORICAL action-decode head.
    # It is created ONLY when actually used (`_need_number_embed`): when BOTH the
    # token value is digit-shared (relbias|rope_digit) AND the head is
    # digit-equivariant, the digit path is fully parameter-free and `number_embed`
    # is dropped entirely -> the param tree is num_digits-INDEPENDENT (transfer).
    assert self.number_encoding in (
        'categorical', 'relative_bias', 'rope_digit'), (
        f"Unknown number_encoding {self.number_encoding!r}; expected "
        "'categorical', 'relative_bias' or 'rope_digit'")
    if self._rope_digit:
      # 'rope_digit' partitions d_head into a seq rotary + a digit rotary, each
      # an EVEN sub-block; d_head>=4 guarantees d_digit=2*(d_head//4)>=2.
      assert (d // self.n_heads) >= 4, (
          "rope_digit needs d_head>=4 (d_head split into two even rotary "
          f"sub-blocks); got d_head={d // self.n_heads}")
    if self._need_number_embed:
      self.number_embed = nn.Embed(self.num_digits + 1, d)  # 0=pad, 1..D
    if self._relbias:
      # Learned per-head bias over the SIGNED digit offset (q_digit - k_digit)
      # in [-(D-1), D-1] (index = offset + (D-1)).  Zero-initialised so that at
      # init the attention is unbiased (identity start).
      self.digit_bias = self.param(
          'digit_bias', nn.initializers.zeros,
          (self.n_heads, 2 * self.num_digits - 1))
    self.type_embed = nn.Embed(3, d)
    self.player_rel_embed = nn.Embed(self.num_players, d)
    # The legacy smooth count encoder is REPLACED by CountMLP(q(c)) in the
    # 'perdigit_tailprob' head, so it is not created there (keeps that head's
    # param tree minimal; every count is embedded via _count_emb -> count_mlp).
    if not self._new_head:
      self.count_enc = CountEncoder(
          d_model=d, total_cards=self.total_cards, max_count=self.max_count,
          num_freqs=self.count_freqs)
    self.globals_dense = nn.Dense(d)
    # Gradient-checkpoint the (heavy) full self-attention pass.  This is
    # semantically transparent -- it only trades recompute for a much smaller
    # backward graph -- but is REQUIRED in practice: without it the fused R-NaD
    # loss (four transformer forwards + v-trace + nerd under one value_and_grad)
    # overruns the XLA CPU compiler at training batch sizes.  The incremental
    # `step` path (KV-cache, stages 2/3) is intentionally NOT rematted.
    # None -> remat only on the XLA CPU backend (where the un-rematted fused loss
    # overruns the compiler); on GPU the recompute is wasteful.  Explicit wins.
    use_remat = (jax.default_backend() == 'cpu'
                 if self.remat is None else self.remat)
    Block = (nn.remat(TransformerBlock, methods=['full'])
             if use_remat else TransformerBlock)
    self.blocks = [Block(d, self.n_heads, d_ff, dtype=self.dtype)
                   for _ in range(self.n_layers)]
    self.query_head = nn.Dense(d)
    self.value_h = nn.Dense(d)
    self.value_out = nn.Dense(1)
    self.challenge_embed = self.param(
        'challenge_embed', nn.initializers.normal(0.02), (d,))
    # Static per-action (count, number) tables for the tied decode keys.
    numbers = np.zeros((self.num_actions,), dtype=np.int32)
    counts = np.zeros((self.num_actions,), dtype=np.int32)
    for a in range(1, self.num_actions):
      bid_id = a - 1
      numbers[a] = bid_id % self.num_digits + 1
      counts[a] = bid_id // self.num_digits + 1
    self._action_numbers = jnp.asarray(numbers)
    self._action_counts = jnp.asarray(counts)
    # ---- 'perdigit_tailprob' head params + closed-form tail tables --------- #
    if self._new_head:
      assert self._rope_digit, (
          "head_mode='perdigit_tailprob' requires number_encoding='rope_digit' "
          "(the per-digit decision tokens are routed by the 2D digit rope); got "
          f"number_encoding={self.number_encoding!r}")
      assert self.head_mode == 'perdigit_tailprob', (
          f"unknown head_mode {self.head_mode!r}")
      # Shared game-size-INVARIANT count encoder (replaces count_enc in this head)
      self.count_mlp = CountMLP(d_model=d, hidden=self.count_mlp_hidden)
      # A distinct learned TYPE embedding for the per-digit decision tokens (kept
      # OUT of ``type_embed`` so the legacy param tree is unchanged).
      self.digit_dec_type_embed = self.param(
          'digit_dec_type_embed', nn.initializers.normal(0.02), (d,))
      # Learned scalar mixing the closed-form belief log-prob into the bid logit.
      # Init 1.0: belief(d,c) IS a log-probability, so alpha=1 gives it its
      # natural logit scale as an informative prior from the first step.
      self.belief_alpha = self.param(
          'belief_alpha', nn.initializers.ones, (), jnp.float32)
      # q(c) = log P(total_count >= c), total_count ~ Binomial(POOL, 1/D_game),
      # POOL = P * hand_length.  Table over c = 0 .. POOL+1 (a jnp constant).
      self._q_table = jnp.asarray(_log_survival_table(
          self._count_total_pool, self._game_D, self.belief_clamp))
      # belief survival: log P(others_count_d >= t), t = 0 .. others_pool+1.
      self._belief_log_surv = jnp.asarray(_log_survival_table(
          self._belief_others_pool, self._game_D, self.belief_clamp))

  # ----- Relative-digit attention bias (relative_bias mode only). ---------- #
  def _digit_bias(self, q_digit: chex.Array, k_digit: chex.Array) -> chex.Array:
    """Learned per-head relative-digit attention bias -> [B, H, Tq, Tk] (fp32).

    Args:
      q_digit: [B, Tq] int per-token digit ids of the QUERY tokens.
      k_digit: [B, Tk] int per-token digit ids of the KEY tokens.
        A digit id is the digit a token is ABOUT: hand token for digit d -> d
        (1..D); bid token about number n -> n; DECISION / null tokens -> 0.
    The bias is a learned table over the signed offset (q_digit - k_digit) in
    [-(D-1), D-1] per head; it is forced to 0 whenever EITHER token is null
    (digit 0), so only genuine digit<->digit pairs get a routing bias.  This lets
    attention learn a sharp same-digit (offset 0) peak -> exact bid-n<->hand-n
    matching, while the token value stays digit-shared.
    """
    d = self.num_digits
    offset = q_digit[:, :, None] - k_digit[:, None, :]          # [B, Tq, Tk]
    null = (q_digit[:, :, None] == 0) | (k_digit[:, None, :] == 0)
    idx = jnp.clip(offset, -(d - 1), d - 1) + (d - 1)           # [B,Tq,Tk] 0..2D-2
    bias = self.digit_bias[:, idx]                             # [H, B, Tq, Tk]
    bias = jnp.transpose(bias, (1, 0, 2, 3))                    # [B, H, Tq, Tk]
    bias = jnp.where(null[:, None, :, :], 0.0, bias)
    return bias.astype(jnp.float32)

  def _hand_digit_ids(self, b: int) -> chex.Array:
    """[b, D] int -- hand token digit ids (digit d -> d, 1..D)."""
    return jnp.broadcast_to(jnp.arange(1, self.num_digits + 1), (b, self.num_digits))

  def _combined_bias(self, digit_ids: chex.Array,
                     mask: chex.Array) -> chex.Array:
    """Fuse the relative-digit bias and the structural mask into ONE additive
    bias [B, H, N, N] fp32 (0/NEG_INF mask folded in).

    Used by the full-forward LOSS paths so the self-attention is the single
    additive-bias form ``softmax(QK/scale + bias)`` -- the pattern XLA's GPU
    fused-attention (cuDNN fMHA) rewrite matches.  Adding an external per-head
    bias AND a separate ``where``-select (the previous relative_bias path) broke
    that match on GPU, forcing an unfused, memory-bound attention (~4x slower)
    even though FLOPs/bytes barely change.  Bias + mask are both
    LAYER-INDEPENDENT, so this is built once per forward and reused by all
    layers.  Numerically identical (up to fp rounding) to bias+``where``:
    allowed logits are unchanged; masked logits underflow to 0 after softmax
    either way.
    """
    digit_bias = self._digit_bias(digit_ids, digit_ids)   # [B, H, N, N] fp32
    mask_add = jnp.where(mask[:, None, :, :], 0.0, NEG_INF).astype(jnp.float32)
    return digit_bias + mask_add                          # [B, H, N, N]

  # ----- Count encoding dispatch (legacy CountEncoder vs new CountMLP). ----- #
  def _count_emb(self, c: chex.Array) -> chex.Array:
    """Embed an integer count ``c`` -> [..., d_model].

    * legacy: the smooth ordinal ``count_enc(c)`` (byte-identical to before).
    * 'perdigit_tailprob': ``CountMLP(q(c))`` with ``q(c) = log P(total >= c)`` --
      a game-size-INVARIANT tail feature (see ``_q_of_count``).  Used EVERYWHERE
      a count is embedded in this head (hand held-counts, bid count, my-count).
    """
    if not self._new_head:
      return self.count_enc(c)
    return self.count_mlp(self._q_of_count(c))

  def _q_of_count(self, c: chex.Array) -> chex.Array:
    """Gather the tail feature ``q(c) = log P(total_count >= c)`` -> same shape."""
    idx = jnp.clip(jnp.asarray(c).astype(jnp.int32), 0, self._q_table.shape[0] - 1)
    return self._q_table[idx]

  # ----- Token embedding helpers (reused by full + incremental paths). ----- #
  def _hand_tokens(self, hand_hist: chex.Array) -> chex.Array:
    """[B, D] counts -> [B, D, d_model] hand tokens.

    In 'relative_bias' / 'rope_digit' modes the per-digit ``number_embed`` is
    DROPPED from the content (digit identity moves to the attention bias /
    the 2nd RoPE axis); the value is then shared across digits.  In
    'categorical' mode it is added as before.
    """
    b, dcount = hand_hist.shape
    cnt_emb = self._count_emb(hand_hist)              # [B, D, d]
    type_emb = self.type_embed(jnp.asarray(TYPE_HAND))  # [d]
    tok = type_emb[None, None, :] + cnt_emb
    if not self._shared_value_number:
      numbers = jnp.arange(1, dcount + 1)  # [D]
      num_emb = self.number_embed(numbers)[None, :, :]  # [1, D, d]
      tok = tok + num_emb
    return tok.astype(self.dtype)

  def _bid_tokens(self, bids: chex.Array, observer: chex.Array,
                  hand_hist: chex.Array) -> chex.Array:
    """[B, L, 3] bids -> [B, L, d_model] bid tokens (no RoPE; applied in attn)."""
    c = bids[..., 0]
    n = bids[..., 1]
    p = bids[..., 2]
    n_clamped = jnp.clip(n, 1, self.num_digits)
    my_count = jnp.take_along_axis(hand_hist, n_clamped - 1, axis=1)  # [B, L]
    rel = jnp.mod(p - observer[:, None], self.num_players)
    tok = (self.type_embed(jnp.asarray(TYPE_BID))[None, None, :]
           + self._count_emb(c)
           + self.player_rel_embed(rel)
           + self._count_emb(my_count))
    if not self._shared_value_number:
      # 'categorical': digit identity in the content; 'relative_bias'/'rope_digit'
      # drop it (it lives in the attention bias / 2nd RoPE axis, keyed by n).
      tok = tok + self.number_embed(n)
    return tok.astype(self.dtype)

  def _decision_token(self, globals_: chex.Array) -> chex.Array:
    """[B, G] -> [B, d_model] DECISION token."""
    tok = self.type_embed(jnp.asarray(TYPE_ACTION))[None, :] \
        + self.globals_dense(globals_)
    return tok.astype(self.dtype)

  def _action_keys(self) -> chex.Array:
    """Tied per-action key table [num_actions, d_model].

    Two head variants (config-gated by ``head_digit_equivariant``):

    * CATEGORICAL (default): key = type_ACTION + ``number_embed(n)`` + count_enc(c).
      The learned per-digit ``number_embed`` selects the digit (byte-identical to
      the historical head).

    * DIGIT-EQUIVARIANT: key base = type_ACTION + count_enc(c) (NO per-digit
      param, shared across digits); the digit is injected by the SAME from-max
      digit rope (phase ``num_digits-n``) applied to the key's digit-rotary slice.
      The decision-token query is unrotated (phase 0), exactly mirroring the null
      decision query rope-selecting hand/bid-n in the trunk.  Fully parameter-free
      in the digit -> new digits inherit the shared head function (transfer).

    Action 0 (challenge) keeps its learned ``challenge_embed`` in both variants
    (its digit is 0 -> phase-0 identity rope leaves ``challenge_embed`` exact).
    """
    if not self.head_digit_equivariant:
      keys = (self.type_embed(jnp.asarray(TYPE_ACTION))[None, :]
              + self.number_embed(self._action_numbers)
              + self.count_enc(self._action_counts))  # [num_actions, d]
      keys = keys.at[0].set(self.challenge_embed)
      return keys
    # Digit-equivariant head: base key has NO per-digit parameter.
    base = (self.type_embed(jnp.asarray(TYPE_ACTION))[None, :]
            + self.count_enc(self._action_counts))    # [num_actions, d]
    base = base.at[0].set(self.challenge_embed)        # challenge: digit 0.
    digit_pos = self._digit_rope_pos(self._action_numbers)  # from-max phases.
    return self._apply_head_digit_rope(base, digit_pos)

  def _heads(self, h: chex.Array, legal_mask: chex.Array):
    """DECISION-position hidden [B, d] -> (logits [B,A], value [B,1]).

    The trunk hidden ``h`` may be bf16; the head modules (fp32 params) run the
    policy/value read-out in fp32, and we cast the outputs to fp32 explicitly so
    the downstream masked softmax / R-NaD loss / v-trace never touch bf16.
    """
    keys = self._action_keys()  # [A, d]  (fp32, from fp32 embeds)
    q = self.query_head(h)       # [B, d]  (fp32: bf16 in, fp32 params)
    scale = jnp.sqrt(jnp.asarray(self.d_model, jnp.float32))
    logits = jnp.matmul(q, keys.T) / scale  # [B, A]
    logits = jnp.where(legal_mask, logits, NEG_INF)
    value = self.value_out(nn.relu(self.value_h(h)))  # [B, 1]
    return logits.astype(jnp.float32), value.astype(jnp.float32)

  # ------------------- 'perdigit_tailprob' head compute -------------------- #
  def _belief_dc(self, hand_hist: chex.Array) -> chex.Array:
    """Closed-form per-(digit,count) belief log-prob -> [..., D, C] (C=max_count).

    ``belief(d, c) = log P(others_count_d >= c - h_d | hand)`` with
    ``others_count_d ~ Binomial((P-1)*hand_length, 1/D_game)`` and ``h_d`` the
    observer's held count of digit ``d`` (``hand_hist[..., d]``).  ``c - h_d <= 0``
    -> threshold 0 -> ``log 1 = 0`` (certain).  A pure gather into the precomputed
    ``_belief_log_surv`` table -> game-size invariant, differentiable, no runtime
    binomial.  ``hand_hist`` may carry arbitrary leading dims.
    """
    counts = jnp.arange(1, self.max_count + 1)                # [C]
    thr = counts[..., None, :] - hand_hist[..., :, None]      # [..., D, C]
    idx = jnp.clip(thr, 0, self._belief_log_surv.shape[0] - 1)
    return self._belief_log_surv[idx]                         # [..., D, C]

  def _new_head_apply(self, h_global: chex.Array, h_digit: chex.Array,
                      hand_hist: chex.Array, legal_mask: chex.Array):
    """Outer-product bid head + global-token challenge/value.

    Args (arbitrary shared leading dims ``...``):
      h_global: [..., d]        -- global DECISION token hidden.
      h_digit:  [..., D, d]     -- the D per-digit decision token hiddens.
      hand_hist:[..., D]        -- observer held-count per digit (for belief).
      legal_mask:[..., A]       -- legal-action mask (A = 1 + max_count*D).
    Returns (logits [..., A] fp32, value [..., 1] fp32).

      * CHALLENGE (action 0): ``<query_head(h_global), challenge_embed>/sqrt(d)``.
      * BID (digit d, count c): ``<query_head(h_digit[d]), CountMLP(q(c))>/sqrt(d)
        + alpha*belief(d, c)`` -> flattened as ``bid_id = (c-1)*D + (d-1)`` (count
        outer, digit inner), action = bid_id + 1.
      * value: ``value_out(relu(value_h(h_global)))``.
    """
    scale = jnp.sqrt(jnp.asarray(self.d_model, jnp.float32))
    # Challenge + value from the retained global decision token.
    q_glob = self.query_head(h_global)                       # [..., d]
    challenge = jnp.einsum('...m,m->...', q_glob,
                           self.challenge_embed) / scale       # [...]
    value = self.value_out(nn.relu(self.value_h(h_global)))   # [..., 1]
    # Per-digit outer-product bid logits.
    q_digit = self.query_head(h_digit)                       # [..., D, d]
    counts = jnp.arange(1, self.max_count + 1)               # [C]
    k_count = self.count_mlp(self._q_of_count(counts))       # [C, d]
    raw = jnp.einsum('...dm,cm->...dc', q_digit,
                     k_count.astype(q_digit.dtype)) / scale    # [..., D, C]
    belief = self._belief_dc(hand_hist)                      # [..., D, C]
    bid_dc = raw.astype(jnp.float32) + self.belief_alpha * belief  # [..., D, C]
    # Flatten count-outer, digit-inner -> bid_id = (c-1)*D + (d-1).
    lead = bid_dc.shape[:-2]
    bid = jnp.swapaxes(bid_dc, -1, -2).reshape(lead + (self.num_actions - 1,))
    logits = jnp.concatenate([challenge[..., None], bid], axis=-1)  # [..., A]
    logits = jnp.where(legal_mask, logits, NEG_INF)
    return logits.astype(jnp.float32), value.astype(jnp.float32)

  def _digit_dec_content(self, globals_dec: chex.Array) -> chex.Array:
    """Per-digit decision-token CONTENT (shared across digits; digit id is
    injected only via the digit rope): ``digit_dec_type + globals_dense(g)``.

    ``globals_dec`` has shape ``[..., G]``; returns ``[..., D, d]`` (broadcast the
    same content over the D per-digit tokens).
    """
    base = (self.digit_dec_type_embed
            + self.globals_dense(globals_dec))               # [..., d]
    d = jnp.broadcast_to(base[..., None, :],
                         base.shape[:-1] + (self.num_digits, self.d_model))
    return d.astype(self.dtype)

  def _global_dec_content(self, globals_dec: chex.Array) -> chex.Array:
    """Global decision-token content: ``type_ACTION + globals_dense(g)`` (== the
    legacy ``_decision_token`` content, broadcast over any leading dims)."""
    tok = (self.type_embed(jnp.asarray(TYPE_ACTION))
           + self.globals_dense(globals_dec))
    return tok.astype(self.dtype)

  # ------------------------- Full parallel forward ------------------------- #
  def _build_mask(self, bid_valid: chex.Array) -> chex.Array:
    """Attention mask [B, N, N] for sequence [hands(D), bids(L), decision(1)]."""
    d, ell = self.num_digits, self.max_seq
    n = d + ell + 1
    idx = np.arange(n)
    q = idx[:, None]
    k = idx[None, :]
    hand_q = q < d
    bid_q = (q >= d) & (q < d + ell)
    dec_q = q == d + ell
    hand_k = k < d
    bid_k = (k >= d) & (k < d + ell)
    dec_k = k == d + ell
    structural = (
        (hand_q & hand_k)
        | (bid_q & (hand_k | (bid_k & (k <= q))))
        | (dec_q & (hand_k | bid_k | dec_k))
    )  # [N, N]
    structural = jnp.asarray(structural)[None, :, :]  # [1, N, N]
    # Key validity: decision always valid; hands gated by the capacity hand-slot
    # mask (all-True in non-capacity mode); bids gated by bid_valid.
    b = bid_valid.shape[0]
    key_valid = jnp.concatenate([
        self._hand_valid_mask(b),
        bid_valid,
        jnp.ones((b, 1), dtype=bool),
    ], axis=1)  # [B, N]
    return structural & key_valid[:, None, :]

  def _full_forward_newhead(self, inputs: LPInputs):
    """'perdigit_tailprob' single-decision forward -> (logits [B,A], value [B,1]).

    Sequence layout ``[hands(D)][bids(L)][global_dec(1)][digit_dec(D)]``.  The
    decision tokens are ALL query-only: each attends to hands + its valid bid
    prefix + ITSELF (never to bids beyond, nor to any OTHER decision token -- so
    the D per-digit tokens are INDEPENDENT channels).  Reuses ``_build_pd_mask``
    with ``1 + D`` decision slots all sharing prefix ``num_valid``.
    """
    b = inputs.hand_hist.shape[0]
    d, ell = self.num_digits, self.max_seq
    hand = self._hand_tokens(inputs.hand_hist)                  # [B,D,d]
    bid = self._bid_tokens(inputs.bids, inputs.observer, inputs.hand_hist)
    gdec = self._global_dec_content(inputs.globals_)[:, None, :]  # [B,1,d]
    ddec = self._digit_dec_content(inputs.globals_)             # [B,D,d]
    x = jnp.concatenate([hand, bid, gdec, ddec], axis=1)        # [B,N,d]
    num_valid = jnp.sum(inputs.bid_valid.astype(jnp.int32), axis=1)  # [B]
    # RoPE seq positions: hands 0, bid j -> j+1, all dec tokens -> num_valid+1.
    hand_pos = jnp.zeros((b, d), dtype=jnp.int32)
    bid_pos = jnp.broadcast_to(jnp.arange(1, ell + 1), (b, ell))
    dec_pos = jnp.broadcast_to((num_valid + 1)[:, None], (b, 1 + d))
    pos = jnp.concatenate([hand_pos, bid_pos, dec_pos], axis=1)  # [B,N]
    # Decision prefix lengths for the mask: all 1+D slots see all valid bids.
    dec_prefix = jnp.broadcast_to(num_valid[:, None], (b, 1 + d))
    mask = self._build_pd_mask(inputs.bid_valid, dec_prefix)    # [B,N,N]
    # Digit rope ids: hands 1..D, bids number n, global dec 0, digit dec 1..D.
    dec_raw = jnp.concatenate([
        jnp.zeros((b, 1), jnp.int32),
        jnp.broadcast_to(jnp.arange(1, d + 1), (b, d)),
    ], axis=1)                                                  # [B, 1+D]
    digit_raw = jnp.concatenate(
        [self._hand_digit_ids(b), inputs.bids[..., 1], dec_raw], axis=1)
    digit_pos = self._digit_rope_pos(digit_raw)                 # [B,N]
    for blk in self.blocks:
      x, _, _ = blk.full(x, mask, pos, digit_pos=digit_pos)
    h = x[:, d + ell:, :]                                       # [B, 1+D, d]
    h_global = h[:, 0, :]                                       # [B, d]
    h_digit = h[:, 1:, :]                                       # [B, D, d]
    return self._new_head_apply(
        h_global, h_digit, inputs.hand_hist, inputs.legal_mask)

  def full_forward(self, inputs: LPInputs):
    """Parallel (decode=False) forward. Returns (logits [B,A], value [B,1])."""
    if self._new_head:
      return self._full_forward_newhead(inputs)
    hand = self._hand_tokens(inputs.hand_hist)                         # [B,D,d]
    bid = self._bid_tokens(inputs.bids, inputs.observer, inputs.hand_hist)
    dec = self._decision_token(inputs.globals_)[:, None, :]            # [B,1,d]
    x = jnp.concatenate([hand, bid, dec], axis=1)                      # [B,N,d]
    b = x.shape[0]
    d, ell = self.num_digits, self.max_seq
    num_valid = jnp.sum(inputs.bid_valid.astype(jnp.int32), axis=1)    # [B]
    hand_pos = jnp.zeros((b, d), dtype=jnp.int32)
    bid_pos = jnp.broadcast_to(jnp.arange(1, ell + 1), (b, ell))
    dec_pos = (num_valid + 1)[:, None]
    pos = jnp.concatenate([hand_pos, bid_pos, dec_pos], axis=1)        # [B,N]
    mask = self._build_mask(inputs.bid_valid)
    # Relative-digit bias: per-token digit ids for [hands, bids, decision].
    # In relative_bias mode the bias + mask are fused into ONE additive term
    # (fMHA-friendly on GPU); categorical keeps the byte-identical mask+where.
    if self._relbias:
      digit_ids = jnp.concatenate([
          self._hand_digit_ids(b),                        # hands: digit d -> d
          inputs.bids[..., 1],                            # bids: number n
          jnp.zeros((b, 1), dtype=jnp.int32),             # decision: null
      ], axis=1)                                          # [B, N]
      cbias = self._combined_bias(digit_ids, mask)
      for blk in self.blocks:
        x, _, _ = blk.full(x, mask, pos, combined_bias=cbias)
    elif self._rope_digit:
      # 2D-RoPE digit routing: per-token digit ids drive a 2nd rotary on Q/K;
      # masking is the plain categorical where (no bias-add / no extra select).
      digit_ids = jnp.concatenate([
          self._hand_digit_ids(b),                        # hands: digit d -> d
          inputs.bids[..., 1],                            # bids: number n
          jnp.zeros((b, 1), dtype=jnp.int32),             # decision: null (0)
      ], axis=1)                                          # [B, N]
      digit_ids = self._digit_rope_pos(digit_ids)         # from-max positions.
      for blk in self.blocks:
        x, _, _ = blk.full(x, mask, pos, digit_pos=digit_ids)
    else:
      for blk in self.blocks:
        x, _, _ = blk.full(x, mask, pos)
    h = x[:, -1, :]  # DECISION position.
    return self._heads(h, inputs.legal_mask)

  def __call__(self, inputs: LPInputs, decode: bool = False):
    # `decode` is accepted for API symmetry; the incremental path is exposed via
    # the explicit `prefill`/`append_bid`/`decision_logits` methods (a functional
    # KV cache is easier to test than a mutable-collection cache).
    return self.full_forward(inputs)

  # ------------------ Causal multi-readout (Stage A, LM-style) ------------- #
  # ONE causal forward over [hands][b_1..b_L] that emits a policy/value readout
  # for EVERY decision turn, instead of one full_forward per turn.  Provably
  # identical to per-turn `full_forward` because the transformer body is causal:
  # bid_j's representation depends only on hands + bids 0..j (never later bids),
  # so appending the whole trajectory does not perturb any earlier prefix.
  #
  # Layout (single sequence, N = D + 2L tokens):
  #   * hands     : indices 0 .. D-1
  #   * bids      : indices D .. D+L-1        (bid j, 0-based -> slot D+j)
  #   * decisions : indices D+L .. D+2L-1     (dec_t, 0-based -> slot D+L+t)
  # `dec_t` is the readout for the info-set that has seen the first `t` (valid)
  # bids -- i.e. the decision that PRODUCED bid_{t+1} (the LM off-by-one: it sees
  # bids 0..t-1, NOT its own bid).  Decision slots are QUERY-ONLY keys (each
  # attends to hands + earlier valid bids + itself, never to bids >= t nor to
  # OTHER decision slots), so parking them all at the tail is equivalent to
  # interleaving them: masked attention is permutation-equivariant once RoPE
  # positions + the mask are supplied explicitly.
  def _build_multi_readout_mask(self, bid_valid: chex.Array) -> chex.Array:
    """Attention mask [B, N, N] for [hands(D), bids(L), decisions(L)].

    mask[b, q, k] is True iff query q may attend to key k.  Structural rules:
      * hand_q -> hand_k                          (bidirectional among hands)
      * bid_j  -> hands, bids 0..j                (causal; NOT decisions)
      * dec_t  -> hands, bids 0..t-1, ITSELF      (NOT bids >= t, NOT other decs)
    Bid keys are additionally gated by `bid_valid`.  The `dec_t -> itself` clause
    mirrors the single-token path (`_build_mask`: `dec_q & ... | dec_k`), where
    the DECISION token's own K/V participates in its softmax.
    """
    d, ell = self.num_digits, self.max_seq
    n = d + 2 * ell
    idx = np.arange(n)
    q = idx[:, None]
    k = idx[None, :]
    hand_q = q < d
    bid_q = (q >= d) & (q < d + ell)
    dec_q = q >= d + ell
    hand_k = k < d
    bid_k = (k >= d) & (k < d + ell)
    dec_k = k >= d + ell
    j_of_q = q - d                # bid 0-based index (valid only for bid_q rows)
    j_of_k = k - d                # bid 0-based index (valid only for bid_k cols)
    t_of_q = q - (d + ell)        # dec slot index    (valid only for dec_q rows)
    t_of_k = k - (d + ell)        # dec slot index    (valid only for dec_k cols)
    structural = (
        (hand_q & hand_k)
        | (bid_q & (hand_k | (bid_k & (j_of_k <= j_of_q))))
        | (dec_q & (hand_k
                    | (bid_k & (j_of_k < t_of_q))
                    | (dec_k & (t_of_k == t_of_q))))
    )  # [N, N]
    structural = jnp.asarray(structural)[None, :, :]  # [1, N, N]
    b = bid_valid.shape[0]
    key_valid = jnp.concatenate([
        self._hand_valid_mask(b),          # hands: capacity slot mask.
        bid_valid,                         # bids: gated by validity.
        jnp.ones((b, ell), dtype=bool),    # decisions: only ever self-keys.
    ], axis=1)  # [B, N]
    return structural & key_valid[:, None, :]

  def causal_readout(self, hand_hist: chex.Array, bids: chex.Array,
                     bid_valid: chex.Array, observer: chex.Array,
                     per_position_globals: chex.Array,
                     legal_mask_per_pos: chex.Array):
    """ONE causal forward -> policy logits [B, L, A] + value [B, L] per turn.

    Output at slot ``t`` (0-based) is the decision for the info-set that has seen
    the first ``t`` valid bids, with globals ``per_position_globals[:, t]`` and
    legal mask ``legal_mask_per_pos[:, t]``.  It equals, bit-for-bit up to fp
    rounding, the single-token full forward::

        full_forward(LPInputs(hand_hist, bids,
                              bid_valid = bid_valid & (arange(L) < t),
                              observer, per_position_globals[:, t],
                              legal_mask_per_pos[:, t]))[:, ...]

    Args:
      hand_hist: [B, D] int.
      bids:      [B, L, 3] int -- ordered actual bids (0-padded).
      bid_valid: [B, L] bool.
      observer:  [B] int.
      per_position_globals: [B, L, G] float -- globals AT EACH turn t (the key
        new input: the single-token path folds one globals vector into the lone
        DECISION token; here each readout slot folds in ITS OWN turn globals).
      legal_mask_per_pos: [B, L, A] bool -- legal mask at each turn t.
    Returns:
      (logits [B, L, A] float32, value [B, L] float32).
    """
    assert not self._new_head, (
        "causal_readout (Stage A, bid-indexed) is legacy-only; the "
        "'perdigit_tailprob' head uses causal_readout_pd on the training path")
    b = hand_hist.shape[0]
    d, ell = self.num_digits, self.max_seq
    # --- Tokens (SAME embeddings as full_forward) --------------------------- #
    hand = self._hand_tokens(hand_hist)                        # [B, D, d]
    bid = self._bid_tokens(bids, observer, hand_hist)          # [B, L, d]
    # Per-position DECISION tokens: TYPE_ACTION + globals_dense(globals_t).  This
    # is exactly `_decision_token` broadcast over the L turn axis, so each slot
    # injects its OWN turn's globals (matching how the single DECISION token
    # folds in globals_ in `_decision_token`).
    dec = (self.type_embed(jnp.asarray(TYPE_ACTION))[None, None, :]
           + self.globals_dense(per_position_globals))         # [B, L, d]
    dec = dec.astype(self.dtype)  # match hand/bid compute dtype (bf16).
    x = jnp.concatenate([hand, bid, dec], axis=1)              # [B, N, d]
    # --- RoPE positions ----------------------------------------------------- #
    # hands=0; bid j (0-based) -> j+1 (== full_forward bid_pos); dec_t -> one
    # past its visible bids: (# valid bids strictly before t) + 1, matching
    # full_forward's decision pos (num_valid + 1) for the corresponding prefix.
    hand_pos = jnp.zeros((b, d), dtype=jnp.int32)
    bid_pos = jnp.broadcast_to(jnp.arange(1, ell + 1), (b, ell))
    valid_i = bid_valid.astype(jnp.int32)
    excl_cumsum = jnp.cumsum(valid_i, axis=1) - valid_i        # valid bids < t
    dec_pos = excl_cumsum + 1                                  # [B, L]
    pos = jnp.concatenate([hand_pos, bid_pos, dec_pos], axis=1)  # [B, N]
    # --- Causal body -------------------------------------------------------- #
    mask = self._build_multi_readout_mask(bid_valid)           # [B, N, N]
    # Relative-digit bias over [hands, bids, decisions(L)] (decisions = null).
    if self._relbias:
      digit_ids = jnp.concatenate([
          self._hand_digit_ids(b),
          bids[..., 1],
          jnp.zeros((b, ell), dtype=jnp.int32),
      ], axis=1)                                               # [B, N]
      cbias = self._combined_bias(digit_ids, mask)
      for blk in self.blocks:
        x, _, _ = blk.full(x, mask, pos, combined_bias=cbias)
    elif self._rope_digit:
      digit_ids = jnp.concatenate([
          self._hand_digit_ids(b),
          bids[..., 1],
          jnp.zeros((b, ell), dtype=jnp.int32),                # decisions: null
      ], axis=1)                                               # [B, N]
      digit_ids = self._digit_rope_pos(digit_ids)              # from-max positions.
      for blk in self.blocks:
        x, _, _ = blk.full(x, mask, pos, digit_pos=digit_ids)
    else:
      for blk in self.blocks:
        x, _, _ = blk.full(x, mask, pos)
    h = x[:, d + ell:, :]                                      # [B, L, d]
    # --- Tied factored head at EVERY readout position ----------------------- #
    # Identical math to `_heads`, vectorised over the L turn axis.
    keys = self._action_keys()                                 # [A, d]
    qh = self.query_head(h)                                    # [B, L, d]
    scale = jnp.sqrt(jnp.asarray(self.d_model, jnp.float32))
    logits = jnp.matmul(qh, keys.T) / scale                    # [B, L, A]
    logits = jnp.where(legal_mask_per_pos, logits, NEG_INF)
    value = self.value_out(nn.relu(self.value_h(h)))           # [B, L, 1]
    # fp32 out (consistent with LPTNetwork).
    logits = logits.astype(jnp.float32)
    value = value[..., 0].astype(jnp.float32)                  # [B, L]
    return logits, value

  # ------------- Per-decision causal multi-readout (Stage B) --------------- #
  # `causal_readout` above fixes ONE decision per bid position (decision t sees
  # the first t bids).  That is exact for a pure bidding LM, but Liar's Poker
  # interleaves CHALLENGE decisions that do NOT add a bid, so several distinct
  # decisions can share the same bid prefix while differing in globals
  # (num_challenges, is_rebid, ...).  A bid-indexed readout cannot represent them
  # (one slot, one globals vector).  `causal_readout_pd` generalises to an
  # ARBITRARY set of ``T`` decision readouts: decision j sees the first
  # ``dec_prefix_len[:, j]`` valid bids and folds in ITS OWN globals / legal.  It
  # is exact vs the single-token `full_forward` for EACH decision by the same
  # argument as `causal_readout` (decision tokens are query-only + independent:
  # they attend to hands + their bid prefix + themselves, never to bids beyond
  # their prefix nor to other decisions, so their position + mask fully determine
  # the output and parking them all at the tail is equivalent to running each
  # alone).  Layout (single sequence, N = D + L + T tokens):
  #   * hands     : 0 .. D-1
  #   * bids      : D .. D+L-1
  #   * decisions : D+L .. D+L+T-1     (decision j -> slot D+L+j)
  def _build_pd_mask(self, bid_valid: chex.Array,
                     dec_prefix_len: chex.Array) -> chex.Array:
    """Attention mask [B, N, N] for [hands(D), bids(L), decisions(T)].

    Structural rules (mask[b,q,k] True iff q may attend to k):
      * hand_q -> hand_k
      * bid_j  -> hands, bids 0..j (causal); NOT decisions
      * dec_j  -> hands, bids i where i < dec_prefix_len[b,j]; ITSELF; no other dec
    Bid keys are additionally gated by ``bid_valid``.
    """
    b = bid_valid.shape[0]
    d, ell = self.num_digits, self.max_seq
    tdec = dec_prefix_len.shape[1]
    n = d + ell + tdec
    idx = np.arange(n)
    q = idx[:, None]
    k = idx[None, :]
    hand_q = q < d
    bid_q = (q >= d) & (q < d + ell)
    dec_q = q >= d + ell
    hand_k = k < d
    bid_k = (k >= d) & (k < d + ell)
    dec_k = k >= d + ell
    j_of_q = q - d
    j_of_k = k - d
    dj_of_q = q - (d + ell)
    dj_of_k = k - (d + ell)
    # Batch-independent structure: hands, bid-causal, dec->hands, dec->itself.
    static = (
        (hand_q & hand_k)
        | (bid_q & (hand_k | (bid_k & (j_of_k <= j_of_q))))
        | (dec_q & (hand_k | (dec_k & (dj_of_k == dj_of_q))))
    )  # [N, N]
    static = jnp.asarray(static)[None, :, :]  # [1, N, N]
    # Per-batch decision->bid: decision j attends bid i iff i < prefix_len & valid.
    bid_idx = jnp.arange(ell)
    dec_bid = (
        (bid_idx[None, None, :] < dec_prefix_len[:, :, None])
        & bid_valid[:, None, :]
    )  # [B, T, L]
    dyn = jnp.zeros((b, n, n), dtype=bool)
    dyn = dyn.at[:, d + ell:, d:d + ell].set(dec_bid)
    mask = static | dyn
    key_valid = jnp.concatenate([
        self._hand_valid_mask(b),
        bid_valid,
        jnp.ones((b, tdec), dtype=bool),
    ], axis=1)  # [B, N]
    return mask & key_valid[:, None, :]

  def _causal_readout_pd_newhead(self, hand_hist, bids, bid_valid, observer,
                                 dec_prefix_len, per_decision_globals,
                                 legal_per_decision):
    """'perdigit_tailprob' causal multi-readout (LM-loss training path).

    Each of the T decisions expands to ``1 + D`` decision tokens (a global token
    + D per-digit tokens), laid out as ``[hands(D)][bids(L)][global(T)][digit(T*D)]``
    (digit token for decision j, digit dd -> slot ``T + j*D + dd``).  Every
    decision token attends to hands + its own valid bid prefix + ITSELF only
    (query-only, mutually independent), reusing ``_build_pd_mask`` with the
    ``T*(1+D)`` prefix-length vector.  Exact vs the single-token new-head
    ``full_forward`` for each decision by the same query-only/independent argument
    as the legacy causal readout.
    """
    b = hand_hist.shape[0]
    d, ell = self.num_digits, self.max_seq
    tdec = dec_prefix_len.shape[1]
    hand = self._hand_tokens(hand_hist)                        # [B, D, d]
    bid = self._bid_tokens(bids, observer, hand_hist)          # [B, L, d]
    gdec = self._global_dec_content(per_decision_globals)      # [B, T, d]
    ddec = self._digit_dec_content(per_decision_globals)       # [B, T, D, d]
    ddec = ddec.reshape(b, tdec * d, self.d_model)             # [B, T*D, d]
    x = jnp.concatenate([hand, bid, gdec, ddec], axis=1)       # [B, N, d]
    # RoPE seq positions: hands 0; bid j -> j+1; every dec token -> prefix+1.
    hand_pos = jnp.zeros((b, d), dtype=jnp.int32)
    bid_pos = jnp.broadcast_to(jnp.arange(1, ell + 1), (b, ell))
    g_pos = dec_prefix_len.astype(jnp.int32) + 1               # [B, T]
    dg_pos = jnp.broadcast_to(g_pos[:, :, None], (b, tdec, d)).reshape(b, tdec * d)
    pos = jnp.concatenate([hand_pos, bid_pos, g_pos, dg_pos], axis=1)  # [B, N]
    # Prefix lengths (mask): global T slots then digit T*D slots, all = decision's.
    g_pref = dec_prefix_len.astype(jnp.int32)                  # [B, T]
    dg_pref = jnp.broadcast_to(g_pref[:, :, None], (b, tdec, d)).reshape(b, tdec * d)
    dec_prefix_all = jnp.concatenate([g_pref, dg_pref], axis=1)  # [B, T*(1+D)]
    mask = self._build_pd_mask(bid_valid, dec_prefix_all)      # [B, N, N]
    # Digit rope ids: hands 1..D; bids number; global dec 0; digit dec 1..D tiled.
    dg_raw = jnp.broadcast_to(jnp.arange(1, d + 1), (b, tdec, d)).reshape(b, tdec * d)
    digit_raw = jnp.concatenate([
        self._hand_digit_ids(b), bids[..., 1],
        jnp.zeros((b, tdec), jnp.int32), dg_raw,
    ], axis=1)                                                 # [B, N]
    digit_pos = self._digit_rope_pos(digit_raw)
    for blk in self.blocks:
      x, _, _ = blk.full(x, mask, pos, digit_pos=digit_pos)
    hdec = x[:, d + ell:, :]                                   # [B, T*(1+D), d]
    h_global = hdec[:, :tdec, :]                               # [B, T, d]
    h_digit = hdec[:, tdec:, :].reshape(b, tdec, d, self.d_model)  # [B, T, D, d]
    hand_bt = jnp.broadcast_to(hand_hist[:, None, :], (b, tdec, d))
    logits, value = self._new_head_apply(
        h_global, h_digit, hand_bt, legal_per_decision)        # [B,T,A],[B,T,1]
    return logits.astype(jnp.float32), value[..., 0].astype(jnp.float32)

  def causal_readout_pd(self, hand_hist: chex.Array, bids: chex.Array,
                        bid_valid: chex.Array, observer: chex.Array,
                        dec_prefix_len: chex.Array,
                        per_decision_globals: chex.Array,
                        legal_per_decision: chex.Array):
    """ONE causal forward -> policy logits [B, T, A] + value [B, T] for T
    arbitrary decisions.  Decision ``j`` equals, up to fp rounding, the single-
    token ``full_forward``::

        full_forward(LPInputs(hand_hist, bids,
                              bid_valid = bid_valid & (arange(L) < dec_prefix_len[:, j]),
                              observer, per_decision_globals[:, j],
                              legal_per_decision[:, j]))

    Args:
      hand_hist: [B, D] int.
      bids:      [B, L, 3] int -- ordered actual bids (0-padded).
      bid_valid: [B, L] bool.
      observer:  [B] int.
      dec_prefix_len: [B, T] int -- #bids each decision has seen.
      per_decision_globals: [B, T, G] float -- globals AT each decision.
      legal_per_decision:   [B, T, A] bool  -- legal mask at each decision.
    Returns:
      (logits [B, T, A] float32, value [B, T] float32).
    """
    if self._new_head:
      return self._causal_readout_pd_newhead(
          hand_hist, bids, bid_valid, observer, dec_prefix_len,
          per_decision_globals, legal_per_decision)
    b = hand_hist.shape[0]
    d, ell = self.num_digits, self.max_seq
    tdec = dec_prefix_len.shape[1]
    hand = self._hand_tokens(hand_hist)                        # [B, D, d]
    bid = self._bid_tokens(bids, observer, hand_hist)          # [B, L, d]
    dec = (self.type_embed(jnp.asarray(TYPE_ACTION))[None, None, :]
           + self.globals_dense(per_decision_globals))         # [B, T, d]
    dec = dec.astype(self.dtype)  # match hand/bid compute dtype (bf16).
    x = jnp.concatenate([hand, bid, dec], axis=1)              # [B, N, d]
    # RoPE positions: hands=0; bid j -> j+1; decision -> (#bids seen) + 1, which
    # matches full_forward's decision pos (num_valid + 1) for the same prefix.
    hand_pos = jnp.zeros((b, d), dtype=jnp.int32)
    bid_pos = jnp.broadcast_to(jnp.arange(1, ell + 1), (b, ell))
    dec_pos = dec_prefix_len.astype(jnp.int32) + 1             # [B, T]
    pos = jnp.concatenate([hand_pos, bid_pos, dec_pos], axis=1)  # [B, N]
    mask = self._build_pd_mask(bid_valid, dec_prefix_len)      # [B, N, N]
    # Relative-digit bias over [hands, bids, decisions(T)] (decisions = null).
    if self._relbias:
      digit_ids = jnp.concatenate([
          self._hand_digit_ids(b),
          bids[..., 1],
          jnp.zeros((b, tdec), dtype=jnp.int32),
      ], axis=1)                                               # [B, N]
      cbias = self._combined_bias(digit_ids, mask)
      for blk in self.blocks:
        x, _, _ = blk.full(x, mask, pos, combined_bias=cbias)
    elif self._rope_digit:
      digit_ids = jnp.concatenate([
          self._hand_digit_ids(b),
          bids[..., 1],
          jnp.zeros((b, tdec), dtype=jnp.int32),               # decisions: null
      ], axis=1)                                               # [B, N]
      digit_ids = self._digit_rope_pos(digit_ids)              # from-max positions.
      for blk in self.blocks:
        x, _, _ = blk.full(x, mask, pos, digit_pos=digit_ids)
    else:
      for blk in self.blocks:
        x, _, _ = blk.full(x, mask, pos)
    h = x[:, d + ell:, :]                                      # [B, T, d]
    keys = self._action_keys()                                 # [A, d]
    qh = self.query_head(h)                                    # [B, T, d]
    scale = jnp.sqrt(jnp.asarray(self.d_model, jnp.float32))
    logits = jnp.matmul(qh, keys.T) / scale                    # [B, T, A]
    logits = jnp.where(legal_per_decision, logits, NEG_INF)
    value = self.value_out(nn.relu(self.value_h(h)))           # [B, T, 1]
    logits = logits.astype(jnp.float32)
    value = value[..., 0].astype(jnp.float32)                  # [B, T]
    return logits, value

  # ----------------------- Incremental KV-cache path ----------------------- #
  def prefill(self, hand_hist: chex.Array) -> List[Tuple[chex.Array, chex.Array]]:
    """Run the D hand tokens once; return per-layer (K, V) cache.

    Hand tokens attend bidirectionally among themselves (all valid, always
    present), matching the full-forward hand sub-block exactly.
    """
    x = self._hand_tokens(hand_hist)  # [B, D, d]
    b, d, _ = x.shape
    pos = jnp.zeros((b, d), dtype=jnp.int32)  # hands at position 0.
    # Bidirectional among hands, but only VALID hand slots are keys (capacity
    # hand-slot mask; all-True in non-capacity mode -> byte-identical).
    mask = jnp.broadcast_to(self._hand_valid_mask(b)[:, None, :], (b, d, d))
    # Hand<->hand digit routing (digits 1..D), matching full_forward: an additive
    # relative-digit bias (relbias) or a 2nd digit rotary on Q/K (rope_digit,
    # baked into cached k).
    bias = None
    dpos = None
    if self._relbias:
      hd = self._hand_digit_ids(b)
      bias = self._digit_bias(hd, hd)
    elif self._rope_digit:
      dpos = self._digit_rope_pos(self._hand_digit_ids(b))  # from-max positions.
    cache = []
    for blk in self.blocks:
      x, k, v = blk.full(x, mask, pos, bias, digit_pos=dpos)
      cache.append((k, v))
    return cache

  def append_bid(self, cache, bid_row: chex.Array, bid_index: int,
                 observer: chex.Array, hand_hist: chex.Array,
                 cached_digit_ids: chex.Array = None):
    """Append ONE bid (0-based `bid_index`) through the stack; return new cache.

    Args:
      cache: current per-layer [(K,V)] cache (hands + bids 0..bid_index-1).
      bid_row: [B, 3] the new bid (count, number, bidder).
      bid_index: 0-based index of this bid (its RoPE position is bid_index+1).
      observer: [B] observing player id.
      hand_hist: [B, D] observer hand counts (for my_count feature).
      cached_digit_ids: [B, T_c] digit ids of the CACHED tokens (hands 1..D then
        earlier bid numbers).  Required in 'relative_bias' mode so the new bid's
        query gets the correct relative-digit bias against the cache + itself;
        ignored (bias is None) in 'categorical' mode.
    """
    b = bid_row.shape[0]
    x = self._bid_tokens(bid_row[:, None, :], observer, hand_hist)  # [B,1,d]
    pos = jnp.full((b, 1), bid_index + 1, dtype=jnp.int32)
    bias = None
    dpos = None
    if self._relbias:
      own_digit = bid_row[:, 1:2]                          # [B,1] this bid's n
      k_digit = jnp.concatenate([cached_digit_ids, own_digit], axis=1)
      bias = self._digit_bias(own_digit, k_digit)          # [B,H,1,Tc+1]
    elif self._rope_digit:
      # This bid's number drives the digit rotary; the phase is baked into own_k,
      # so cached keys already carry their own digit phase (no cached_digit_ids).
      dpos = self._digit_rope_pos(bid_row[:, 1:2])         # [B,1] from-max pos.
    new_cache = []
    for blk, (ck, cv) in zip(self.blocks, cache):
      x, ok, ov = blk.step(x, pos, ck, cv, attn_bias=bias, digit_pos=dpos)
      new_cache.append((jnp.concatenate([ck, ok], axis=2),
                        jnp.concatenate([cv, ov], axis=2)))
    return new_cache

  def _decision_newhead(self, cache, globals_, legal_mask, hand_hist,
                        seq_pos, key_valid):
    """'perdigit_tailprob' decision query: run the ``1 + D`` decision tokens
    (global + per-digit) as length-(1+D) queries against ``cache``.

    The tokens are NOT appended to the cache.  An IDENTITY ``self_mask`` makes
    each decision token attend to the cache + ITSELF only (never to the other
    decision tokens), matching the full-forward decision-token mask -> the cached
    decision equals ``full_forward`` bit-for-bit up to fp rounding.  ``seq_pos``
    [B] is the shared decision RoPE position (num_bids+1); ``key_valid`` is the
    cache validity mask ([B, T_c] for the fixed cache, ``None`` for the growing
    cache where every cached slot is valid).
    """
    b = globals_.shape[0]
    d = self.num_digits
    gdec = self._global_dec_content(globals_)[:, None, :]      # [B,1,d]
    ddec = self._digit_dec_content(globals_)                   # [B,D,d]
    x = jnp.concatenate([gdec, ddec], axis=1)                  # [B,1+D,d]
    pos = jnp.broadcast_to(seq_pos.astype(jnp.int32)[:, None], (b, 1 + d))
    dec_raw = jnp.concatenate([
        jnp.zeros((b, 1), jnp.int32),
        jnp.broadcast_to(jnp.arange(1, d + 1), (b, d)),
    ], axis=1)                                                 # [B,1+D]
    dpos = self._digit_rope_pos(dec_raw)                       # [B,1+D]
    self_m = jnp.eye(1 + d, dtype=bool)                        # identity self-mask
    for blk, (ck, cv) in zip(self.blocks, cache):
      x, _, _ = blk.step(x, pos, ck, cv, key_valid=key_valid,
                         digit_pos=dpos, self_mask=self_m)
    h_global = x[:, 0, :]                                       # [B,d]
    h_digit = x[:, 1:, :]                                       # [B,D,d]
    return self._new_head_apply(h_global, h_digit, hand_hist, legal_mask)

  def decision_logits(self, cache, globals_: chex.Array,
                      legal_mask: chex.Array, num_bids: int,
                      hand_hist: chex.Array = None):
    """Run the transient DECISION token as a length-1 query against `cache`.

    The DECISION token is NOT appended to the cache (globals change each step).
    Its RoPE position is num_bids+1 (matching the full-forward decision pos).
    ``hand_hist`` [B, D] is REQUIRED only for the 'perdigit_tailprob' head (the
    per-digit tokens' belief feature); it is ignored by the legacy head.
    """
    b = globals_.shape[0]
    if self._new_head:
      assert hand_hist is not None, (
          "decision_logits needs hand_hist for the 'perdigit_tailprob' head")
      seq_pos = jnp.full((b,), num_bids + 1, dtype=jnp.int32)
      return self._decision_newhead(
          cache, globals_, legal_mask, hand_hist, seq_pos, None)
    x = self._decision_token(globals_)[:, None, :]  # [B,1,d]
    pos = jnp.full((b, 1), num_bids + 1, dtype=jnp.int32)
    # DECISION is a null-digit (0) query.  In rope_digit its digit slice must get
    # angle-0 (identity) rotation to match the full-forward decision token, so we
    # pass digit_pos=0 (NOT None -> that would seq-rotate the digit slice too).
    dpos = jnp.zeros((b, 1), dtype=jnp.int32) if self._rope_digit else None
    for blk, (ck, cv) in zip(self.blocks, cache):
      x, _, _ = blk.step(x, pos, ck, cv, digit_pos=dpos)
    h = x[:, 0, :]
    return self._heads(h, legal_mask)

  # ------------------- Fixed-size (preallocated) KV cache ------------------- #
  # The growing-cache methods above (`prefill` / `append_bid` / `decision_logits`)
  # `jnp.concatenate` a slot per bid, so the cache's time axis GROWS.  That is
  # perfect for a Python-loop test but cannot live in a `lax.scan` carry, whose
  # leaves must keep a STATIC shape.  The `*_fixed` methods below implement the
  # same math against a PREALLOCATED cache of static length ``D + max_seq``:
  #   * slots 0..D-1    : the (always-valid) hand tokens (from `prefill_fixed`).
  #   * slots D..D+K-1  : the first K real bids (K == per-env `counter`).
  #   * slots D+K..     : padding (masked out via `key_valid`, so their garbage
  #                       contents never affect attention).
  # A bid is written in-place at slot ``D + counter`` and the counter bumped,
  # so `append_bid_fixed` still touches exactly ONE new slot per turn.  These
  # feed the Stage-2 self-play rollout (per-player caches in the scan carry).
  def _bid_key_valid(self, counter: chex.Array, max_seq: int) -> chex.Array:
    """[B, D+max_seq] bool: hands always valid, first `counter` bid slots valid."""
    b = counter.shape[0]
    d = self.num_digits
    hand_valid = self._hand_valid_mask(b)  # capacity hand-slot mask (all-True default)
    bid_valid = jnp.arange(max_seq)[None, :] < counter[:, None]  # [B, max_seq]
    return jnp.concatenate([hand_valid, bid_valid], axis=1)

  def prefill_fixed(self, hand_hist: chex.Array, max_seq: int):
    """Prefill hands into a PREALLOCATED [B,H,D+max_seq,d_head] per-layer cache.

    Identical hand math to `prefill`; the bid region is zero-padded and made
    valid lazily (per-env `counter`) via `key_valid` in later steps.
    """
    x = self._hand_tokens(hand_hist)  # [B, D, d]
    b, d, _ = x.shape
    pos = jnp.zeros((b, d), dtype=jnp.int32)  # hands at position 0.
    # Bidirectional among hands, but only VALID hand slots are keys (capacity
    # hand-slot mask; all-True in non-capacity mode -> byte-identical).
    mask = jnp.broadcast_to(self._hand_valid_mask(b)[:, None, :], (b, d, d))
    bias = None
    dpos = None
    if self._relbias:
      hd = self._hand_digit_ids(b)
      bias = self._digit_bias(hd, hd)
    elif self._rope_digit:
      dpos = self._digit_rope_pos(self._hand_digit_ids(b))  # from-max positions.
    cache = []
    for blk in self.blocks:
      x, k, v = blk.full(x, mask, pos, bias, digit_pos=dpos)  # k,v: [B,H,D,d_head]
      h, dh = k.shape[1], k.shape[3]
      pad = jnp.zeros((b, h, max_seq, dh), dtype=k.dtype)
      cache.append((jnp.concatenate([k, pad], axis=2),
                    jnp.concatenate([v, pad], axis=2)))
    return cache

  def append_bid_fixed(self, cache, bid_row: chex.Array, counter: chex.Array,
                       observer: chex.Array, hand_hist: chex.Array,
                       bid_active: chex.Array, max_seq: int,
                       digit_ids: chex.Array = None):
    """Append ONE bid in-place at slot ``D + counter`` (per env), fixed shapes.

    Args:
      cache: per-layer [(K,V)] each [B, H, D+max_seq, d_head].
      bid_row: [B, 3] the new bid (count, number, bidder).  Ignored where
        ``bid_active`` is False (masked write), so its value there is arbitrary.
      counter: [B] int -- number of bids already in the cache (== new bid index).
      observer: [B] int -- this cache's observing player (0 for cache_p0, ...).
      hand_hist: [B, D] the observer's hand histogram (for the my_count feature).
      bid_active: [B] bool -- envs that actually placed a bid this turn.
      max_seq: static bid-slot capacity.
      digit_ids: [B, D+max_seq] digit ids of the fixed cache slots (hands 1..D,
        then real bid numbers, then 0-padding).  Required in 'relative_bias'
        mode; the padded slots are digit 0 (null -> zero bias) AND masked by
        ``key_valid``, so garbage there is doubly harmless.  Shared across both
        player caches (the bid NUMBER is public).  Ignored in 'categorical'.
    Returns:
      updated cache (same shapes).  Counter bump is the caller's responsibility
      (``counter + bid_active``) so both player caches stay in lock-step.
    """
    d = self.num_digits
    key_valid = self._bid_key_valid(counter, max_seq)  # [B, D+max_seq]
    x = self._bid_tokens(bid_row[:, None, :], observer, hand_hist)  # [B,1,d]
    b = x.shape[0]
    pos = (counter + 1)[:, None].astype(jnp.int32)  # RoPE pos = bid_index+1.
    bias = None
    dpos = None
    if self._relbias:
      own_digit = bid_row[:, 1:2]                          # [B,1] this bid's n
      k_digit = jnp.concatenate([digit_ids, own_digit], axis=1)  # [B,D+max_seq+1]
      bias = self._digit_bias(own_digit, k_digit)          # [B,H,1,D+max_seq+1]
    elif self._rope_digit:
      # This bid's number drives the digit rotary (baked into own_k); the
      # threaded `digit_ids` cache array is unused in this mode.
      dpos = self._digit_rope_pos(bid_row[:, 1:2])         # [B,1] from-max pos.
    slot = d + counter  # [B] absolute cache index to write.
    total = d + max_seq
    write = (jnp.arange(total)[None, :] == slot[:, None]) & bid_active[:, None]
    write = write[:, None, :, None]  # [B, 1, total, 1] broadcast over H, d_head.
    new_cache = []
    for blk, (ck, cv) in zip(self.blocks, cache):
      x, ok, ov = blk.step(x, pos, ck, cv, key_valid=key_valid, attn_bias=bias,
                           digit_pos=dpos)
      new_cache.append((jnp.where(write, ok, ck), jnp.where(write, ov, cv)))
    return new_cache

  def decision_logits_fixed(self, cache, globals_: chex.Array,
                            legal_mask: chex.Array, counter: chex.Array,
                            max_seq: int, hand_hist: chex.Array = None):
    """DECISION query against the preallocated cache (per-env `counter` bids).

    No relative-digit bias here: the DECISION token is a null-digit (id 0) query,
    so its bias row is identically 0 in every mode (matching full_forward, whose
    decision row is also null) -- omitting it is exact, not an approximation.
    ``hand_hist`` [B, D] (the acting observer's histogram) is REQUIRED only for
    the 'perdigit_tailprob' head (the belief feature); ignored by the legacy head.
    """
    key_valid = self._bid_key_valid(counter, max_seq)  # [B, D+max_seq]
    if self._new_head:
      assert hand_hist is not None, (
          "decision_logits_fixed needs hand_hist for 'perdigit_tailprob'")
      seq_pos = (counter + 1).astype(jnp.int32)
      return self._decision_newhead(
          cache, globals_, legal_mask, hand_hist, seq_pos, key_valid)
    x = self._decision_token(globals_)[:, None, :]  # [B,1,d]
    pos = (counter + 1)[:, None].astype(jnp.int32)   # matches full-forward dec pos.
    b = globals_.shape[0]
    # DECISION is a null-digit (0) query: in rope_digit its digit slice gets
    # angle-0 (identity) rotation (digit_pos=0), matching full_forward's null
    # decision token.  In categorical/relbias digit_pos=None (unchanged).
    dpos = jnp.zeros((b, 1), dtype=jnp.int32) if self._rope_digit else None
    for blk, (ck, cv) in zip(self.blocks, cache):
      x, _, _ = blk.step(x, pos, ck, cv, key_valid=key_valid, digit_pos=dpos)
    h = x[:, 0, :]
    return self._heads(h, legal_mask)
