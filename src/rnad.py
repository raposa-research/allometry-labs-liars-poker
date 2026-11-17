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
# Modified by Allometry Labs, 2026. Derived from OpenSpiel's
# open_spiel/python/algorithms/rnad/rnad.py (as of OpenSpiel v1.5; no longer in
# current OpenSpiel). Main changes: ported from Haiku to Flax; N-player support
# with zero-sum regularization scaling (zero_sum_reg_scaling, opp_reg_coef);
# alternative policy updates (PPO clipped surrogate, NashPG) and advantage
# estimators (GRPO alongside v-trace); optional entropy bonuses; a factored-MLP
# network; learning-rate warmup. This file, including these modifications, is
# licensed under the Apache License 2.0 (see LICENSE-APACHE), not the
# repository's PolyForm Noncommercial license.
"""Python implementation of R-NaD (https://arxiv.org/pdf/2206.15378.pdf)."""

import enum
import functools
from typing import Any, Callable, Optional, Sequence, Tuple

import chex
import flax.linen as nn
import flax.serialization
import jax
from jax import lax
from jax import numpy as jnp
from jax import tree_util as tree
import numpy as np
import optax

from open_spiel.python import policy as policy_lib
import pyspiel

# Reused for the (config-gated) 'tailprob' count encoding of the factored head:
# the SAME closed-form log-survival binomial table the transformer's
# 'perdigit_tailprob' head uses for its q(c) feature, so the two match
# bit-for-bit for the same game config.  transformer_seq does not import rnad,
# so this top-level import introduces no cycle.
from transformer_seq import _log_survival_table


# Some handy aliases.
# Since most of these are just aliases for a "bag of tensors", the goal
# is to improve the documentation, and not to actually enforce correctness
# through pytype.
Params = chex.ArrayTree


class EntropySchedule:
  """An increasing list of steps where the regularisation network is updated.

  Example
    EntropySchedule([3, 5, 10], [2, 4, 1])
    =>   [0, 3, 6, 11, 16, 21, 26, 36]
          | 3 x2 |      5 x4     | 10 x1
  """

  def __init__(self, *, sizes: Sequence[int], repeats: Sequence[int]):
    """Constructs a schedule of entropy iterations.

    Args:
      sizes: the list of iteration sizes.
      repeats: the list, parallel to sizes, with the number of times for each
        size from `sizes` to repeat.
    """
    try:
      if len(repeats) != len(sizes):
        raise ValueError("`repeats` must be parallel to `sizes`.")
      if not sizes:
        raise ValueError("`sizes` and `repeats` must not be empty.")
      if any([(repeat <= 0) for repeat in repeats]):
        raise ValueError("All repeat values must be strictly positive")
      if repeats[-1] != 1:
        raise ValueError("The last value in `repeats` must be equal to 1, "
                         "ince the last iteration size is repeated forever.")
    except ValueError as e:
      raise ValueError(
          f"Entropy iteration schedule: repeats ({repeats}) and sizes"
          f" ({sizes})."
      ) from e

    schedule = [0]
    for size, repeat in zip(sizes, repeats):
      schedule.extend([schedule[-1] + (i + 1) * size for i in range(repeat)])

    self.schedule = jnp.array(schedule, dtype=jnp.int32)

  def __call__(self, learner_step: int) -> Tuple[float, bool]:
    """Entropy scheduling parameters for a given `learner_step`.

    Args:
      learner_step: The current learning step.

    Returns:
      alpha: The mixing weight (from [0, 1]) of the previous policy with
        the one before for computing the intrinsic reward.
      update_target_net: A boolean indicator for updating the target network
        with the current network.
    """

    # The complexity below is because at some point we might go past
    # the explicit schedule, and then we'd need to just use the last step
    # in the schedule and apply the logic of
    # ((learner_step - last_step) % last_iteration) == 0)

    # The schedule might look like this:
    # X----X-------X--X--X--X--------X
    # learner_step | might be here ^    |
    # or there     ^                    |
    # or even past the schedule         ^

    # We need to deal with two cases below.
    # Instead of going for the complicated conditional, let's just
    # compute both and then do the A * s + B * (1 - s) with s being a bool
    # selector between A and B.

    # 1. assume learner_step is past the schedule,
    #    ie schedule[-1] <= learner_step.
    last_size = self.schedule[-1] - self.schedule[-2]
    last_start = self.schedule[-1] + (
        learner_step - self.schedule[-1]) // last_size * last_size
    # 2. assume learner_step is within the schedule.
    start = jnp.amax(self.schedule * (self.schedule <= learner_step))
    finish = jnp.amin(
        self.schedule * (learner_step < self.schedule),
        initial=self.schedule[-1],
        where=(learner_step < self.schedule))
    size = finish - start

    # Now select between the two.
    beyond = (self.schedule[-1] <= learner_step)  # Are we past the schedule?
    iteration_start = (last_start * beyond + start * (1 - beyond))
    iteration_size = (last_size * beyond + size * (1 - beyond))

    update_target_net = jnp.logical_and(
        learner_step > 0,
        jnp.sum(learner_step == iteration_start + iteration_size - 1),
    )
    alpha = jnp.minimum(
        (2.0 * (learner_step - iteration_start)) / iteration_size, 1.0)

    return alpha, update_target_net  # pytype: disable=bad-return-type  # jax-types


@chex.dataclass(frozen=True)
class FineTuning:
  """Fine tuning options, aka policy post-processing.

  Even when fully trained, the resulting softmax-based policy may put
  a small probability mass on bad actions. This results in an agent
  waiting for the opponent (itself in self-play) to commit an error.

  To address that the policy is post-processed using:
  - thresholding: any action with probability smaller than self.threshold
    is simply removed from the policy.
  - discretization: the probability values are rounded to the closest
    multiple of 1/self.discretization.

  The post-processing is used on the learner, and thus must be jit-friendly.
  """
  # The learner step after which the policy post processing (aka finetuning)
  # will be enabled when learning. A strictly negative value is equivalent
  # to infinity, ie disables finetuning completely.
  from_learner_steps: int = -1
  # All policy probabilities below `threshold` are zeroed out. Thresholding
  # is disabled if this value is non-positive.
  policy_threshold: float = 0.03
  # Rounds the policy probabilities to the "closest"
  # multiple of 1/`self.discretization`.
  # Discretization is disabled for non-positive values.
  policy_discretization: int = 32

  def __call__(self, policy: chex.Array, mask: chex.Array,
               learner_steps: int) -> chex.Array:
    """A configurable fine tuning of a policy."""
    chex.assert_equal_shape((policy, mask))
    do_finetune = jnp.logical_and(self.from_learner_steps >= 0,
                                  learner_steps > self.from_learner_steps)

    return jnp.where(do_finetune, self.post_process_policy(policy, mask),
                     policy)

  def post_process_policy(
      self,
      policy: chex.Array,
      mask: chex.Array,
  ) -> chex.Array:
    """Unconditionally post process a given masked policy."""
    chex.assert_equal_shape((policy, mask))
    policy = self._threshold(policy, mask)
    policy = self._discretize(policy)
    return policy

  def _threshold(self, policy: chex.Array, mask: chex.Array) -> chex.Array:
    """Remove from the support the actions 'a' where policy(a) < threshold."""
    chex.assert_equal_shape((policy, mask))
    if self.policy_threshold <= 0:
      return policy

    mask = mask * (
        # Values over the threshold.
        (policy >= self.policy_threshold) +
        # Degenerate case is when policy is less than threshold *everywhere*.
        # In that case we just keep the policy as-is.
        (jnp.max(policy, axis=-1, keepdims=True) < self.policy_threshold))
    return mask * policy / jnp.sum(mask * policy, axis=-1, keepdims=True)

  def _discretize(self, policy: chex.Array) -> chex.Array:
    """Round all action probabilities to a multiple of 1/self.discretize."""
    if self.policy_discretization <= 0:
      return policy

    # The unbatched/single policy case:
    if len(policy.shape) == 1:
      return self._discretize_single(policy)

    # policy may be [B, A] or [T, B, A], etc.
    dims = len(policy.shape) - 1

    # TODO(author18): avoid mixing vmap and BatchApply since the two could
    # be folded into either a single BatchApply or a sequence of vmaps, but
    # not the mix.
    vmapped = jax.vmap(self._discretize_single)
    orig_shape = policy.shape
    policy = vmapped(policy.reshape(-1, orig_shape[-1])).reshape(orig_shape)

    return policy

  def _discretize_single(self, mu: chex.Array) -> chex.Array:
    """A version of self._discretize but for the unbatched data."""
    # TODO(author18): try to merge _discretize and _discretize_single
    # into one function that handles both batched and unbatched cases.
    if len(mu.shape) == 2:
      mu_ = jnp.squeeze(mu, axis=0)
    else:
      mu_ = mu
    n_actions = mu_.shape[-1]
    roundup = jnp.ceil(mu_ * self.policy_discretization).astype(jnp.int32)
    result = jnp.zeros_like(mu_)
    order = jnp.argsort(-mu_)  # Indices of descending order.
    weight_left = self.policy_discretization

    def f_disc(i, order, roundup, weight_left, result):
      x = jnp.minimum(roundup[order[i]], weight_left)
      result = jax.numpy.where(weight_left >= 0, result.at[order[i]].add(x),
                               result)
      weight_left -= x
      return i + 1, order, roundup, weight_left, result

    def f_scan_scan(carry, x):
      i, order, roundup, weight_left, result = carry
      i_next, order_next, roundup_next, weight_left_next, result_next = f_disc(
          i, order, roundup, weight_left, result)
      carry_next = (i_next, order_next, roundup_next, weight_left_next,
                    result_next)
      return carry_next, x

    (_, _, _, weight_left_next, result_next), _ = jax.lax.scan(
        f_scan_scan,
        init=(jnp.asarray(0), order, roundup, weight_left, result),
        xs=None,
        length=n_actions)

    result_next = jnp.where(weight_left_next > 0,
                            result_next.at[order[0]].add(weight_left_next),
                            result_next)
    if len(mu.shape) == 2:
      result_next = jnp.expand_dims(result_next, axis=0)
    return result_next / self.policy_discretization


def _legal_policy(logits: chex.Array, legal_actions: chex.Array) -> chex.Array:
  """A soft-max policy that respects legal_actions."""
  chex.assert_equal_shape((logits, legal_actions))
  # Fiddle a bit to make sure we don't generate NaNs or Inf in the middle.
  l_min = logits.min(axis=-1, keepdims=True)
  logits = jnp.where(legal_actions, logits, l_min)
  logits -= logits.max(axis=-1, keepdims=True)
  logits *= legal_actions
  exp_logits = jnp.where(legal_actions, jnp.exp(logits),
                         0)  # Illegal actions become 0.
  exp_logits_sum = jnp.sum(exp_logits, axis=-1, keepdims=True)
  return exp_logits / exp_logits_sum


def legal_log_policy(logits: chex.Array,
                     legal_actions: chex.Array) -> chex.Array:
  """Return the log of the policy on legal action, 0 on illegal action."""
  chex.assert_equal_shape((logits, legal_actions))
  # logits_masked has illegal actions set to -inf.
  logits_masked = logits + jnp.log(legal_actions)
  max_legal_logit = logits_masked.max(axis=-1, keepdims=True)
  logits_masked = logits_masked - max_legal_logit
  # exp_logits_masked is 0 for illegal actions.
  exp_logits_masked = jnp.exp(logits_masked)

  baseline = jnp.log(jnp.sum(exp_logits_masked, axis=-1, keepdims=True))
  # Subtract baseline from logits. We do not simply return
  #     logits_masked - baseline
  # because that has -inf for illegal actions, or
  #     legal_actions * (logits_masked - baseline)
  # because that leads to 0 * -inf == nan for illegal actions.
  log_policy = jnp.multiply(legal_actions,
                            (logits - max_legal_logit - baseline))
  return log_policy


def _opp_reg_weight(num_players: int, zero_sum_scaling: bool = True) -> float:
  """Weight applied to an OPPONENT's regularization term, 1/(N-1).

  R-NaD adds a regularization reward to the game reward: player k receives its
  own ``reg_k`` and pays the others'. With a fixed weight of 1 per opponent the
  regularized returns sum to

      sum_k [ r_k + reg_k - sum_{j!=k} reg_j ] = -(N-2) * sum_j reg_j

  which is 0 ONLY at N=2. At N>=3 the regularized game is no longer zero-sum --
  it manufactures total payoff whenever anyone deviates from its anchor (each
  own-reg is -eta*KL(pi||pi_reg) <= 0) -- so it is not the regularized zero-sum
  game whose Nash R-NaD is derived to find. Weighting each opponent by 1/(N-1)
  restores the identity for every N:

      sum_k [ r_k + reg_k - (1/(N-1)) * sum_{j!=k} reg_j ] = 0

  At N=2 the weight is 1/1 = 1, i.e. EXACTLY the previous behavior, so every
  2-player result is unaffected. ``zero_sum_scaling=False`` restores the old
  fixed weight of 1 for reproducing pre-fix runs.
  """
  if not zero_sum_scaling:
    return 1.0
  return 1.0 / max(num_players - 1, 1)


def _player_others(player_ids: chex.Array, valid: chex.Array,
                   player: int, opp_weight: float = 1.0) -> chex.Array:
  """A vector of 1 for the current player and -opp_weight for others.

  Args:
    player_ids: Tensor [...] containing player ids (0 <= player_id < N).
    valid: Tensor [...] containing whether these states are valid.
    player: The player id as int.
    opp_weight: magnitude of the opponent entries; 1.0 (default) reproduces the
      original +1/-1 vector. Pass ``_opp_reg_weight(num_players)`` = 1/(N-1) to
      keep the REGULARIZED game zero-sum at 3+ players (see that function).

  Returns:
    player_other: is 1 for the current player and -opp_weight for others
    [..., 1].
  """
  chex.assert_equal_shape((player_ids, valid))
  current_player_tensor = (player_ids == player).astype(jnp.int32)  # pytype: disable=attribute-error  # numpy-scalars

  if opp_weight == 1.0:
    res = 2 * current_player_tensor - 1
  else:
    cur = current_player_tensor.astype(jnp.float32)
    res = cur - (1.0 - cur) * opp_weight
  res = res * valid
  return jnp.expand_dims(res, axis=-1)


def _policy_ratio(pi: chex.Array, mu: chex.Array, actions_oh: chex.Array,
                  valid: chex.Array) -> chex.Array:
  """Returns a ratio of policy pi/mu when selecting action a.

  By convention, this ratio is 1 on non valid states
  Args:
    pi: the policy of shape [..., A].
    mu: the sampling policy of shape [..., A].
    actions_oh: a one-hot encoding of the current actions of shape [..., A].
    valid: 0 if the state is not valid and else 1 of shape [...].

  Returns:
    pi/mu on valid states and 1 otherwise. The shape is the same
    as pi, mu or actions_oh but without the last dimension A.
  """
  chex.assert_equal_shape((pi, mu, actions_oh))
  chex.assert_shape((valid,), actions_oh.shape[:-1])

  def _select_action_prob(pi):
    return (jnp.sum(actions_oh * pi, axis=-1, keepdims=False) * valid +
            (1 - valid))

  pi_actions_prob = _select_action_prob(pi)
  mu_actions_prob = _select_action_prob(mu)
  return pi_actions_prob / mu_actions_prob


def _where(pred: chex.Array, true_data: chex.ArrayTree,
           false_data: chex.ArrayTree) -> chex.ArrayTree:
  """Similar to jax.where but treats `pred` as a broadcastable prefix."""

  def _where_one(t, f):
    chex.assert_equal_rank((t, f))
    # Expand the dimensions of pred if true_data and false_data are higher rank.
    p = jnp.reshape(pred, pred.shape + (1,) * (len(t.shape) - len(pred.shape)))
    return jnp.where(p, t, f)

  return tree.tree_map(_where_one, true_data, false_data)


def _has_played(valid: chex.Array, player_id: chex.Array,
                player: int) -> chex.Array:
  """Compute a mask of states which have a next state in the sequence."""
  chex.assert_equal_shape((valid, player_id))

  def _loop_has_played(carry, x):
    valid, player_id = x
    chex.assert_equal_shape((valid, player_id))

    our_res = jnp.ones_like(player_id)
    opp_res = carry
    reset_res = jnp.zeros_like(carry)

    our_carry = carry
    opp_carry = carry
    reset_carry = jnp.zeros_like(player_id)

    # pyformat: disable
    return _where(valid, _where((player_id == player),
                                (our_carry, our_res),
                                (opp_carry, opp_res)),
                  (reset_carry, reset_res))
    # pyformat: enable

  _, result = lax.scan(
      f=_loop_has_played,
      init=jnp.zeros_like(player_id[-1]),
      xs=(valid, player_id),
      reverse=True)
  return result


# V-Trace
#
# Custom implementation of VTrace to handle trajectories having a mix of
# different player steps. The standard rlax.vtrace can't be applied here
# out of the box because a trajectory could look like '121211221122'.


def v_trace(
    v: chex.Array,
    valid: chex.Array,
    player_id: chex.Array,
    acting_policy: chex.Array,
    merged_policy: chex.Array,
    merged_log_policy: chex.Array,
    player_others: chex.Array,
    actions_oh: chex.Array,
    reward: chex.Array,
    player: int,
    # Scalars below.
    eta: float,
    lambda_: float,
    c: float,
    rho: float,
    opp_reg_weight: float = 1.0,
) -> Tuple[Any, Any, Any]:
  """Custom VTrace for trajectories with a mix of different player steps."""
  gamma = 1.0

  has_played = _has_played(valid, player_id, player)

  policy_ratio = _policy_ratio(merged_policy, acting_policy, actions_oh, valid)
  inv_mu = _policy_ratio(
      jnp.ones_like(merged_policy), acting_policy, actions_oh, valid)

  # NOTE this re-derives the sign from player_others instead of using it
  # directly, so the opponent magnitude must be supplied separately:
  # opp_reg_weight is 1.0 (the original behavior, exact at N=2) or 1/(N-1),
  # which keeps the regularized game zero-sum at 3+ players. The `== 1` test
  # still selects exactly the acting player's steps, and invalid steps keep
  # their old -opp_reg_weight value (they are dropped by the scan's reset).
  eta_reg_entropy = (
    -eta *
    jnp.sum(merged_policy * merged_log_policy, axis=-1) *
    jnp.where(
      jnp.squeeze(player_others, axis=-1) == 1,
      1,
      -opp_reg_weight
    )
  )
  eta_log_policy = -eta * merged_log_policy * player_others

  @chex.dataclass(frozen=True)
  class LoopVTraceCarry:
    """The carry of the v-trace scan loop."""
    reward: chex.Array
    # The cumulated reward until the end of the episode. Uncorrected (v-trace).
    # Gamma discounted and includes eta_reg_entropy.
    reward_uncorrected: chex.Array
    next_value: chex.Array
    next_v_target: chex.Array
    importance_sampling: chex.Array

  init_state_v_trace = LoopVTraceCarry(
      reward=jnp.zeros_like(reward[-1]),
      reward_uncorrected=jnp.zeros_like(reward[-1]),
      next_value=jnp.zeros_like(v[-1]),
      next_v_target=jnp.zeros_like(v[-1]),
      importance_sampling=jnp.ones_like(policy_ratio[-1]))

  def _loop_v_trace(carry: LoopVTraceCarry, x) -> Tuple[LoopVTraceCarry, Any]:
    (cs, player_id, v, reward, eta_reg_entropy, valid, inv_mu, actions_oh,
     eta_log_policy) = x

    reward_uncorrected = (
        reward + gamma * carry.reward_uncorrected + eta_reg_entropy)
    discounted_reward = reward + gamma * carry.reward

    # V-target:
    our_v_target = (
        v + jnp.expand_dims(
            jnp.minimum(rho, cs * carry.importance_sampling), axis=-1) *
        (jnp.expand_dims(reward_uncorrected, axis=-1) +
         gamma * carry.next_value - v) + lambda_ * jnp.expand_dims(
             jnp.minimum(c, cs * carry.importance_sampling), axis=-1) * gamma *
        (carry.next_v_target - carry.next_value))

    opp_v_target = jnp.zeros_like(our_v_target)
    reset_v_target = jnp.zeros_like(our_v_target)

    # Learning output:
    our_learning_output = (
        v +  # value
        eta_log_policy +  # regularisation
        actions_oh * jnp.expand_dims(inv_mu, axis=-1) *
        (jnp.expand_dims(discounted_reward, axis=-1) + gamma * jnp.expand_dims(
            carry.importance_sampling, axis=-1) * carry.next_v_target - v))

    opp_learning_output = jnp.zeros_like(our_learning_output)
    reset_learning_output = jnp.zeros_like(our_learning_output)

    # State carry:
    our_carry = LoopVTraceCarry(
        reward=jnp.zeros_like(carry.reward),
        next_value=v,
        next_v_target=our_v_target,
        reward_uncorrected=jnp.zeros_like(carry.reward_uncorrected),
        importance_sampling=jnp.ones_like(carry.importance_sampling))
    opp_carry = LoopVTraceCarry(
        reward=eta_reg_entropy + cs * discounted_reward,
        reward_uncorrected=reward_uncorrected,
        next_value=gamma * carry.next_value,
        next_v_target=gamma * carry.next_v_target,
        importance_sampling=cs * carry.importance_sampling)
    reset_carry = init_state_v_trace

    # Invalid turn: init_state_v_trace and (zero target, learning_output)
    # pyformat: disable
    return _where(
      valid,  # pytype: disable=bad-return-type  # numpy-scalars
      _where((player_id == player),
             (our_carry, (our_v_target, our_learning_output)),
             (opp_carry, (opp_v_target, opp_learning_output))),
      (reset_carry, (reset_v_target, reset_learning_output))
    )
    # pyformat: enable

  _, (v_target, learning_output) = lax.scan(
      f=_loop_v_trace,
      init=init_state_v_trace,
      xs=(policy_ratio, player_id, v, reward, eta_reg_entropy, valid, inv_mu,
          actions_oh, eta_log_policy),
      reverse=True)

  return v_target, has_played, learning_output


def get_loss_v(v_list: Sequence[chex.Array],
               v_target_list: Sequence[chex.Array],
               mask_list: Sequence[chex.Array]) -> chex.Array:
  """Define the loss function for the critic."""
  chex.assert_trees_all_equal_shapes(v_list, v_target_list)
  # v_list and v_target_list come with a degenerate trailing dimension,
  # which mask_list tensors do not have.
  chex.assert_shape(mask_list, v_list[0].shape[:-1])
  loss_v_list = []
  for (v_n, v_target, mask) in zip(v_list, v_target_list, mask_list):
    assert v_n.shape[0] == v_target.shape[0]

    loss_v = jnp.expand_dims(
        mask, axis=-1) * (v_n - lax.stop_gradient(v_target))**2
    normalization = jnp.sum(mask)
    loss_v = jnp.sum(loss_v) / (normalization + (normalization == 0.0))

    loss_v_list.append(loss_v)
  return sum(loss_v_list)


def apply_force_with_threshold(decision_outputs: chex.Array, force: chex.Array,
                               threshold: float,
                               threshold_center: chex.Array,
                               lower_threshold: chex.Array = None) -> chex.Array:
  """Apply the force with below a given threshold.

  The NeuRD trust region freezes an action against being pushed DOWN once its
  centered logit drops to the lower bound (default ``-threshold``, i.e. ``-beta``,
  symmetric with the ``+threshold`` upper bound).  ``lower_threshold`` (per-action,
  same shape as ``decision_outputs``) overrides that lower bound so suppression can
  continue below ``-beta`` -- e.g. down to a heuristic teacher's (much more
  negative) centered logit -- while the upper bound stays the scalar ``+threshold``.
  """
  chex.assert_equal_shape((decision_outputs, force, threshold_center))
  centered = decision_outputs - threshold_center
  if lower_threshold is None:
    can_decrease = centered > -threshold
  else:
    can_decrease = centered > lower_threshold
  can_increase = centered < threshold
  force_negative = jnp.minimum(force, 0.0)
  force_positive = jnp.maximum(force, 0.0)
  clipped_force = can_decrease * force_negative + can_increase * force_positive
  return decision_outputs * lax.stop_gradient(clipped_force)


def renormalize(loss: chex.Array, mask: chex.Array) -> chex.Array:
  """The `normalization` is the number of steps over which loss is computed."""
  chex.assert_equal_shape((loss, mask))
  loss = jnp.sum(loss * mask)
  normalization = jnp.sum(mask)
  return loss / (normalization + (normalization == 0.0))


def get_loss_nerd(logit_list: Sequence[chex.Array],
                  policy_list: Sequence[chex.Array],
                  q_vr_list: Sequence[chex.Array],
                  valid: chex.Array,
                  player_ids: Sequence[chex.Array],
                  legal_actions: chex.Array,
                  importance_sampling_correction: Sequence[chex.Array],
                  clip: float = 100,
                  threshold: float = 2,
                  teacher_centered_list: Sequence[chex.Array] = None,
                  loss_action_mask: chex.Array = None) -> chex.Array:
  """Define the nerd loss.

  ``teacher_centered_list`` (optional, one [..., A] per player): the heuristic
  teacher's mean-centered logit.  When given, the per-action LOWER trust-region
  bound becomes ``min(teacher_centered, -threshold)`` -- so an action the teacher
  nearly-zeroes (very negative centered logit) can be pushed BELOW ``-beta`` toward
  the teacher's level, draining the ``-beta`` floor pile, while actions the teacher
  likes (>= -beta) keep the unchanged ``-beta`` floor.

  ``loss_action_mask`` (optional [..., A]): which actions contribute to the
  force-sum.  Default (None) -> ``legal_actions`` (the force is applied to ALL
  legal actions, the original behavior).  Pass the SAMPLED-action one-hot to sum
  the force over ONLY the taken action, so the |A|-1 unsampled per-action
  regularizers don't swamp the single sampled action's return signal.  This
  changes only the LOSS mask; the adv_pi centering (baseline over all actions) and
  the renormalize denominator (number of valid steps) are unchanged.
  """
  if loss_action_mask is None:
    loss_action_mask = legal_actions
  assert isinstance(importance_sampling_correction, list)
  loss_pi_list = []
  for k, (logit_pi, pi, q_vr, is_c) in enumerate(
      zip(logit_list, policy_list, q_vr_list, importance_sampling_correction)):
    assert logit_pi.shape[0] == q_vr.shape[0]
    # loss policy
    adv_pi = q_vr - jnp.sum(pi * q_vr, axis=-1, keepdims=True)
    adv_pi = is_c * adv_pi  # importance sampling correction
    adv_pi = jnp.clip(adv_pi, min=-clip, max=clip)
    adv_pi = lax.stop_gradient(adv_pi)

    logits = logit_pi - jnp.mean(
        logit_pi * legal_actions, axis=-1, keepdims=True)

    threshold_center = jnp.zeros_like(logits)

    lower_threshold = None
    if teacher_centered_list is not None:
      # Per-action floor: allow pushing DOWN to the teacher's centered logit when
      # that is below -beta; otherwise keep -beta.  stop_gradient (it is a fixed
      # target from the collected state, not a function of the params).
      lower_threshold = lax.stop_gradient(
          jnp.minimum(teacher_centered_list[k], -threshold))

    nerd_loss = jnp.sum(
        loss_action_mask *
        apply_force_with_threshold(logits, adv_pi, threshold, threshold_center,
                                   lower_threshold),
        axis=-1)
    nerd_loss = -renormalize(nerd_loss, valid * (player_ids == k))
    loss_pi_list.append(nerd_loss)
  return sum(loss_pi_list)


@chex.dataclass(frozen=True)
class AdamConfig:
  """Adam optimizer related params."""
  b1: float = 0.0
  b2: float = 0.999
  eps: float = 10e-8


@chex.dataclass(frozen=True)
class NerdConfig:
  """Nerd related params."""
  beta: float = 2.0
  clip: float = 10_000


class StateRepresentation(str, enum.Enum):
  INFO_SET = "info_set"
  OBSERVATION = "observation"


@chex.dataclass(frozen=True)
class RNaDConfig:
  """Configuration parameters for the RNaDSolver."""
  # The game parameter string including its name and parameters.
  game_name: str
  # The games longer than this value are truncated. Must be strictly positive.
  trajectory_max: int = 10

  # The content of the EnvStep.obs tensor.
  state_representation: StateRepresentation = StateRepresentation.INFO_SET

  # Network configuration.
  policy_network_layers: Sequence[int] = (256, 256)

  # The batch size to use when learning/improving parameters.
  batch_size: int = 256
  # The learning rate for `params`.
  learning_rate: float = 0.00005
  # Optional linear LR warmup for the main-agent optimizer.  0 (default) keeps
  # the original constant-LR behavior exactly (byte-identical optimizer).  When
  # >0, the LR ramps linearly from 0 to `learning_rate` over this many learner
  # steps, then stays constant.  Transformers are LR-sensitive and benefit from
  # warmup; MLP/factored runs should leave this at 0.
  warmup_steps: int = 0
  # The config related to the ADAM optimizer used for updating `params`.
  adam: AdamConfig = AdamConfig()
  # All gradients values are clipped to [-clip_gradient, clip_gradient].
  clip_gradient: float = 10_000
  # The "speed" at which `params_target` is following `params`.
  target_network_avg: float = 0.001

  # RNaD algorithm configuration.
  # Entropy schedule configuration. See EntropySchedule class documentation.
  entropy_schedule_repeats: Sequence[int] = (1,)
  entropy_schedule_size: Sequence[int] = (20_000,)
  # The weight of the reward regularisation term in RNaD.
  eta_reward_transform: float = 0.2
  # Weight each OPPONENT's regularization term by 1/(num_players-1) so the
  # REGULARIZED game stays zero-sum at 3+ players. With the old fixed weight of
  # 1 the regularized returns summed to -(N-2)*sum(reg) -- exactly 0 at N=2, but
  # NOT at N>=3, where the transform manufactures total payoff whenever a player
  # deviates from its anchor, so the fixed point is not the Nash of a
  # regularized zero-sum game. At N=2 the weight is 1/1 = 1, i.e. this flag is
  # BYTE-IDENTICAL for every 2-player run. False reproduces pre-fix 3+ player
  # runs. See the _opp_reg_weight docstring for the derivation.
  zero_sum_reg_scaling: bool = True
  # Explicit override for the opponent regularization weight; None (default)
  # uses zero_sum_reg_scaling above. The interesting setting is 0.0: each
  # player's reward then carries ONLY ITS OWN regularizer,
  #     R_k = r_k + reg_k                  (instead of r_k + reg_k - w*sum_j reg_j)
  # which is the "natural" per-player KL-regularized game -- the same object
  # NashPG penalizes, and the same one every cross-player accounting question
  # disappears from. Trade-offs vs. the standard transform:
  #   - The regularized game is NO LONGER ZERO-SUM. Keeping it zero-sum is
  #     precisely why the opponent terms exist, so this steps outside R-NaD's
  #     stated theory (nothing in this implementation structurally requires it:
  #     v_trace is called per player with per-player masked value targets).
  #   - It removes a real and probably unintended incentive. The opponent terms
  #     look strategically inert in a normal-form view (KL_j does not depend on
  #     pi_k), but the penalties are visitation-weighted in the SEQUENTIAL game,
  #     so with a negative coefficient player k is rewarded for steering others
  #     into states where they have drifted from their anchor.
  # Still NOT equivalent to NashPG even at 0.0: the penalty is in the RETURN
  # (bootstrapped, so future KL cost shapes state visitation) rather than in the
  # loss (myopic per-state), it reaches the policy as a single-sample
  # REINFORCE-style estimate on the taken action rather than an exact
  # full-support analytic gradient, and there is still no entropy bonus
  # (see ppo_ent_coef).
  opp_reg_coef: Optional[float] = None
  nerd: NerdConfig = NerdConfig()
  c_vtrace: float = 1.0

  # Advantage estimator for the policy (NeuRD) update:
  #   "vtrace" (default) — the standard RNaD critic + custom v-trace baseline.
  #   "grpo"             — GRPO-style, critic-free, group-relative Monte-Carlo
  #                        advantage. Deals `batch_size / grpo_group_size`
  #                        distinct hands and replicates each `grpo_group_size`
  #                        times (a "group" shares the chance node / both hands
  #                        but samples actions independently); the per-trajectory
  #                        REGULARIZED return is baselined by the leave-one-out
  #                        group mean. Drops the value head / v-trace bootstrap
  #                        but KEEPS the eta reward-transform regularization.
  #                        Works with every network_type, including lpt (the
  #                        network application is delegated to _loss_rollouts).
  advantage_estimator: str = "vtrace"
  # Group size G for the "grpo" advantage estimator. `batch_size` must be an
  # integer multiple of this value.
  grpo_group_size: int = 8
  # v2: importance-weight the taken-action group advantage by 1/mu(a_t) (v-trace's
  # inv_mu), making the single-sample advantage an unbiased estimate of the
  # all-actions NeuRD Q-vector. NOTE: we do NOT divide the group advantage by its
  # std (standard GRPO does) -- the eta reward-transform sets an absolute scale the
  # NeuRD update balances against, so per-group std-normalization would distort it.
  grpo_importance_weight: bool = True
  # Cap on the GRPO importance weight inv_mu = 1/mu(a_t) (only used with
  # grpo_importance_weight). The 1e-6 mu-floor makes inv_mu reach ~1e6, so a
  # rarely-sampled deeply-suppressed action gets a huge adv*inv_mu kick (a
  # variance bomb). >0 clips inv_mu to this value; 0 (default) = uncapped =
  # byte-identical to the current behavior.
  grpo_inv_mu_cap: float = 0.0

  # Mask the NeuRD policy loss to ONLY the sampled action. When True, the
  # force-sum in get_loss_nerd uses the sampled-action one-hot instead of the
  # full legal-actions mask, so the |A|-1 unsampled per-action regularizers don't
  # swamp the single sampled action's return signal (esp. under multiple gradient
  # steps). The adv_pi centering (baseline over all actions) is unchanged. False
  # (default) = all-legal-actions = byte-identical to the current behavior.
  nerd_mask_unsampled: bool = False

  # Remove the GRPO immediate-reg double-count. In _loss_grpo the acting player's
  # own instantaneous reg at each step (reg_reward_own*sign_k) is summed into the
  # trajectory return reg_return_k AND added again per-action by eta_log_policy_k
  # (reg_reward_own*sign_k == eta_log_policy on the taken action). So the current
  # step's reg is counted twice on the sampled action, amplified by inv_mu -- a
  # divergence from the v-trace path, whose return carries only the FUTURE reg
  # (not the current step's) and whose per-action reg is eta_log_policy. When True,
  # subtract the current step's reg_reward_own*sign_k from the broadcast advantage
  # so the return contributes only the OTHER-step reg (continuation) and the
  # current step's reg is counted once (per-action). False (default) = byte-
  # identical to the current (double-counting) behavior.
  grpo_dedup_reg: bool = False

  # Policy update rule:
  #   "neurd" (default) — the unchanged NeuRD update (get_loss_nerd), fed by
  #                       whichever advantage_estimator is selected.
  #                       Byte-identical to before.
  #   "ppo"             — replace NeuRD with a PPO clipped-surrogate update on
  #                       the taken action: ratio = pi(a_t)/mu(a_t) against the
  #                       behavior policy that collected the batch. The R-NaD
  #                       reward transform stays ON either way (it lives in the
  #                       returns); only the update rule changes. Works with
  #                       BOTH advantage estimators:
  #                         "grpo"   — the leave-one-out group-relative
  #                                    REGULARIZED Monte-Carlo return (critic-
  #                                    free, high variance).
  #                         "vtrace" — the R-NaD critic's v-trace advantage of
  #                                    the taken action, eta_log_policy(a_t) +
  #                                    G_t - v(s_t), read out of v-trace's
  #                                    per-action learning output with its
  #                                    NeuRD-only 1/mu weight divided back out
  #                                    (see _loss_ppo_vtrace). The critic loss
  #                                    is included in the total.
  #                       The NeuRD-specific machinery does not apply under
  #                       either: no per-action eta_log_policy Q-vector (the
  #                       taken-action reg is in the scalar advantage, so there
  #                       is no double count and grpo_dedup_reg is ignored), no
  #                       1/mu inv_mu weight (the PPO ratio IS the importance
  #                       correction; grpo_importance_weight / grpo_inv_mu_cap
  #                       are ignored), and no logit trust region (nerd.beta /
  #                       nerd.clip / nerd_mask_unsampled / nerd_teacher_floor
  #                       are ignored).
  #   "nashpg"          — NashPG (https://arxiv.org/abs/2510.18183), "policy
  #                       gradient with iteratively refined regularization".
  #                       Drops the reward transform ENTIRELY (eta_reward_transform
  #                       is ignored / treated as 0) and moves the anchor into the
  #                       LOSS as an explicit divergence penalty toward the magnet
  #                       policy pi_reg:
  #                         loss = -PPO_clip(pi/mu, adv)
  #                                - nashpg_ent_coef * H(pi)
  #                                + nashpg_mag_coef * D(pi || pi_reg)
  #                                [+ critic loss, vtrace estimator only]
  #                       pi_reg is the SAME iteratively-refined anchor R-NaD
  #                       already maintains (the alpha-mixed params_prev /
  #                       params_prev_, rolled forward by the entropy schedule) —
  #                       that periodic roll-forward IS NashPG's refinement step.
  #                       Unlike "ppo", this works with BOTH advantage estimators:
  #                       "vtrace" (critic + v-trace advantage, the analogue of the
  #                       reference implementation's GAE+critic PPO) and "grpo"
  #                       (critic-free group-relative Monte-Carlo advantage of the
  #                       RAW return). The NeuRD machinery (nerd.*,
  #                       nerd_mask_unsampled, nerd_teacher_floor) and the GRPO
  #                       NeuRD-specific knobs (grpo_importance_weight,
  #                       grpo_inv_mu_cap, grpo_dedup_reg) do not apply.
  policy_update: str = "neurd"
  # PPO ratio clip epsilon (used when policy_update is "ppo" or "nashpg").
  ppo_clip_epsilon: float = 0.2
  # Standardize the advantage (masked mean/std over the acting player's valid
  # steps) before the PPO surrogate, as NashPG / reference PPO implementations
  # do. Only used when policy_update="ppo" (nashpg has its own
  # nashpg_normalize_advantage). False (default) = byte-identical to before.
  ppo_normalize_advantage: bool = False
  # Entropy bonus weight for policy_update="ppo" (nashpg has its own
  # nashpg_ent_coef). Note R-NaD's reward transform ALREADY carries an entropy
  # term (v_trace's eta_reg_entropy), unlike NashPG where the explicit bonus was
  # load-bearing — so 0.0 (default, off, byte-identical to before) is the
  # starting point here.
  ppo_ent_coef: float = 0.0
  # Entropy bonus weight for policy_update="neurd" — the NeuRD counterpart of
  # ppo_ent_coef, added so the 2x2 ablation (update rule x explicit entropy
  # bonus) has a NeuRD+entropy cell. Vanilla NeuRD has no entropy term at all:
  # its bound on determinism is the nerd.beta logit trust region plus the
  # all-legal-actions force, NOT a bonus (measured: NeuRD parks at 94% of the
  # beta ceiling, EXPERIMENTS.md §20). Applied exactly as in
  # _ppo_surrogate_loss: per player, over that player's own valid steps.
  # 0.0 (default, off) is byte-identical to before.
  neurd_ent_coef: float = 0.0

  # ---- NashPG knobs (only used when policy_update="nashpg") ----------------
  # Weight of the explicit magnet penalty D(pi || pi_reg) in the loss. This is
  # NashPG's regularization strength — it plays the role eta_reward_transform
  # plays in R-NaD, but applied to the policy directly instead of the reward.
  nashpg_mag_coef: float = 0.2
  # Divergence used for the magnet penalty:
  #   "kl" — KL(pi || pi_reg)             (the reference implementation's default)
  #   "l2" — 0.5 * ||pi - pi_reg||^2
  nashpg_mag_divergence: str = "kl"
  # Entropy bonus weight (the reference implementation's ent_coef).
  nashpg_ent_coef: float = 0.05
  # Standardize the advantage (masked mean/std over the acting player's valid
  # steps) before the PPO surrogate, as the reference implementation does.
  nashpg_normalize_advantage: bool = True

  # Per-action NeuRD trust-region floor from the heuristic teacher. When True the
  # NeuRD lower bound for each action becomes min(teacher_centered_logit, -beta)
  # instead of the scalar -beta, so actions the teacher nearly-zeroes can be pushed
  # below the -beta floor (draining the high-jump pile) while teacher-liked actions
  # keep -beta. The teacher (belief*exp(-lam*jump) bids + fixed challenge) is
  # computed per-state at COLLECTION time and stored on EnvStep.teacher_centered.
  # False (default) => byte-identical to the scalar-beta behavior.
  nerd_teacher_floor: bool = False
  # Teacher hyperparameters (only used when nerd_teacher_floor=True): the softmj1
  # teacher = aggressive anti-jump lam + fixed challenge probability.
  nerd_teacher_lam: float = 5.0
  nerd_teacher_challenge_prob: float = 0.1

  # Options related to fine tuning of the agent.
  finetune: FineTuning = FineTuning()

  # The seed that fully controls the randomness.
  seed: int = 42


@chex.dataclass(frozen=True)
class EnvStep:
  """Holds the tensor data representing the current game state."""
  # Indicates whether the state is a valid one or just a padding. Shape: [...]
  # The terminal state being the first one to be marked !valid.
  # All other tensors in EnvStep contain data, but only for valid timesteps.
  # Once !valid the data needs to be ignored, since it's a duplicate of
  # some other previous state.
  # The rewards is the only exception that contains reward values
  # in the terminal state, which is marked !valid.
  # TODO(author16): This is a confusion point and would need to be clarified.
  valid: chex.Array = ()  # pytype: disable=annotation-type-mismatch  # numpy-scalars
  # The single tensor representing the state observation. Shape: [..., ??]
  obs: chex.Array = ()  # pytype: disable=annotation-type-mismatch  # numpy-scalars
  # The legal actions mask for the current player. Shape: [..., A]
  legal: chex.Array = ()  # pytype: disable=annotation-type-mismatch  # numpy-scalars
  # The current player id as an int. Shape: [...]
  player_id: chex.Array = ()  # pytype: disable=annotation-type-mismatch  # numpy-scalars
  # The rewards of all the players. Shape: [..., P]
  rewards: chex.Array = ()  # pytype: disable=annotation-type-mismatch  # numpy-scalars
  # OPTIONAL heuristic-teacher mean-centered logit for the per-action NeuRD
  # trust-region floor (nerd_teacher_floor). Shape: [..., A]. Empty ``()`` (the
  # default) contributes NO pytree leaf, so when the feature is off every EnvStep
  # is byte-identical to before.
  teacher_centered: chex.Array = ()  # pytype: disable=annotation-type-mismatch  # numpy-scalars


@chex.dataclass(frozen=True)
class ActorStep:
  """The actor step tensor summary."""
  # The action (as one-hot) of the current player. Shape: [..., A]
  action_oh: chex.Array = ()  # pytype: disable=annotation-type-mismatch  # numpy-scalars
  # The policy of the current player. Shape: [..., A]
  policy: chex.Array = ()  # pytype: disable=annotation-type-mismatch  # numpy-scalars
  # The rewards of all the players. Shape: [..., P]
  # Note - these are rewards obtained *after* the actor step, and thus
  # these are the same as EnvStep.rewards visible before the *next* step.
  rewards: chex.Array = ()  # pytype: disable=annotation-type-mismatch  # numpy-scalars


@chex.dataclass(frozen=True)
class TimeStep:
  """The tensor data for one game transition (env_step, actor_step)."""
  env: EnvStep = EnvStep()
  actor: ActorStep = ActorStep()


Optimizer = Callable[[Params, Params], Params]  # (params, grads) -> params


def lr_scale_transform(learning_rate: float,
                       warmup_steps: int = 0) -> optax.GradientTransformation:
  """Negative-LR scaling for the main optimizer, with optional linear warmup.

  ``warmup_steps == 0`` (default) returns ``optax.scale(-learning_rate)`` --
  byte-identical to the original constant-LR optimizer.  ``warmup_steps > 0``
  ramps the LR linearly from 0 to ``learning_rate`` over ``warmup_steps`` learner
  steps (then holds constant) via an ``optax`` step-counted schedule.
  """
  if warmup_steps and warmup_steps > 0:
    schedule = optax.linear_schedule(
        init_value=0.0,
        end_value=learning_rate,
        transition_steps=warmup_steps,
    )
    return optax.scale_by_schedule(lambda count: -schedule(count))
  return optax.scale(-learning_rate)


def optax_optimizer(
    params: chex.ArrayTree,
    init_and_update: optax.GradientTransformation) -> Optimizer:
  """Creates a parameterized function that represents an optimizer."""
  init_fn, update_fn = init_and_update

  @chex.dataclass
  class OptaxOptimizer:
    """A jax-friendly representation of an optimizer state with the update."""
    state: chex.Array

    def __call__(self, params: Params, grads: Params) -> Params:
      updates, self.state = update_fn(grads, self.state)  # pytype: disable=annotation-type-mismatch  # numpy-scalars
      return optax.apply_updates(params, updates)

  return OptaxOptimizer(state=init_fn(params))


class RNaDNetwork(nn.Module):
  """MLP torso with a policy head and a value head."""
  policy_network_layers: Sequence[int]
  num_actions: int
  num_players: int = 1  # accepted for API compatibility; value head remains scalar
  # Policy TEMPERATURE, baked into the forward pass: pi = softmax(logit / T).
  # The tempered policy IS the policy -- it is what gets sampled, scored,
  # entropy-regularised, trained and evaluated -- so collection stays exactly
  # on-policy (the PPO ratio is 1 at collection) instead of the tempered
  # sampling being an uncorrected off-policy perturbation.
  #
  # NOTE this is a REPARAMETERISATION, not a constraint: the net can still
  # reach any sharpness by growing ||logit||, and T only divides. What it buys
  # is a flatter STARTING policy for a warm-started BR (whose loaded logits
  # arrive at the agent's trained scale), plus a 1/T damping of the logit
  # gradient. A durable entropy floor needs ent_coef > 0.
  temperature: float = 1.0

  @nn.compact
  def __call__(
      self, env_step: EnvStep
  ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    x = env_step.obs
    for size in self.policy_network_layers:
      x = nn.Dense(size)(x)
      x = nn.relu(x)
    logit = nn.Dense(self.num_actions)(x)
    v = nn.Dense(1)(x)
    if self.temperature != 1.0:
      logit = logit / self.temperature
    pi = _legal_policy(logit, env_step.legal)
    log_pi = legal_log_policy(logit, env_step.legal)
    return pi, v, log_pi, logit


class FactoredMLPNetwork(nn.Module):
  """MLP torso with a FACTORED (structured) policy head and a value head.

  Identical to `RNaDNetwork` except the policy logits are produced by a
  dot-product between a per-state query vector and a per-action *key* vector.
  The key is a small shared MLP applied to a static per-action feature table
  `phi` built from the game structure (challenge vs bid, bid count, bid
  number).  This shares parameters across related bids (c,n)/(c,n+1)/(c+1,n)
  instead of learning |A| independent logits, at the *same* cost/shape as the
  flat `Dense(num_actions)` head (trunk runs once; the head is one matmul).
  """
  policy_network_layers: Sequence[int]
  num_actions: int
  num_digits: int
  total_cards: int
  num_players: int = 1  # accepted for API compatibility; value head is scalar
  head_dim: int = 128
  # Count encoding used for phi col 1 (the per-action bid COUNT feature):
  #   'linear'   -- count / total_cards (the legacy normalized scalar; DEFAULT,
  #                 byte-identical to the original head).
  #   'tailprob' -- q(c) = log P(total_count >= c) for total_count ~
  #                 Binomial(total_cards, 1/num_digits) (POOL = total_cards =
  #                 num_players*hand_length, p = 1/num_digits).  This is the SAME
  #                 closed-form log-survival feature the transformer's
  #                 'perdigit_tailprob' head uses; it is PARAMETER-FREE in game
  #                 size (recomputed per config from the game shape), so the
  #                 action-count encoding is game-size-invariant -> transfers
  #                 across count / num_players.  The key-MLP over phi then acts as
  #                 the smooth transform (analogous to the transformer's
  #                 CountMLP(q(c))).
  count_encoding: str = 'linear'
  # Log-prob clamp for the 'tailprob' survival table (deep tail underflows to
  # -inf; clamped to a finite differentiable logit).  Matches the transformer
  # head's belief_clamp default (30.0) so q(c) is bit-for-bit identical there.
  count_clamp: float = 30.0

  def _build_phi(self) -> chex.Array:
    """Static per-action feature table `phi` of shape [num_actions, 2+D].

    Column layout (D = num_digits):
      col 0        : is_challenge (1 for action 0, else 0)
      col 1        : bid COUNT feature (0 for challenge):
                       * 'linear'   -> count / total_cards
                       * 'tailprob' -> q(count) = log P(total_count >= count)
      cols 2..2+D  : one-hot of (number-1) (all zeros for challenge)
    Mirrors `decode_bid`: for action a>=1, bid_id=a-1,
    count=bid_id//num_digits+1, number=bid_id%num_digits+1.
    """
    if self.count_encoding not in ('linear', 'tailprob'):
      raise ValueError(
          f"count_encoding must be 'linear' or 'tailprob', got "
          f"{self.count_encoding!r}")
    # 'tailprob': precompute q(c) = log P(total_count >= c) once from the game
    # config (POOL = total_cards, p = 1/num_digits).  qtab has length
    # total_cards+2 (c = 0 .. total_cards+1); qtab[0] == 0 exactly.  This REUSES
    # the transformer head's exact numerics (_log_survival_table), so factored +
    # transformer q(c) match bit-for-bit for the same config.
    qtab = None
    if self.count_encoding == 'tailprob':
      qtab = _log_survival_table(
          self.total_cards, self.num_digits, self.count_clamp)
    d_phi = 2 + self.num_digits
    phi = np.zeros((self.num_actions, d_phi), dtype=np.float32)
    # Action 0 == challenge.
    phi[0, 0] = 1.0
    for a in range(1, self.num_actions):
      bid_id = a - 1
      number = bid_id % self.num_digits + 1
      count = bid_id // self.num_digits + 1
      if qtab is None:
        phi[a, 1] = count / self.total_cards
      else:
        phi[a, 1] = qtab[count]
      phi[a, 2 + (number - 1)] = 1.0
    return jnp.asarray(phi)

  # See RNaDNetwork.temperature -- same semantics, same default.
  temperature: float = 1.0

  @nn.compact
  def __call__(
      self, env_step: EnvStep
  ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    x = env_step.obs
    for size in self.policy_network_layers:
      x = nn.Dense(size)(x)
      x = nn.relu(x)
    d = self.head_dim
    # Per-state query.
    q = nn.Dense(d)(x)  # [B, d]
    # Per-action key from the static feature table (computed once per forward).
    phi = self._build_phi()  # [num_actions, 2 + num_digits]
    k = nn.Dense(d)(phi)
    k = nn.relu(k)
    key = nn.Dense(d)(k)  # [num_actions, d]
    # Dot-product logits; 1/sqrt(d) scaling keeps the logit scale comparable to
    # the flat Dense head and avoids large-magnitude logits / instability.
    logit = jnp.matmul(q, key.T) / jnp.sqrt(jnp.asarray(d, dtype=q.dtype))
    v = nn.Dense(1)(x)
    if self.temperature != 1.0:
      logit = logit / self.temperature
    pi = _legal_policy(logit, env_step.legal)
    log_pi = legal_log_policy(logit, env_step.legal)
    return pi, v, log_pi, logit


class RNaDSolver(policy_lib.Policy):
  """Implements a solver for the R-NaD Algorithm.

  See https://arxiv.org/abs/2206.15378.

  Define all networks. Derive losses & learning steps. Initialize the game
  state and algorithmic variables.
  """

  def __init__(self, config: RNaDConfig):
    self.config = config

    # Learner and actor step counters.
    self.learner_steps = 0
    self.actor_steps = 0

    self.init()

  def init(self):
    """Initialize the network and losses."""
    # The random facilities for jax and numpy.
    self._rngkey = jax.random.PRNGKey(self.config.seed)
    self._np_rng = np.random.RandomState(self.config.seed)
    # TODO(author16): serialize both above to get the fully deterministic behaviour.

    # Create a game and an example of a state.
    self._game = pyspiel.load_game(self.config.game_name)
    self._ex_state = self._play_chance(self._game.new_initial_state())

    # The network.
    self.network = RNaDNetwork(
        policy_network_layers=self.config.policy_network_layers,
        num_actions=self._game.num_distinct_actions(),
    )

    # The machinery related to updating parameters/learner.
    self._entropy_schedule = EntropySchedule(
        sizes=self.config.entropy_schedule_size,
        repeats=self.config.entropy_schedule_repeats)
    self._loss_and_grad = jax.value_and_grad(self.loss, has_aux=False)

    # Create initial parameters.
    env_step = self._state_as_env_step(self._ex_state)
    key = self._next_rng_key()  # Make sure to use the same key for all.
    self.params = self.network.init(key, env_step)
    self.params_target = self.network.init(key, env_step)
    self.params_prev = self.network.init(key, env_step)
    self.params_prev_ = self.network.init(key, env_step)

    # Parameter optimizers.
    self.optimizer = optax_optimizer(
        self.params,
        optax.chain(
            optax.scale_by_adam(
                eps_root=0.0,
                **self.config.adam,
            ),
            lr_scale_transform(
                self.config.learning_rate, self.config.warmup_steps),
            optax.clip(self.config.clip_gradient)))
    self.optimizer_target = optax_optimizer(
        self.params_target, optax.sgd(self.config.target_network_avg))

  def _loss_rollouts(self, params: Params, params_target: Params,
                     params_prev: Params, params_prev_: Params, ts: TimeStep):
    """Apply the four networks over the [T, B] trajectory.

    Returns the seven per-timestep tensors the R-NaD loss consumes:
    ``(pi, v, log_pi, logit, v_target, log_pi_prev, log_pi_prev_)``.  This is the
    ONLY place the loss touches the network, so subclasses can swap in a
    different network-application strategy (e.g. the lpt causal batched forward)
    WITHOUT re-implementing the v-trace / neurd recurrence below.
    """
    rollout = jax.vmap(self.network.apply, (None, 0), 0)
    pi, v, log_pi, logit = rollout(params, ts.env)
    _, v_target, _, _ = rollout(params_target, ts.env)
    _, _, log_pi_prev, _ = rollout(params_prev, ts.env)
    _, _, log_pi_prev_, _ = rollout(params_prev_, ts.env)
    return pi, v, log_pi, logit, v_target, log_pi_prev, log_pi_prev_

  def loss(self, params: Params, params_target: Params, params_prev: Params,
           params_prev_: Params, ts: TimeStep, alpha: float,
           learner_steps: int, policy_only: bool = False) -> float:
    """Total loss. ``policy_only=True`` drops the critic term ``loss_v``.

    ``policy_only`` exists for the update-variance diagnostic
    (``src/update_variance.py``): ``loss_v`` is IDENTICAL for the neurd and ppo
    branches at the same params and data, so leaving it in adds the same
    gradient (and the same variance) to both and compresses any ratio between
    them toward 1. Dropping it isolates the thing being compared -- the policy
    update rule. Default False is byte-identical to before. Only the v-trace
    neurd/ppo branches honour it; the others raise rather than silently
    returning the full loss.
    """
    # GRPO branch: critic-free group-relative advantage. `advantage_estimator`
    # lives on the (static) config, so this is a compile-time branch and the
    # v-trace path below is left byte-for-byte identical as the control.
    policy_update = getattr(self.config, "policy_update", "neurd")
    if policy_update not in ("neurd", "ppo", "nashpg"):
      raise ValueError(
          f"unknown policy_update: {policy_update!r} "
          "(expected 'neurd', 'ppo' or 'nashpg')")
    # NashPG replaces both the NeuRD update and the reward-transform
    # regularization, and supplies its own advantage for either estimator.
    if policy_only and (policy_update == "nashpg"
                        or getattr(self.config, "advantage_estimator",
                                   "vtrace") == "grpo"):
      raise NotImplementedError(
          "policy_only is implemented only for the v-trace neurd/ppo branches; "
          f"got policy_update={policy_update!r} advantage_estimator="
          f"{getattr(self.config, 'advantage_estimator', 'vtrace')!r}")
    if policy_update == "nashpg":
      return self._loss_nashpg(params, params_target, params_prev, params_prev_,
                               ts, alpha, learner_steps)
    if getattr(self.config, "advantage_estimator", "vtrace") == "grpo":
      return self._loss_grpo(params, params_target, params_prev, params_prev_,
                             ts, alpha, learner_steps)

    (pi, v, log_pi, logit, v_target, log_pi_prev,
     log_pi_prev_) = self._loss_rollouts(
        params, params_target, params_prev, params_prev_, ts)

    policy_pprocessed = self.config.finetune(pi, ts.env.legal, learner_steps)
    # This line creates the reward transform log(pi(a|x)/pi_reg(a|x)).
    # For the stability reasons, reward changes smoothly between iterations.
    # The mixing between old and new reward transform is a convex combination
    # parametrised by alpha.
    log_policy_reg = log_pi - (alpha * log_pi_prev + (1 - alpha) * log_pi_prev_)

    v_target_list, has_played_list, v_trace_policy_target_list = [], [], []
    for player in range(self._game.num_players()):
      reward = ts.actor.rewards[:, :, player]  # [T, B, Player]
      v_target_, has_played, policy_target_ = v_trace(
          v_target,
          ts.env.valid,
          ts.env.player_id,
          ts.actor.policy,
          policy_pprocessed,
          log_policy_reg,
          _player_others(ts.env.player_id, ts.env.valid, player,
                         self._opp_reg_weight()),
          ts.actor.action_oh,
          reward,
          player,
          lambda_=1.0,
          c=self.config.c_vtrace,
          rho=np.inf,
          eta=self.config.eta_reward_transform,
          opp_reg_weight=self._opp_reg_weight())
      v_target_list.append(v_target_)
      has_played_list.append(has_played)
      v_trace_policy_target_list.append(policy_target_)
    loss_v = get_loss_v([v] * self._game.num_players(), v_target_list,
                        has_played_list)

    # PPO clipped surrogate fed by the v-trace critic, in place of NeuRD. The
    # R-NaD reward transform is untouched (it is inside the v-trace returns),
    # only the update rule changes.
    if policy_update == "ppo":
      loss_pi = self._loss_ppo_vtrace(
          ts, pi, log_pi, v_target, log_policy_reg,
          v_trace_policy_target_list)
      return loss_pi if policy_only else loss_v + loss_pi

    is_vector = jnp.expand_dims(jnp.ones_like(ts.env.valid), axis=-1)
    importance_sampling_correction = [is_vector] * self._game.num_players()
    # Uses v-trace to define q-values for Nerd
    teacher_centered_list = self._nerd_teacher_centered_list(ts)
    loss_nerd = get_loss_nerd(
        [logit] * self._game.num_players(), [pi] * self._game.num_players(),
        v_trace_policy_target_list,
        ts.env.valid,
        ts.env.player_id,
        ts.env.legal,
        importance_sampling_correction,
        clip=self.config.nerd.clip,
        threshold=self.config.nerd.beta,
        teacher_centered_list=teacher_centered_list,
        loss_action_mask=self._nerd_loss_action_mask(ts))
    # Optional explicit entropy bonus on the NeuRD update (neurd_ent_coef).
    # Vanilla NeuRD has none; this exists so the ablation can separate "update
    # rule" from "explicit entropy bonus". Same form as _ppo_surrogate_loss:
    # one renormalized term per player over that player's own valid steps, and
    # pi / log_pi are both 0 on illegal actions so the sum is over the legal
    # support with no 0 * -inf hazard.
    neurd_ent_coef = getattr(self.config, "neurd_ent_coef", 0.0)
    if neurd_ent_coef:
      entropy = -jnp.sum(pi * log_pi, axis=-1)                       # [T, B]
      for player in range(self._game.num_players()):
        mask_k = ts.env.valid * (ts.env.player_id == player)         # [T, B]
        loss_nerd = loss_nerd - neurd_ent_coef * renormalize(entropy, mask_k)
    return loss_nerd if policy_only else loss_v + loss_nerd  # pytype: disable=bad-return-type  # numpy-scalars

  def _opp_reg_weight(self) -> float:
    """Opponent regularization weight (see the module-level ``_opp_reg_weight``).

    ``config.opp_reg_coef`` overrides it outright when set; in particular 0.0
    drops the opponent terms entirely, so each player's reward carries only its
    OWN regularizer (see the config docstring).
    """
    override = getattr(self.config, "opp_reg_coef", None)
    if override is not None:
      return float(override)
    return _opp_reg_weight(
        self._game.num_players(),
        getattr(self.config, "zero_sum_reg_scaling", True))

  def _nerd_teacher_centered_list(self, ts: TimeStep):
    """Per-player teacher centered-logit list for the NeuRD floor, or None.

    Returns None (byte-identical scalar-beta behavior) unless
    ``config.nerd_teacher_floor`` is set, in which case the [T, B, A]
    ``ts.env.teacher_centered`` (populated at collection) is broadcast per player.
    """
    if not getattr(self.config, "nerd_teacher_floor", False):
      return None
    return [ts.env.teacher_centered] * self._game.num_players()

  def _nerd_loss_action_mask(self, ts: TimeStep):
    """Sampled-action one-hot for the NeuRD force-sum, or None.

    Returns None (byte-identical all-legal-actions behavior) unless
    ``config.nerd_mask_unsampled`` is set, in which case the [T, B, A] sampled
    one-hot ``ts.actor.action_oh`` masks the force to only the taken action. It
    is 0 on non-acting players' steps too, but those steps are already dropped by
    the per-player ``renormalize`` mask, so the single array works for all players.
    """
    if not getattr(self.config, "nerd_mask_unsampled", False):
      return None
    return ts.actor.action_oh

  def _ppo_surrogate_loss(self, ts: TimeStep, pi, log_pi, adv, mask):
    """PPO clipped surrogate (+ optional entropy bonus) on the taken action.

    ``adv`` is a [T, B] stop-gradiented advantage for ONE player and ``mask`` is
    that player's valid-step mask; the ratio is ``pi(a_t)/mu(a_t)`` against the
    behavior policy that collected the batch. Shared by both PPO advantage
    sources (``_loss_grpo``'s group advantage and ``_loss_ppo_vtrace``'s v-trace
    advantage) so the surrogate, the clip and the two tuning knobs
    (``ppo_normalize_advantage``, ``ppo_ent_coef``) behave identically in both.
    With those knobs at their defaults (off / 0.0) this is byte-identical to the
    original inline surrogate.
    """
    if getattr(self.config, "ppo_normalize_advantage", False):
      adv_mean = renormalize(adv, mask)
      adv_std = jnp.sqrt(renormalize(jnp.square(adv - adv_mean), mask))
      adv = lax.stop_gradient((adv - adv_mean) / (adv_std + 1e-8))

    actions_oh = ts.actor.action_oh                                  # [T, B, A]
    log_pi_taken = jnp.sum(actions_oh * log_pi, axis=-1)             # [T, B]
    mu_taken = jnp.sum(actions_oh * ts.actor.policy, axis=-1)        # [T, B]
    log_mu_taken = jnp.log(jnp.maximum(mu_taken, 1e-10))
    ratio = jnp.exp(log_pi_taken - lax.stop_gradient(log_mu_taken))  # [T, B]
    eps = getattr(self.config, "ppo_clip_epsilon", 0.2)
    surrogate = jnp.minimum(
        ratio * adv, jnp.clip(ratio, 1.0 - eps, 1.0 + eps) * adv)    # [T, B]
    loss = -renormalize(surrogate, mask)

    ent_coef = getattr(self.config, "ppo_ent_coef", 0.0)
    if ent_coef:
      # pi and log_pi are both 0 on illegal actions, so this sum is over the
      # legal support with no 0 * -inf hazard.
      entropy = -jnp.sum(pi * log_pi, axis=-1)                       # [T, B]
      loss = loss - ent_coef * renormalize(entropy, mask)
    return loss

  def _loss_ppo_vtrace(self, ts: TimeStep, pi, log_pi, v_target_net,
                       log_policy_reg, q_vr_list):
    """PPO clipped surrogate fed by the R-NaD v-trace critic.

    ``policy_update="ppo"`` with ``advantage_estimator="vtrace"``: keep the
    whole R-NaD machinery (reward transform inside the returns, target-network
    critic, v-trace bootstrap) and swap ONLY the NeuRD update for a PPO
    clipped surrogate on the taken action. This is the critic-based counterpart
    of the ``grpo`` PPO arm, whose Monte-Carlo group advantage is much higher
    variance.

    The advantage is read straight out of v-trace's per-action learning output
    ``q_vr`` (``v_trace_policy_target_list``), which on the acting player's own
    valid steps equals

        q_vr(a_t) = v(s_t) + eta_log_policy(a_t) + inv_mu * (G_t - v(s_t))

    with ``G_t`` the v-trace-bootstrapped regularized return, ``v`` the TARGET
    net's value and ``inv_mu = 1/mu(a_t)``. NeuRD wants that ``inv_mu`` (it makes
    the single sampled action an unbiased estimate of the all-actions Q-vector
    it centers), but a policy gradient does NOT — sampling ``a_t ~ mu`` already
    supplies that factor, and the PPO ratio is the importance correction. So we
    divide it back out (multiply by ``mu(a_t)``; ``inv_mu`` is exactly
    ``1/mu(a_t)`` on valid steps, no epsilon floor) and keep the per-action
    regularization term outside it:

        A_t = eta_log_policy(a_t) + (G_t - v(s_t))

    i.e. the regularized advantage of the taken action, with the immediate step's
    reg counted exactly once (v-trace's ``G_t`` carries only the continuation
    reg, the current step's is the ``eta_log_policy`` term) — the same accounting
    the NeuRD path uses.

    ``q_vr`` is 0 on the other player's and invalid steps, so ``adv`` is
    meaningless there; every use is masked by that player's own valid steps.
    """
    loss_list = []
    for player, q_vr in enumerate(q_vr_list):
      adv = self._ppo_vtrace_advantage(ts, v_target_net, log_policy_reg, q_vr)
      mask_k = ts.env.valid * (ts.env.player_id == player)           # [T, B]
      loss_list.append(self._ppo_surrogate_loss(ts, pi, log_pi, adv, mask_k))
    return sum(loss_list)

  def _ppo_vtrace_advantage(self, ts: TimeStep, v_target_net, log_policy_reg,
                            q_vr):
    """One player's taken-action v-trace advantage ``eta_log_policy + G - v``.

    See ``_loss_ppo_vtrace`` for the derivation. Valid only on that player's own
    valid steps (``q_vr`` is 0 elsewhere).
    """
    eta = self.config.eta_reward_transform
    actions_oh = ts.actor.action_oh                                  # [T, B, A]
    v_taken = jnp.squeeze(v_target_net, axis=-1)                     # [T, B]
    mu_taken = jnp.sum(actions_oh * ts.actor.policy, axis=-1)        # [T, B]
    q_taken = jnp.sum(actions_oh * q_vr, axis=-1)                    # [T, B]
    # eta_log_policy on the taken action. player_others is +1 on the acting
    # player's own steps -- the only steps this advantage is read on -- so the
    # sign is implicit.
    reg_taken = -eta * jnp.sum(actions_oh * log_policy_reg, axis=-1)  # [T, B]
    # * mu_taken undoes v-trace's inv_mu = 1/mu(a_t) on the return term only.
    return lax.stop_gradient(
        reg_taken + mu_taken * (q_taken - v_taken - reg_taken))      # [T, B]

  def _loss_grpo(self, params: Params, params_target: Params,
                 params_prev: Params, params_prev_: Params, ts: TimeStep,
                 alpha: float, learner_steps: int) -> float:
    """GRPO-style, critic-free, group-relative advantage NeuRD loss.

    Assumes ``ts`` was collected with GROUPED deals: trajectories
    ``i = b * G + g`` for group ``b`` and member ``g`` share the SAME dealt
    hands (chance node) but sampled actions independently. ``G`` is
    ``config.grpo_group_size`` and ``batch_size`` is ``num_groups * G``.

    The value head / v-trace bootstrap is dropped. Instead each trajectory's
    REGULARIZED Monte-Carlo return (terminal reward + the same eta reward
    transform the v-trace path uses, accumulated over the episode) is
    baselined by the leave-one-out mean of its group. That trajectory-level
    advantage is broadcast to every timestep and injected into the *unchanged*
    NeuRD update (``get_loss_nerd``) exactly where v-trace would put its
    return-minus-value signal — the value baseline cancels in NeuRD's policy
    centering, so the only change vs. v-trace is the advantage source.

    ``config.policy_update == "ppo"`` swaps the NeuRD update for a PPO/GRPO
    clipped surrogate on the taken action (same group advantage, ratio
    ``pi(a_t)/mu(a_t)`` vs. the collection-time behavior policy); see the
    config docstring for which NeuRD-specific knobs stop applying.

    Network application is DELEGATED to ``_loss_rollouts`` (the same hook the
    v-trace path uses), so this is architecture-agnostic: the mlp / factored /
    transformer path uses the base per-timestep vmap, while ``network_type ==
    'lpt'`` transparently uses its causal-readout override. The value-head
    outputs (``v`` / ``v_target``) returned by that hook are simply discarded.
    """
    (pi, _v_unused, log_pi, logit, _v_target_unused,
     log_pi_prev, log_pi_prev_) = self._loss_rollouts(
        params, params_target, params_prev, params_prev_, ts)

    # Same reward-transform anchor log(pi/pi_reg) used by the v-trace path.
    log_policy_reg = log_pi - (alpha * log_pi_prev + (1 - alpha) * log_pi_prev_)

    eta = self.config.eta_reward_transform
    G = self.config.grpo_group_size
    valid = ts.env.valid                     # [T, B]
    actions_oh = ts.actor.action_oh          # [T, B, A]
    B = valid.shape[1]
    num_groups = B // G

    # Per-step regularization reward for the ACTING player:
    #   -eta * log(pi(a_t)/pi_reg(a_t)).  Invalid steps are masked later.
    reg_reward_own = -eta * jnp.sum(actions_oh * log_policy_reg, axis=-1)  # [T,B]

    use_ppo = getattr(self.config, "policy_update", "neurd") == "ppo"

    q_vr_list = []
    loss_ppo_list = []
    for player in range(self._game.num_players()):
      reward_k = ts.actor.rewards[:, :, player]  # [T, B]
      # player_others: +1 on the current player's own valid steps, -1 on the
      # opponent's valid steps, 0 on invalid steps (already * valid).
      player_others = _player_others(ts.env.player_id, ts.env.valid, player,
                                     self._opp_reg_weight())
      sign_k = jnp.squeeze(player_others, axis=-1)  # [T, B]

      # Regularized Monte-Carlo return from player k's (zero-sum) perspective.
      # reward_k is the sparse terminal return (non-zero only at the single
      # transition-to-terminal step); * valid drops the terminal-repeat steps.
      # The regularization is the acting player's own pull, signed to k's view.
      reg_step_k = valid * reward_k + reg_reward_own * sign_k       # [T, B]
      reg_return_k = jnp.sum(reg_step_k, axis=0)                    # [B]

      # Leave-one-out group baseline over the shared-deal group.
      rr = reg_return_k.reshape(num_groups, G)
      group_sum = jnp.sum(rr, axis=1, keepdims=True)
      loo_mean = (group_sum - rr) / (G - 1)
      adv = (rr - loo_mean).reshape(B)                             # [B]
      adv = lax.stop_gradient(adv)

      # PPO/GRPO clipped surrogate on the taken action. The trajectory-level
      # regularized group advantage is broadcast to every step; the reg is
      # entirely inside the (stop-gradient) return — there is no per-action
      # eta vector, hence no double count and no grpo_dedup_reg — and the
      # pi/mu importance ratio replaces NeuRD's inv_mu weight. Gradient flows
      # only through log_pi(a_t) via the ratio.
      if use_ppo:
        adv_bt = jnp.broadcast_to(adv[None, :], valid.shape)          # [T, B]
        mask_k = valid * (ts.env.player_id == player)
        loss_ppo_list.append(
            self._ppo_surrogate_loss(ts, pi, log_pi, adv_bt, mask_k))
        continue

      # NeuRD q-values: the eta_log_policy regularization VECTOR (identical to
      # the v-trace `eta_log_policy` term) plus the scalar group advantage on
      # the taken action. In NeuRD's centering (q - sum(pi*q)) this reproduces
      # a regularized policy gradient with the group-relative baseline.
      eta_log_policy_k = -eta * log_policy_reg * player_others      # [T, B, A]
      # Per-step advantage, broadcast from the trajectory-level regularized return.
      adv_bt = jnp.broadcast_to(adv[None, :], valid.shape)          # [T, B]
      # Fix the immediate-reg double-count: reg_reward_own*sign_k (the acting
      # player's own reg at THIS step) is inside reg_return_k -> adv, and it is
      # ALSO added per-action below by eta_log_policy_k (they are equal on the
      # taken action). Subtract it here so the return carries only the OTHER-step
      # reg (continuation) and the current step's reg is counted once. stop-grad
      # to match adv (q_vr is stop-gradiented downstream in get_loss_nerd anyway).
      if getattr(self.config, "grpo_dedup_reg", False):
        adv_bt = adv_bt - lax.stop_gradient(reg_reward_own * sign_k)
      # v2: importance-weight the taken-action advantage by 1/mu(a_t) (matching
      # v-trace's inv_mu), so the single-sample group advantage is an unbiased
      # estimate of the all-actions NeuRD Q-vector. mu ~= current pi (on-policy),
      # stop-grad; valid-masked and eps-floored for stability. No std-division.
      if getattr(self.config, "grpo_importance_weight", True):
        mu_taken = jnp.sum(actions_oh * lax.stop_gradient(pi), axis=-1)  # [T, B]
        inv_mu = valid / jnp.maximum(mu_taken, 1e-6)                     # [T, B]
        # Optional cap on the importance weight: the 1e-6 mu-floor lets inv_mu
        # reach ~1e6, so a rarely-sampled (deeply-suppressed) action gets a huge
        # adv*inv_mu upward kick -- a variance bomb. grpo_inv_mu_cap>0 clips it
        # (a typical on-policy action has mu~1/|A_legal| -> inv_mu up to ~|A|, so
        # a cap of e.g. 20 touches only the extreme tail). 0 (default) = uncapped
        # = byte-identical to before.
        inv_mu_cap = getattr(self.config, "grpo_inv_mu_cap", 0.0)
        if inv_mu_cap and inv_mu_cap > 0.0:
          inv_mu = jnp.minimum(inv_mu, inv_mu_cap)
        adv_term = adv_bt * inv_mu                                       # [T, B]
      else:
        adv_term = adv_bt                                                # [T, B]
      q_vr_k = eta_log_policy_k + actions_oh * adv_term[..., None]  # [T, B, A]
      q_vr_list.append(q_vr_k)

    if use_ppo:
      return sum(loss_ppo_list)  # pytype: disable=bad-return-type  # numpy-scalars

    is_vector = jnp.expand_dims(jnp.ones_like(ts.env.valid), axis=-1)
    importance_sampling_correction = [is_vector] * self._game.num_players()
    teacher_centered_list = self._nerd_teacher_centered_list(ts)
    loss_nerd = get_loss_nerd(
        [logit] * self._game.num_players(), [pi] * self._game.num_players(),
        q_vr_list,
        ts.env.valid,
        ts.env.player_id,
        ts.env.legal,
        importance_sampling_correction,
        clip=self.config.nerd.clip,
        threshold=self.config.nerd.beta,
        teacher_centered_list=teacher_centered_list,
        loss_action_mask=self._nerd_loss_action_mask(ts))
    return loss_nerd  # pytype: disable=bad-return-type  # numpy-scalars

  def _loss_nashpg(self, params: Params, params_target: Params,
                   params_prev: Params, params_prev_: Params, ts: TimeStep,
                   alpha: float, learner_steps: int) -> float:
    """NashPG: policy gradient with iteratively refined regularization.

    https://arxiv.org/abs/2510.18183 (reference implementation:
    nash_policy_gradient_public/train/nash_pg.py + train/core/update_agent.py).

    Where R-NaD injects the anchor into the REWARD (the eta * log(pi/pi_reg)
    transform, which then has to be propagated by v-trace and consumed by
    NeuRD), NashPG leaves the reward as the raw game payoff and puts the anchor
    in the LOSS as an explicit divergence penalty:

        loss = -E[ min(rho*A, clip(rho, 1-eps, 1+eps)*A) ]   # PPO surrogate
               -  ent_coef * H(pi)                           # entropy bonus
               +  mag_coef * D(pi || pi_reg)                 # magnet penalty
               [+ critic loss]                               # vtrace estimator

    with rho = pi(a_t)/mu(a_t) and pi_reg the "magnet". The magnet here is the
    SAME iteratively-refined anchor R-NaD already maintains: the alpha-mixed
    params_prev / params_prev_, rolled forward to the current policy every
    entropy-schedule window. That periodic roll-forward is precisely NashPG's
    refinement step (`mag_agent = clone(agent)` at each outer update in the
    reference); alpha only makes the hand-over smooth instead of a jump.

    The advantage follows ``config.advantage_estimator``:
      "vtrace" — critic + v-trace advantage ``v_trace_target(s) - V(s)``, the
                 analogue of the reference implementation's GAE+critic PPO. The
                 critic loss is included in the returned total.
      "grpo"   — critic-free leave-one-out group-relative Monte-Carlo advantage
                 over shared-deal groups (see ``_loss_grpo``), computed on the
                 RAW return: NashPG applies no reward transform, so unlike
                 ``_loss_grpo`` the return carries no eta regularization.

    Because one gradient step is taken per collected batch, pi == mu at the
    update and the PPO ratio is exactly 1 (the clip is inert), so the surrogate
    reduces to a policy gradient with ``A``. The ratio/clip machinery is kept so
    the loss stays correct if the batch is ever reused off-policy.

    Network application is delegated to ``_loss_rollouts`` (the same hook the
    v-trace and GRPO paths use), so every architecture is supported, including
    ``network_type='lpt'`` with its causal-readout override.
    """
    (pi, v, log_pi, _logit, v_target_net, log_pi_prev,
     log_pi_prev_) = self._loss_rollouts(
        params, params_target, params_prev, params_prev_, ts)

    valid = ts.env.valid                            # [T, B]
    legal = ts.env.legal.astype(jnp.float32)        # [T, B, A]
    actions_oh = ts.actor.action_oh                 # [T, B, A]
    num_players = self._game.num_players()

    # The magnet pi_reg, mixed in PROBABILITY space so it stays a distribution
    # (R-NaD mixes in log space because it only ever needs a log-ratio, whose
    # normalization cancels). NOTE `legal_log_policy` returns 0 -- not -inf --
    # on illegal actions, so exp() of it is 1 there and MUST be re-masked.
    pi_reg = lax.stop_gradient(
        legal * (alpha * jnp.exp(log_pi_prev)
                 + (1.0 - alpha) * jnp.exp(log_pi_prev_)))     # [T, B, A]
    log_pi_reg = jnp.log(jnp.maximum(pi_reg, 1e-10))           # [T, B, A]

    # ---- Advantage --------------------------------------------------------
    loss_v = 0.0
    adv_list = []
    if getattr(self.config, "advantage_estimator", "vtrace") == "grpo":
      G = self.config.grpo_group_size
      B = valid.shape[1]
      num_groups = B // G
      for player in range(num_players):
        # RAW Monte-Carlo return: `rewards` is the sparse terminal payoff, and
        # NashPG adds no reward transform, so this is the plain game return
        # (contrast `_loss_grpo`, whose return also carries the eta reg).
        ret_k = jnp.sum(valid * ts.actor.rewards[:, :, player], axis=0)   # [B]
        rr = ret_k.reshape(num_groups, G)
        loo_mean = (jnp.sum(rr, axis=1, keepdims=True) - rr) / (G - 1)
        adv_k = lax.stop_gradient((rr - loo_mean).reshape(B))             # [B]
        adv_list.append(jnp.broadcast_to(adv_k[None, :], valid.shape))    # [T,B]
    else:
      policy_pprocessed = self.config.finetune(pi, ts.env.legal, learner_steps)
      # eta=0 => v_trace's eta_reg_entropy / eta_log_policy terms vanish, i.e.
      # the reward transform is switched off. log_policy_reg is passed for shape
      # compatibility only; it is multiplied by eta inside.
      log_policy_reg = log_pi - log_pi_reg
      v_target_list, has_played_list = [], []
      for player in range(num_players):
        v_target_, has_played, _ = v_trace(
            v_target_net,
            valid,
            ts.env.player_id,
            ts.actor.policy,
            policy_pprocessed,
            log_policy_reg,
            _player_others(ts.env.player_id, valid, player,
                           self._opp_reg_weight()),
            actions_oh,
            ts.actor.rewards[:, :, player],
            player,
            lambda_=1.0,
            c=self.config.c_vtrace,
            rho=np.inf,
            eta=0.0,
            opp_reg_weight=self._opp_reg_weight())
        v_target_list.append(v_target_)
        has_played_list.append(has_played)
        # A(s_t) = v_trace_target(s_t) - V(s_t), the standard v-trace/GAE(1)
        # advantage. Both terms use the TARGET net's value -- the same baseline
        # v_trace bootstraps from. It is 0 on the other player's / invalid steps,
        # which the per-player mask below drops anyway.
        adv_list.append(
            lax.stop_gradient(jnp.squeeze(v_target_ - v_target_net, axis=-1)))
      loss_v = get_loss_v([v] * num_players, v_target_list, has_played_list)

    # ---- Policy loss: PPO surrogate + entropy bonus + magnet penalty -------
    eps = getattr(self.config, "ppo_clip_epsilon", 0.2)
    ent_coef = getattr(self.config, "nashpg_ent_coef", 0.05)
    mag_coef = getattr(self.config, "nashpg_mag_coef", 0.2)
    divergence_type = getattr(self.config, "nashpg_mag_divergence", "kl")
    normalize_adv = getattr(self.config, "nashpg_normalize_advantage", True)

    log_pi_taken = jnp.sum(actions_oh * log_pi, axis=-1)                 # [T, B]
    mu_taken = jnp.sum(actions_oh * ts.actor.policy, axis=-1)            # [T, B]
    log_mu_taken = jnp.log(jnp.maximum(mu_taken, 1e-10))
    ratio = jnp.exp(log_pi_taken - lax.stop_gradient(log_mu_taken))      # [T, B]

    # pi and log_pi are both 0 on illegal actions, so these sums are over the
    # legal support with no 0 * -inf hazard.
    entropy = -jnp.sum(pi * log_pi, axis=-1)                             # [T, B]
    if divergence_type == "kl":
      divergence = jnp.sum(pi * (log_pi - log_pi_reg), axis=-1)          # [T, B]
    elif divergence_type == "l2":
      divergence = 0.5 * jnp.sum(jnp.square(pi - pi_reg), axis=-1)       # [T, B]
    else:
      raise ValueError(
          f"unknown nashpg_mag_divergence: {divergence_type!r} "
          "(expected 'kl' or 'l2')")

    loss_pi_list = []
    for player in range(num_players):
      mask_k = valid * (ts.env.player_id == player)                      # [T, B]
      adv = adv_list[player]
      if normalize_adv:
        adv_mean = renormalize(adv, mask_k)
        adv_std = jnp.sqrt(renormalize(jnp.square(adv - adv_mean), mask_k))
        adv = lax.stop_gradient((adv - adv_mean) / (adv_std + 1e-8))
      surrogate = jnp.minimum(
          ratio * adv, jnp.clip(ratio, 1.0 - eps, 1.0 + eps) * adv)      # [T, B]
      loss_pi_list.append(
          -renormalize(surrogate, mask_k)
          - ent_coef * renormalize(entropy, mask_k)
          + mag_coef * renormalize(divergence, mask_k))

    return loss_v + sum(loss_pi_list)  # pytype: disable=bad-return-type  # numpy-scalars

  @functools.partial(jax.jit, static_argnums=(0,), donate_argnums=(1, 5, 6))
  def update_parameters(
      self,
      params: Params,
      params_target: Params,
      params_prev: Params,
      params_prev_: Params,
      optimizer: Optimizer,
      optimizer_target: Optimizer,
      timestep: TimeStep,
      alpha: float,
      learner_steps: int,
      update_target_net: bool):
    """A jitted pure-functional part of the `step`."""
    loss_val, grad = self._loss_and_grad(params, params_target, params_prev,
                                         params_prev_, timestep, alpha,
                                         learner_steps)
    # Update `params`` using the computed gradient.
    params = optimizer(params, grad)
    # Update `params_target` towards `params`.
    params_target = optimizer_target(
        params_target, tree.tree_map(lambda a, b: a - b, params_target, params))

    # Rolls forward the prev and prev_ params if update_target_net is 1.
    # pyformat: disable
    params_prev, params_prev_ = jax.lax.cond(
        update_target_net,
        lambda: (params_target, params_prev),
        lambda: (params_prev, params_prev_))
    # pyformat: enable

    logs = {
        "loss": loss_val,
    }
    return (params, params_target, params_prev, params_prev_, optimizer,
            optimizer_target), logs

  def _serializable_state(self) -> dict:
    """Returns a dict of arrays and scalars suitable for flax.serialization."""
    return dict(
        params=self.params,
        params_target=self.params_target,
        params_prev=self.params_prev,
        params_prev_=self.params_prev_,
        optimizer=self.optimizer.state,  # pytype: disable=attribute-error  # always-use-return-annotations
        optimizer_target=self.optimizer_target.state,  # pytype: disable=attribute-error  # always-use-return-annotations
        rngkey=self._rngkey,
        learner_steps=self.learner_steps,
        actor_steps=self.actor_steps,
    )

  def state_bytes(self) -> bytes:
    """Serialize solver state to bytes via flax.serialization (msgpack)."""
    return flax.serialization.to_bytes(self._serializable_state())

  def restore_state(self, encoded: bytes) -> None:
    """Restore solver state from bytes produced by state_bytes()."""
    state = flax.serialization.from_bytes(self._serializable_state(), encoded)
    self.params = state["params"]
    self.params_target = state["params_target"]
    self.params_prev = state["params_prev"]
    self.params_prev_ = state["params_prev_"]
    self.optimizer.state = state["optimizer"]  # pytype: disable=attribute-error  # always-use-return-annotations
    self.optimizer_target.state = state["optimizer_target"]  # pytype: disable=attribute-error  # always-use-return-annotations
    self._rngkey = state["rngkey"]
    self.learner_steps = int(state["learner_steps"])
    self.actor_steps = int(state["actor_steps"])

  def step(self):
    """One step of the algorithm, that plays the game and improves params."""
    timestep = self.collect_batch_trajectory()
    alpha, update_target_net = self._entropy_schedule(self.learner_steps)
    (self.params, self.params_target, self.params_prev, self.params_prev_,
     self.optimizer, self.optimizer_target), logs = self.update_parameters(
         self.params, self.params_target, self.params_prev, self.params_prev_,
         self.optimizer, self.optimizer_target, timestep, alpha,
         self.learner_steps, update_target_net)
    self.learner_steps += 1
    logs.update({
        "actor_steps": self.actor_steps,
        "learner_steps": self.learner_steps,
    })
    return logs

  def _next_rng_key(self) -> chex.PRNGKey:
    """Get the next rng subkey from class rngkey.

    Must *not* be called from under a jitted function!

    Returns:
      A fresh rng_key.
    """
    self._rngkey, subkey = jax.random.split(self._rngkey)
    return subkey

  def _state_as_env_step(self, state: pyspiel.State) -> EnvStep:
    # A terminal state must be communicated to players, however since
    # it's a terminal state things like the state_representation or
    # the set of legal actions are meaningless and only needed
    # for the sake of creating well a defined trajectory tensor.
    # Therefore the code below:
    # - extracts the rewards
    # - if the state is terminal, uses a dummy other state for other fields.
    rewards = np.array(state.returns())

    valid = not state.is_terminal()
    if not valid:
      state = self._ex_state

    if self.config.state_representation == StateRepresentation.OBSERVATION:
      obs = state.observation_tensor()
    elif self.config.state_representation == StateRepresentation.INFO_SET:
      obs = state.information_state_tensor()
    else:
      raise ValueError(
          f"Invalid StateRepresentation: {self.config.state_representation}.")

    # TODO(author16): clarify the story around rewards and valid.
    return EnvStep(
        obs=np.array(obs),
        legal=np.array(state.legal_actions_mask(), dtype=np.int8),
        player_id=np.array(state.current_player()),
        valid=np.array(valid),
        rewards=rewards)

  def action_probabilities(self,
                           state: pyspiel.State,
                           player_id: Any = None):
    """Returns action probabilities dict for a single batch."""
    env_step = self._batch_of_states_as_env_step([state])
    probs = self._network_jit_apply_and_post_process(
        self.params_target, env_step)
    probs = jax.device_get(probs[0])  # Squeeze out the 1-element batch.
    return {
        action: probs[action]
        for action, valid in enumerate(jax.device_get(env_step.legal[0]))
        if valid
    }

  @functools.partial(jax.jit, static_argnums=(0,))
  def _network_jit_apply_and_post_process(
      self, params: Params, env_step: EnvStep) -> chex.Array:
    pi, _, _, _ = self.network.apply(params, env_step)
    pi = self.config.finetune.post_process_policy(pi, env_step.legal)
    return pi

  @functools.partial(jax.jit, static_argnums=(0,))
  def _network_jit_apply(self, params: Params, env_step: EnvStep) -> chex.Array:
    pi, _, _, _ = self.network.apply(params, env_step)
    return pi

  def actor_step(self, env_step: EnvStep):
    pi = self._network_jit_apply(self.params, env_step)
    pi = np.asarray(pi)
    # TODO(author18): is this policy normalization really needed?
    pi = pi / np.sum(pi, axis=-1, keepdims=True)

    action = np.apply_along_axis(
        lambda x: self._np_rng.choice(range(pi.shape[1]), p=x), axis=-1, arr=pi)
    # TODO(author16): reapply the legal actions mask to bullet-proof sampling.
    action_oh = np.zeros(pi.shape, dtype=pi.dtype)
    action_oh[range(pi.shape[0]), action] = 1.0

    actor_step = ActorStep(policy=pi, action_oh=action_oh, rewards=())  # pytype: disable=wrong-arg-types  # numpy-scalars

    return action, actor_step

  def collect_batch_trajectory(self) -> TimeStep:
    states = [
        self._play_chance(self._game.new_initial_state())
        for _ in range(self.config.batch_size)
    ]
    timesteps = []

    env_step = self._batch_of_states_as_env_step(states)
    for _ in range(self.config.trajectory_max):
      prev_env_step = env_step
      a, actor_step = self.actor_step(env_step)

      states = self._batch_of_states_apply_action(states, a)
      env_step = self._batch_of_states_as_env_step(states)
      timesteps.append(
          TimeStep(
              env=prev_env_step,
              actor=ActorStep(
                  action_oh=actor_step.action_oh,
                  policy=actor_step.policy,
                  rewards=env_step.rewards),
          ))
    # Concatenate all the timesteps together to form a single rollout [T, B, ..]
    return jax.tree_util.tree_map(lambda *xs: np.stack(xs, axis=0), *timesteps)

  def _batch_of_states_as_env_step(self,
                                   states: Sequence[pyspiel.State]) -> EnvStep:
    envs = [self._state_as_env_step(state) for state in states]
    return jax.tree_util.tree_map(lambda *e: np.stack(e, axis=0), *envs)

  def _batch_of_states_apply_action(
      self, states: Sequence[pyspiel.State],
      actions: chex.Array) -> Sequence[pyspiel.State]:
    """Apply a batch of `actions` to a parallel list of `states`."""
    for state, action in zip(states, list(actions)):
      if not state.is_terminal():
        self.actor_steps += 1
        state.apply_action(action)
        self._play_chance(state)
    return states

  def _play_chance(self, state: pyspiel.State) -> pyspiel.State:
    """Plays the chance nodes until we end up at another type of node.

    Args:
      state: to be updated until it does not correspond to a chance node.
    Returns:
      The same input state object, but updated. The state is returned
      only for convenience, to allow chaining function calls.
    """
    while state.is_chance_node():
      chance_outcome, chance_proba = zip(*state.chance_outcomes())
      action = self._np_rng.choice(chance_outcome, p=chance_proba)
      state.apply_action(action)
    return state
