#include "lp_tree.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace lp {

namespace {
constexpr int CHALLENGE = 0;

void check(bool cond, const std::string &msg) {
  if (!cond) throw std::runtime_error(msg);
}
}  // namespace

int Config::num_hands() const {
  int n = 1;
  for (int i = 0; i < hand_length; ++i) n *= num_digits;
  return n;
}

Traverser::Traverser(Config cfg) : cfg_(cfg) {
  check(cfg.num_digits >= 1 && cfg.num_digits <= 9,
        "num_digits must be in [1, 9] (digit 0 of the 10-digit deck is not supported)");
  check(cfg.hand_length >= 1, "hand_length must be >= 1");
  check(cfg.num_bids() <= 64, "at most 64 bids are supported");
  D_ = cfg.num_digits;
  H_ = cfg.hand_length;
  NB_ = cfg.num_bids();
  NA_ = cfg.num_actions();
  NH_ = cfg.num_hands();
  IS_ = cfg.infostate_size();

  count_.assign((size_t)NH_ * D_, 0);
  for (int h = 0; h < NH_; ++h)
    for (int j = 0; j < H_; ++j) ++count_[(size_t)h * D_ + (hand_digit(h, j) - 1)];

  build_tree_();
}

int Traverser::hand_digit(int hand, int slot) const {
  for (int j = 0; j < slot; ++j) hand /= D_;
  return hand % D_ + 1;
}

uint64_t Traverser::num_infosets(int player) const {
  uint64_t n = 0;
  for (int L = 0; L < num_levels(); ++L)
    if (level_player(L) == player) n += (uint64_t)nodes_[L].size() * NH_;
  return n;
}

// Mirrors LiarsPokerState._apply_action for the two-player game.
void Traverser::apply_(PublicState &s, int action, bool &terminal) const {
  const int p = s.player;
  terminal = false;
  if (action == CHALLENGE) {
    s.chal[p] |= uint64_t(1) << s.cur_bid;
    ++s.num_challenges;
    if ((!s.is_rebid && s.num_challenges == 2) ||
        (s.is_rebid && s.num_challenges == 1))
      terminal = true;
  } else {
    const int bid = action - 1;
    s.cur_bid = bid;
    s.is_rebid = (p == s.originator);
    s.originator = p;
    s.bids[p] |= uint64_t(1) << bid;
    s.num_challenges = 0;
  }
  s.player ^= 1;
}

// Mirrors LiarsPokerState._legal_actions.
void Traverser::legal_mask(const PublicState &s, uint8_t *buf) const {
  std::fill(buf, buf + NA_, 0);
  if (s.cur_bid >= 0) buf[CHALLENGE] = 1;
  const bool rebid_possible = !s.is_rebid && s.num_challenges == 1;
  if (s.player != s.originator || rebid_possible)
    for (int b = s.cur_bid + 1; b < NB_; ++b) buf[b + 1] = 1;
}

PublicState Traverser::replay(int level, uint32_t idx) const {
  std::vector<uint8_t> acts(level);
  for (int L = level; L > 0; --L) {
    acts[L - 1] = nodes_[L][idx].action;
    idx = nodes_[L][idx].parent;
  }
  PublicState s;
  bool terminal;
  for (int d = 0; d < level; ++d) apply_(s, acts[d], terminal);
  return s;
}

void Traverser::build_tree_() {
  nodes_.clear();
  nodes_.push_back({NodeRec{0, 0}});
  child_begin_.clear();
  term_begin_.clear();
  terms_.clear();
  terms_.push_back({});
  num_decision_nodes_ = 1;
  num_terminals_ = 0;

  for (int L = 0;; ++L) {
    const uint32_t N = nodes_[L].size();
    std::vector<uint32_t> cb(N + 1, 0), tb(N + 1, 0);
#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < N; ++i) {
      const PublicState s = replay(L, i);
      uint8_t legal[65];
      legal_mask(s, legal);
      uint32_t nd = 0, nt = 0;
      for (int a = 0; a < NA_; ++a) {
        if (!legal[a]) continue;
        PublicState ss = s;
        bool terminal;
        apply_(ss, a, terminal);
        if (terminal)
          ++nt;
        else
          ++nd;
      }
      cb[i + 1] = nd;
      tb[i + 1] = nt;
    }
    for (uint32_t i = 0; i < N; ++i) {
      cb[i + 1] += cb[i];
      tb[i + 1] += tb[i];
    }
    const uint32_t nd_total = cb[N], nt_total = tb[N];
    check(nd_total < std::numeric_limits<uint32_t>::max(), "level too large");
    std::vector<NodeRec> next(nd_total);
    std::vector<uint32_t> next_terms(nt_total);
#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < N; ++i) {
      const PublicState s = replay(L, i);
      uint8_t legal[65];
      legal_mask(s, legal);
      uint32_t od = cb[i], ot = tb[i];
      for (int a = 0; a < NA_; ++a) {
        if (!legal[a]) continue;
        PublicState ss = s;
        bool terminal;
        apply_(ss, a, terminal);
        if (terminal)
          next_terms[ot++] = i;
        else
          next[od++] = NodeRec{i, (uint8_t)a};
      }
    }
    child_begin_.push_back(std::move(cb));
    term_begin_.push_back(std::move(tb));
    num_decision_nodes_ += nd_total;
    num_terminals_ += nt_total;
    terms_.push_back(std::move(next_terms));
    if (nd_total == 0) break;
    nodes_.push_back(std::move(next));
  }
  term_buckets_.assign(terms_.size(), {});
}

void Traverser::infostate_tensor(const PublicState &s, int hand,
                                 uint8_t *buf) const {
  std::fill(buf, buf + IS_, 0);
  buf[s.player] = 1;
  for (int j = 0; j < H_; ++j) buf[2 + j] = hand_digit(hand, j);
  buf[2 + H_] = s.is_rebid;
  buf[3 + H_] = 0;  // counts_state: never terminal at a decision node
  uint8_t *bh = buf + 4 + H_;
  uint8_t *ch = bh + 2 * NB_;
  for (int p = 0; p < 2; ++p)
    for (int b = 0; b < NB_; ++b) {
      bh[2 * b + p] = (s.bids[p] >> b) & 1;
      ch[2 * b + p] = (s.chal[p] >> b) & 1;
    }
}

std::string Traverser::infostate_string(int level, uint32_t idx,
                                        int hand) const {
  const PublicState s = replay(level, idx);
  std::string out = "p" + std::to_string(s.player) + " hand:[";
  for (int j = 0; j < H_; ++j) {
    if (j) out += ", ";
    out += std::to_string(hand_digit(hand, j));
  }
  out += "] rebid:[" + std::to_string((int)s.is_rebid) + "] counts:[0]";
  for (int b = 0; b < NB_; ++b)
    if (((s.bids[0] | s.bids[1]) >> b) & 1) out += " b:" + std::to_string(b) + ".";
  for (int b = 0; b < NB_; ++b)
    if (((s.chal[0] | s.chal[1]) >> b) & 1) out += " c:" + std::to_string(b) + ".";
  return out;
}

// ---------------------------------------------------------------------------
// Streaming top-down pass
// ---------------------------------------------------------------------------

void Traverser::begin(uint32_t chunk_nodes) {
  check(chunk_nodes > 0, "chunk_nodes must be positive");
  chunk_nodes_ = chunk_nodes;
  pass_active_ = true;
  pass_done_ = false;
  chunk_pending_ = false;
  ev0_ = 0;
  cur_chunk_ = Chunk{0, 0, 0, 0};
  reach_cur_.assign((size_t)2 * NH_, 1.0 / NH_);  // root: chance folded in
  reach_next_.assign((size_t)nodes_[1].size() * 2 * NH_, 0.0);
  for (auto &tb : term_buckets_) tb.clear();
  term_buckets_[1].assign((size_t)terms_[1].size() * 2 * (H_ + 1), 0.0);
  lose_mass_.assign(num_levels(), std::vector<Real>());
}

void Traverser::advance_level_() {
  const int L = cur_chunk_.level + 1;
  cur_chunk_ = Chunk{L, 0, 0, level_player(L)};
  reach_cur_.swap(reach_next_);
  if (L + 1 < num_levels()) {
    reach_next_.assign((size_t)nodes_[L + 1].size() * 2 * NH_, 0.0);
  } else {
    std::vector<Real>().swap(reach_next_);
  }
  if (L + 1 < (int)terms_.size())
    term_buckets_[L + 1].assign((size_t)terms_[L + 1].size() * 2 * (H_ + 1), 0.0);
}

bool Traverser::next_chunk(Chunk &chunk, std::vector<uint8_t> &infostates,
                           std::vector<uint8_t> &legal) {
  check(pass_active_, "call begin() first");
  check(!chunk_pending_, "push_probs() must be called before the next chunk");
  if (pass_done_) return false;
  // advance to the next non-empty range
  while (cur_chunk_.end >= nodes_[cur_chunk_.level].size()) {
    if (cur_chunk_.level + 1 >= num_levels()) {
      pass_done_ = true;
      std::vector<Real>().swap(reach_cur_);
      return false;
    }
    advance_level_();
  }
  const int L = cur_chunk_.level;
  cur_chunk_.begin = cur_chunk_.end;
  cur_chunk_.end = std::min<uint64_t>((uint64_t)cur_chunk_.begin + chunk_nodes_,
                                      nodes_[L].size());
  cur_chunk_.player = level_player(L);
  const uint32_t n = cur_chunk_.end - cur_chunk_.begin;
  infostates.resize((size_t)n * NH_ * IS_);
  legal.resize((size_t)n * NH_ * NA_);
#pragma omp parallel for schedule(static)
  for (uint32_t k = 0; k < n; ++k) {
    const PublicState s = replay(L, cur_chunk_.begin + k);
    uint8_t *lg = legal.data() + (size_t)k * NH_ * NA_;
    legal_mask(s, lg);
    for (int h = 1; h < NH_; ++h) std::memcpy(lg + (size_t)h * NA_, lg, NA_);
    for (int h = 0; h < NH_; ++h)
      infostate_tensor(s, h, infostates.data() + ((size_t)k * NH_ + h) * IS_);
  }
  chunk = cur_chunk_;
  chunk_pending_ = true;
  return true;
}

void Traverser::push_probs(const Real *probs, size_t stride, const Real *lose,
                           size_t lose_stride) {
  check(chunk_pending_, "no chunk pending");
  check(stride >= (size_t)NA_, "row stride must be at least num_actions");
  const int L = cur_chunk_.level;
  const int p = cur_chunk_.player;
  const uint32_t n = cur_chunk_.end - cur_chunk_.begin;
  const int NHd = NH_, NAd = NA_, Hp1 = H_ + 1;
  if (lose && lose_mass_[L].empty()) lose_mass_[L].assign(nodes_[L].size(), 0.0);
  Real *lose_mass = lose ? lose_mass_[L].data() : nullptr;
  Real ev0 = 0;
  bool bad_probs = false;
#pragma omp parallel for schedule(dynamic, 64) reduction(+ : ev0) reduction(|| : bad_probs)
  for (uint32_t k = 0; k < n; ++k) {
    const uint32_t i = cur_chunk_.begin + k;
    const PublicState s = replay(L, i);
    const Real *pr = probs + (size_t)k * NHd * stride;
    const Real *lo = lose ? lose + (size_t)k * NHd * lose_stride : nullptr;
    const Real *r = reach_cur_.data() + (size_t)i * 2 * NHd;
    uint8_t legal[65];
    legal_mask(s, legal);
    // validate: probabilities on legal actions (+ the forfeit mass) sum to one
    for (int h = 0; h < NHd; ++h) {
      Real sum = lo ? lo[(size_t)h * lose_stride] : 0.0;
      for (int a = 0; a < NAd; ++a)
        if (legal[a]) sum += pr[(size_t)h * stride + a];
      if (std::abs(sum - 1.0) > 1e-4 || !std::isfinite(sum)) bad_probs = true;
    }
    if (lo) {
      // Forfeits end the game at once with a hand-independent payoff (-1 for
      // the acting player), so their contribution factorises into the acting
      // player's forfeit-weighted reach times the opponent's total reach.
      Real m = 0, ropp = 0;
      for (int h = 0; h < NHd; ++h) {
        m += r[(size_t)p * NHd + h] * lo[(size_t)h * lose_stride];
        ropp += r[(size_t)(1 - p) * NHd + h];
      }
      lose_mass[i] = m;
      ev0 += m * ropp * (p == 0 ? -1.0 : 1.0);
    }
    uint32_t od = child_begin_[L][i], ot = term_begin_[L][i];
    for (int a = 0; a < NAd; ++a) {
      if (!legal[a]) continue;
      PublicState ss = s;
      bool terminal;
      apply_(ss, a, terminal);
      if (terminal) {
        // bucket sums of both players' reach (acting player's reach includes
        // its challenge probability), indexed by #cards of the bid number
        const int number = bid_number_(s.cur_bid), count = bid_count_(s.cur_bid);
        Real *B = term_buckets_[L + 1].data() + (size_t)(ot++) * 2 * Hp1;
        for (int q = 0; q < 2; ++q)
          for (int h = 0; h < NHd; ++h) {
            const Real rq = r[(size_t)q * NHd + h] *
                            (q == p ? pr[(size_t)h * stride + a] : 1.0);
            B[q * Hp1 + count_[(size_t)h * D_ + number - 1]] += rq;
          }
        for (int k0 = 0; k0 <= H_; ++k0)
          for (int k1 = 0; k1 <= H_; ++k1)
            ev0 += B[k0] * B[Hp1 + k1] *
                   terminal_utility_(0, s.originator, k0 + k1 >= count);
      } else {
        Real *rn = reach_next_.data() + (size_t)(od++) * 2 * NHd;
        for (int h = 0; h < NHd; ++h) {
          rn[(size_t)p * NHd + h] = r[(size_t)p * NHd + h] * pr[(size_t)h * stride + a];
          rn[(size_t)(1 - p) * NHd + h] = r[(size_t)(1 - p) * NHd + h];
        }
      }
    }
  }
  check(!bad_probs, "probabilities over legal actions (plus the lose column, if any) must sum to 1");
  ev0_ += ev0;
  chunk_pending_ = false;
}

// ---------------------------------------------------------------------------
// Bottom-up best-response pass
// ---------------------------------------------------------------------------

Result Traverser::finish(bool store_br) {
  check(pass_active_ && pass_done_ && !chunk_pending_,
        "finish() requires a completed top-down pass");
  const int NHd = NH_, Hp1 = H_ + 1;
  br_action_.clear();
  if (store_br) br_action_.resize(num_levels());
  // v[level][i][br_player][h]
  std::vector<Real> v_next, v_cur;
  for (int L = num_levels() - 1; L >= 0; --L) {
    const uint32_t N = nodes_[L].size();
    v_cur.assign((size_t)N * 2 * NHd, 0.0);
    if (store_br) br_action_[L].assign((size_t)N * NHd, 0);
#pragma omp parallel for schedule(dynamic, 64)
    for (uint32_t i = 0; i < N; ++i) {
      const PublicState s = replay(L, i);
      const int p = s.player;
      Real *v = v_cur.data() + (size_t)i * 2 * NHd;
      uint8_t *br = store_br ? br_action_[L].data() + (size_t)i * NHd : nullptr;
      // When the opponent of the best responder acts here, it forfeits with
      // (reach-weighted) probability lose_mass_[L][i], paying +1 to the best
      // responder whatever the latter's hand. A forfeit by the best responder
      // itself is weakly dominated by every real action (all payoffs are
      // >= -1), so it never enters the max below.
      const Real forfeit = lose_mass_[L].empty() ? 0.0 : lose_mass_[L][i];
      for (int q = 0; q < 2; ++q)
        for (int h = 0; h < NHd; ++h)
          v[q * NHd + h] = (q == p) ? -std::numeric_limits<Real>::infinity() : forfeit;
      int cur_action = 0;
      auto combine = [&](int q, int h, Real val) {
        Real &dst = v[q * NHd + h];
        if (q == p) {
          if (val > dst) {
            dst = val;
            if (br) br[h] = (uint8_t)cur_action;
          }
        } else {
          dst += val;
        }
      };
      uint8_t legal[65];
      legal_mask(s, legal);
      uint32_t od = child_begin_[L][i], ot = term_begin_[L][i];
      for (int a = 0; a < NA_; ++a) {
        if (!legal[a]) continue;
        cur_action = a;
        PublicState ss = s;
        bool terminal;
        apply_(ss, a, terminal);
        if (terminal) {
          const int number = bid_number_(s.cur_bid), count = bid_count_(s.cur_bid);
          const Real *B = term_buckets_[L + 1].data() + (size_t)(ot++) * 2 * Hp1;
          for (int q = 0; q < 2; ++q) {
            const Real *Bopp = B + (1 - q) * Hp1;
            for (int h = 0; h < NHd; ++h) {
              const int mine = count_[(size_t)h * D_ + number - 1];
              Real val = 0;
              for (int k = 0; k <= H_; ++k)
                val += Bopp[k] * terminal_utility_(q, s.originator, k + mine >= count);
              combine(q, h, val);
            }
          }
        } else {
          const Real *vc = v_next.data() + (size_t)(od++) * 2 * NHd;
          for (int q = 0; q < 2; ++q)
            for (int h = 0; h < NHd; ++h) combine(q, h, vc[q * NHd + h]);
        }
      }
    }
    v_next.swap(v_cur);
  }
  Result out;
  out.ev0 = ev0_;
  for (int q = 0; q < 2; ++q) {
    Real s = 0;
    for (int h = 0; h < NHd; ++h) s += v_next[(size_t)q * NHd + h];
    // v does not include the best responder's own chance probability
    out.br_value[q] = s / NHd;
  }
  out.expl[0] = out.ev0 + out.br_value[1];
  out.expl[1] = -out.ev0 + out.br_value[0];
  pass_active_ = false;
  return out;
}

int64_t Traverser::child_index(int level, uint32_t idx, int action) const {
  check(level >= 0 && level < num_levels() && idx < nodes_[level].size(), "invalid node");
  const PublicState s = replay(level, idx);
  uint8_t legal[65];
  legal_mask(s, legal);
  if (action < 0 || action >= NA_ || !legal[action]) return -1;
  uint32_t od = child_begin_[level][idx];
  for (int a = 0; a < NA_; ++a) {
    if (!legal[a]) continue;
    PublicState ss = s;
    bool terminal;
    apply_(ss, a, terminal);
    if (a == action) return terminal ? -1 : (int64_t)od;
    if (!terminal) ++od;
  }
  return -1;
}

int Traverser::br_action(int level, uint32_t idx, int hand) const {
  check(has_br_policy(), "call finish(store_br=True) first");
  check(level >= 0 && level < num_levels() && idx < nodes_[level].size() &&
            hand >= 0 && hand < NH_,
        "invalid (node, hand)");
  return br_action_[level][(size_t)idx * NH_ + hand];
}

}  // namespace lp
