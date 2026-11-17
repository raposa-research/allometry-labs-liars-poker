#pragma once

// Exact exploitability for 2-player Liar's Poker (OpenSpiel `python_liars_poker`
// rules, including the single rebid after a challenge).
//
// The game has public actions and chance only at the root (the deal), so the
// public tree is enumerated once, level by level (a level = number of actions
// played; the player to act is level % 2). Policies are streamed through the
// tree in chunks of public nodes: for every chunk the caller receives the
// OpenSpiel information-state tensors and legal-action masks of all
// (node, hand) pairs, and pushes back the action probabilities. Reach
// probabilities for both players are propagated top-down and, at terminals,
// collapsed into per-player "bucket sums" indexed by how many cards of the
// challenged number the player holds -- the only thing the payoff depends on.
// A bottom-up pass (no policy needed) then yields the best-response values.

#include <cstdint>
#include <string>
#include <vector>

namespace lp {

using Real = double;

struct Config {
  int num_digits;   // D: digits are 1..D
  int hand_length;  // H: cards per player
  int num_bids() const { return 2 * num_digits * hand_length; }  // NB
  int num_actions() const { return num_bids() + 1; }             // NA
  int num_hands() const;                                         // NH = D^H
  // player one-hot (2) + hand (H) + rebid (1) + counts (1) + bid_history
  // (NB x 2) + challenge_history (NB x 2)
  int infostate_size() const { return 4 + hand_length + 4 * num_bids(); }
};

// Public state (everything but the hands), replayed from the action history.
struct PublicState {
  uint64_t bids[2] = {0, 0};  // bit b set <=> player p made bid b
  uint64_t chal[2] = {0, 0};  // bit b set <=> player p challenged bid b
  int cur_bid = -1;           // last bid id, -1 if none
  int originator = -1;        // player who made the last bid
  int num_challenges = 0;     // challenges since the last bid
  bool is_rebid = false;      // last bid was a rebid by the originator
  int player = 0;             // player to act
};

struct NodeRec {
  uint32_t parent;  // index into the previous level
  uint8_t action;   // action that led here (0 = challenge, b+1 = bid b)
};

struct Chunk {
  int level;
  uint32_t begin, end;  // node range within the level
  int player;
};

struct Result {
  Real ev0;            // expected utility of player 0 under the profile
  Real br_value[2];    // br_value[p]: value (in p's utility) of p's best
                       // response to the opponent's strategy
  Real expl[2];        // expl[p]: exploitability of player p's strategy
                       // (eas convention: expl0 = ev0 + br_value[1], expl1 =
                       // -ev0 + br_value[0])
};

class Traverser {
 public:
  explicit Traverser(Config cfg);

  const Config &config() const { return cfg_; }
  int num_levels() const { return (int)nodes_.size(); }
  uint32_t level_size(int level) const { return nodes_[level].size(); }
  uint64_t num_decision_nodes() const { return num_decision_nodes_; }
  uint64_t num_terminals() const { return num_terminals_; }
  uint64_t num_infosets(int player) const;
  static int level_player(int level) { return level % 2; }

  // --- introspection -----------------------------------------------------
  PublicState replay(int level, uint32_t idx) const;
  // OpenSpiel `information_state_string` of (node, hand)
  std::string infostate_string(int level, uint32_t idx, int hand) const;
  // Fills `buf` (infostate_size()) with the OpenSpiel information_state_tensor.
  void infostate_tensor(const PublicState &s, int hand, uint8_t *buf) const;
  // Fills `buf` (num_actions()) with the legal action mask.
  void legal_mask(const PublicState &s, uint8_t *buf) const;
  // digit (1..D) in slot j (0..H-1) of hand index h
  int hand_digit(int hand, int slot) const;

  // --- streaming evaluation ---------------------------------------------
  // Starts a new top-down pass. `chunk_nodes` = public nodes per chunk.
  void begin(uint32_t chunk_nodes);
  // Returns false when the top-down pass is complete. Otherwise fills
  // `chunk`, `infostates` ((end-begin)*NH x IS, row-major) and `legal`
  // ((end-begin)*NH x NA).
  bool next_chunk(Chunk &chunk, std::vector<uint8_t> &infostates,
                  std::vector<uint8_t> &legal);
  // Consumes the probabilities for the chunk returned by the last
  // `next_chunk`: row (node, hand) starts at probs + row * stride and holds NA
  // entries. `lose` (optional, one entry per row at lose + row * lose_stride)
  // is the probability mass the acting player puts on "dummy" actions that
  // end the game immediately with the acting player losing (the sibling
  // repo's dummy_mode="lose" action-space probe). All such actions are
  // interchangeable, so only their total mass matters: the per-node cost is
  // independent of how many there are. Rows must satisfy
  // sum(probs over legal) + lose == 1.
  void push_probs(const Real *probs, size_t stride, const Real *lose = nullptr,
                  size_t lose_stride = 1);
  // Runs the bottom-up best-response pass; requires the top-down pass to be
  // complete. With `store_br` the best-response action of every (node, hand)
  // of the acting player is kept (one byte per infoset) for `br_action`.
  Result finish(bool store_br = false);

  // --- navigation / best-response policy ---------------------------------
  // Index (in level+1) of the child reached by `action`; -1 if the action is
  // illegal or leads to a terminal.
  int64_t child_index(int level, uint32_t idx, int action) const;
  // Best-response action at (node, hand) for the player acting there
  // (requires finish(store_br=true)).
  int br_action(int level, uint32_t idx, int hand) const;
  bool has_br_policy() const { return !br_action_.empty(); }

 private:
  Config cfg_;
  int NB_, NA_, NH_, IS_, D_, H_;
  std::vector<std::vector<NodeRec>> nodes_;      // nodes_[level]
  std::vector<std::vector<uint32_t>> child_begin_;  // [level][i] -> offset in nodes_[level+1]
  std::vector<std::vector<uint32_t>> term_begin_;   // [level][i] -> offset in terms_[level+1]
  std::vector<std::vector<uint32_t>> terms_;     // terms_[level] = parent index (level-1)
  std::vector<std::vector<Real>> term_buckets_;  // [level][t * 2 * (H+1)]
  std::vector<int> count_;  // count_[h * D + (n-1)] = #cards of digit n in hand h
  std::vector<std::vector<uint8_t>> br_action_;  // [level][i * NH + h]
  uint64_t num_decision_nodes_ = 0, num_terminals_ = 0;

  // streaming state
  bool pass_active_ = false, pass_done_ = false, chunk_pending_ = false;
  uint32_t chunk_nodes_ = 0;
  Chunk cur_chunk_;
  std::vector<Real> reach_cur_, reach_next_;  // [N][2][NH]
  Real ev0_ = 0;
  // lose_mass_[level][i] = sum_h reach_p[h] * lose[h] for the acting player p:
  // the probability (times chance) that p forfeits at node i. Empty for a
  // level unless `lose` was pushed for it. One double per public node.
  std::vector<std::vector<Real>> lose_mass_;

  void build_tree_();
  void apply_(PublicState &s, int action, bool &terminal) const;
  int bid_count_(int bid) const { return bid / D_ + 1; }
  int bid_number_(int bid) const { return bid % D_ + 1; }
  Real terminal_utility_(int player, int originator, bool bidder_wins) const {
    return (player == originator) == bidder_wins ? 1.0 : -1.0;
  }
  void advance_level_();
};

}  // namespace lp
