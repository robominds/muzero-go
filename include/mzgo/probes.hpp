// Diagnostics: looking inside the learned model.
//
// MuZero's search never sees the real board below the root. Every node
// deeper in the tree is a hidden state the dynamics function g imagined, and
// the search trusts the prediction function f on it. Whether that trust is
// deserved cannot be read off a win rate; these probes measure it directly
// by comparing imagined states with the real positions they stand for.
//
// All probes use the same construction, along real games:
//   start at position t, take s_0 = h(observation at t), and apply g along the
//   moves the game really played, s_k = g(s_{k-1}, move t+k-1). The real
//   counterpart of s_k is position t+k of the same game. k = 0 is a fresh h.
//
//   latentProbe: does f say the same on s_k as on a fresh h(position t+k)?
//                Mean |value difference| and policy total-variation distance.
//   rulesProbe:  does f's policy on s_k avoid moves that are illegal in the
//                real position t+k (occupied points, suicide, superko)? It
//                was never told the rules, only shown which moves search liked.
//   boardProbe:  can a linear readout, fitted on fresh h states, read the
//                stones (empty / own / opponent per point) back out of s_k?
//                The closest thing to asking "what board does it think this
//                is?" -- the network was never trained to decode the board.
//   analyzePosition: one root search, printed: visits, priors and Q per move,
//                then the principal variation in imagined space.
//
// Probes only read the network (batched inference, no gradients into it).
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "mzgo/game.hpp"
#include "mzgo/game_record.hpp"
#include "mzgo/mcts.hpp"
#include "mzgo/model.hpp"

namespace mzgo {

// Games to probe on, independent of self-play: RolloutMctsPlayer(64) against
// itself (real, varied Go, no network involved). Game i is seeded from
// (seed, i), so the games do not depend on `workers` (0 = all cores). The
// records hold one-hot policies of the moves played and root values 0 (there
// was no MuZero search); no probe reads them.
std::vector<GameRecord> probeGames(int boardSize, float komi, int games, uint64_t seed, int workers);

// One row per k = 0..maxK. Samples: starts t with t + k <= T - 1 (the real
// position must still be one where a move is chosen).
struct LatentRow {
    int k;
    long samples;
    double valueError;   // mean |v(s_k) - v(h(position t+k))|
    double policyTv;     // mean 0.5 * sum_a |p(s_k)_a - p(h(position t+k))_a|, over all actions
};
std::vector<LatentRow> latentProbe(const Model&, const std::vector<GameRecord>&, int maxK);

// One row per k = 0..maxK, samples as for latentProbe. Means over samples of
// the policy mass softmax(f(s_k)) puts on:
struct RulesRow {
    int k;
    long samples;
    double illegalMass;        // moves Game::isLegal rejects at position t+k (pass is always legal)
    double occupiedMass;       // points with a stone at position t+k
    double koSuperkoMass;      // empty, locally legal (Board) points that positional superko forbids
    double capturedPointMass;  // points occupied at t and empty at t+k: freed by a capture
                               // within the k steps (usually legal again)
};
std::vector<RulesRow> rulesProbe(const Model&, const std::vector<GameRecord>&, int maxK);

// A linear readout from a hidden state to the stones, the same at every
// point: at point i, logits_c = b[c] + sum_f w[c][f] * s[f][i] over the F
// hidden channels, classes c = 0 empty, 1 own stone, 2 opponent stone, where
// "own" is the side to move at that position.
struct BoardProbe {
    std::vector<float> w;   // [3][filters]
    std::vector<float> b;   // [3]
};
// Softmax regression fitted on fresh h states of every position 0..T of the
// games (at most kBoardProbeFitPositions of them, a random subset when there
// are more): plain mini-batch gradient descent on the cross-entropy, on
// standardized features (the scaling is folded into w and b afterwards).
// Deterministic for a seed. The network gets no gradient.
constexpr int kBoardProbeFitPositions = 2048;
BoardProbe fitBoardProbe(const Model&, const std::vector<GameRecord>&, int epochs, uint64_t seed);

// One row per k = 0..maxK. Samples: starts t with t + k <= T (the final
// board counts too). The probe's argmax class at every point of s_k against
// position t+k; changed points are those whose content (empty / black /
// white) differs between positions t and t+k. changedAccuracy is 0 when
// changedPoints is 0 (always at k = 0).
struct BoardRow {
    int k;
    long points;
    double accuracy;
    long changedPoints;
    double changedAccuracy;
};
std::vector<BoardRow> boardProbe(const Model&, const BoardProbe&, const std::vector<GameRecord>&, int maxK);

// Runs one search (cfg with root noise turned off) on a non-terminal game
// and describes it: the position, the root value, a table of the `topMoves`
// most visited root moves (GTP coordinates, visits, prior, Q from the side to
// move), search depth statistics, then the principal variation in imagined
// space with each ply's value.
std::string analyzePosition(const Model&, const Game&, const MctsConfig&, int topMoves);

} // namespace mzgo
