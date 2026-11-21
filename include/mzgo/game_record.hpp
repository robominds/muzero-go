// A finished self-play game, stored for training, and the MuZero targets cut from it.
//
// MuZero never replays the rules during training: the network learns its own
// model of the game from recorded data. So a record keeps, for every position
// p (the board before move p), the stones on the board, plus what search said
// there (visit distribution and root value). From a record we can rebuild the
// network's input for any position (encodeRecordPosition) and the training
// targets for a K-step unroll starting at position t (fillSample):
//
//   step k looks at position p = t + k, with
//     value  target = the game's final outcome from the view of the player to
//                     move at p (board games: pure Monte-Carlo return),
//     policy target = the search's visit distribution at p,
//     reward target = nonzero only for the move that ended the game,
//   and the actions fed to the dynamics model are the moves actually played.
//   Past the end of a game the players ended (two passes) the state is
//   "absorbing": value 0, reward 0, no policy loss (mask 0), and the action
//   is random padding.
//   A game the move cap stopped did not end, it was truncated: every reward
//   target is 0 (its last move was an ordinary move), the value target at
//   p = T is outcomeFor(T) (the score at the cap, the best estimate of that
//   position's value), and past T there is no target (value, reward and
//   policy masks 0; the actions are still random padding).
#pragma once
#include <cstdint>
#include <vector>
#include "mzgo/board.hpp"
#include "mzgo/rng.hpp"
#include "mzgo/train_batch.hpp"

namespace mzgo {

class Game;

struct GameRecord {
    int boardSize = 0;
    float komi = 0;
    std::vector<int> moves;           // T moves; Black moves at even plies
    std::vector<uint8_t> stones;      // [(T+1)][area] Color bytes; position p = before move p
    std::vector<float> policies;      // [T][actions] root visit distribution
    std::vector<float> rootValues;    // [T] root search value, side to move's view
    float resultForBlack = 0;         // +1 / -1 / 0
    // [T] or empty; 0 at p means the move at p was not chosen by the net being
    // trained (a league opponent), so position p trains value/reward/dynamics
    // but not the policy. Empty means every move trains the policy.
    std::vector<uint8_t> policyMask;

    int length() const { return static_cast<int>(moves.size()); }
    int area() const { return boardSize * boardSize; }
    int numActions() const { return area() + 1; }
    Color toMoveAt(int p) const { return p % 2 == 0 ? Color::Black : Color::White; }
    // Result from the view of the side to move at position p.
    float outcomeFor(int p) const { return p % 2 == 0 ? resultForBlack : -resultForBlack; }
    bool policyTrainedAt(int p) const {
        return policyMask.empty() || policyMask[static_cast<size_t>(p)] != 0;
    }
    // True iff the players ended the game (its last two moves are passes);
    // false for a game the move cap stopped.
    bool endedByPasses() const {
        const int T = length(), pass = area();
        return T >= 2 && moves[T - 1] == pass && moves[T - 2] == pass;
    }
};

// policies is flat [T][actions], rootValues is [T] (sizes asserted).
GameRecord recordFromGame(const Game& finished, const std::vector<float>& policies,
                          const std::vector<float>& rootValues);

// Network input for position p (p <= T) with `history` boards of history and
// symmetry `sym`; identical to encodeGame on the same position (including the
// optional pass plane, set iff move p-1 was a pass).
void encodeRecordPosition(const GameRecord&, int p, int history, int sym, float* out, bool passPlane = false);

// Fills sample b of `batch` from position t (0 <= t < T) of the record.
// The batch must have been resize()d for K, numPlanes(history, passPlane), area, actions.
// Policy entries and actions are moved by symmetry `sym`; `rng` supplies the
// random actions past the end of the game.
void fillSample(const GameRecord&, int t, int K, int history, bool passPlane, int sym, Rng& rng,
                TrainBatch& batch, int b);

} // namespace mzgo
