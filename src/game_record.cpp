#include "mzgo/game_record.hpp"
#include <algorithm>
#include <cassert>
#include "mzgo/encode.hpp"
#include "mzgo/game.hpp"
#include "mzgo/symmetry.hpp"

namespace mzgo {

GameRecord recordFromGame(const Game& finished, const std::vector<float>& policies,
                          const std::vector<float>& rootValues) {
    GameRecord r;
    r.boardSize = finished.size();
    r.komi = finished.komi();
    r.moves = finished.moves();
    const int T = r.length();
    const int area = r.area();
    assert(policies.size() == static_cast<size_t>(T) * r.numActions());
    assert(rootValues.size() == static_cast<size_t>(T));
    r.stones.resize(static_cast<size_t>(T + 1) * area);
    for (int p = 0; p <= T; ++p) {
        finished.boardAt(p).exportStones(r.stones.data() + static_cast<size_t>(p) * area);
    }
    r.policies = policies;
    r.rootValues = rootValues;
    r.resultForBlack = finished.resultForBlack();
    return r;
}

void encodeRecordPosition(const GameRecord& g, int p, int history, int sym, float* out, bool passPlane) {
    assert(p >= 0 && p <= g.length());
    const int area = g.area();
    std::vector<const uint8_t*> stones;
    for (int k = 0; k < history && k <= p; ++k) {
        stones.push_back(g.stones.data() + static_cast<size_t>(p - k) * area);
    }
    const bool lastMovePass = p > 0 && g.moves[static_cast<size_t>(p - 1)] == g.area();
    encodePlanes(stones, g.boardSize, g.toMoveAt(p), history, sym, out, passPlane, lastMovePass);
}

void fillSample(const GameRecord& g, int t, int K, int history, bool passPlane, int sym, Rng& rng,
                TrainBatch& batch, int b) {
    const int T = g.length();
    const int n = g.boardSize;
    const int A = g.numActions();
    const int planes = numPlanes(history, passPlane);
    assert(t >= 0 && t < T);
    assert(batch.K == K && b >= 0 && b < batch.B);

    encodeRecordPosition(g, t, history, sym,
                         batch.observations.data() + static_cast<size_t>(b) * planes * g.area(), passPlane);

    for (int k = 0; k < K; ++k) {
        const int a = (t + k < T) ? g.moves[t + k] : rng.below(A);
        batch.actions[static_cast<size_t>(b) * K + k] = transformAction(sym, a, n);
    }

    // A game the players ended (two passes) is over at T: its last move
    // earns the outcome and the state after it is absorbing (value 0, reward
    // 0 forever). A game the move cap stopped was only truncated: its last
    // move was ordinary (reward 0), the score at the cap is the best
    // estimate of position T's value, and past T there is no target at all.
    const bool ended = g.endedByPasses();
    for (int k = 0; k <= K; ++k) {
        const int p = t + k;
        const size_t slot = static_cast<size_t>(b) * (K + 1) + k;
        float* pol = batch.policyTargets.data() + slot * A;
        std::fill(pol, pol + A, 0.0f);
        const bool known = ended || p <= T;
        batch.valueMask[slot] = (known && g.trainValue) ? 1.0f : 0.0f;
        batch.rewardMask[slot] = known ? 1.0f : 0.0f;
        if (p < T) {
            const float* src = g.policies.data() + static_cast<size_t>(p) * A;
            for (int a = 0; a < A; ++a) pol[transformAction(sym, a, n)] = src[a];
            batch.policyMask[slot] = g.policyTrainedAt(p) ? 1.0f : 0.0f;
            batch.valueTargets[slot] = g.outcomeFor(p);
        } else {
            batch.policyMask[slot] = 0.0f;
            batch.valueTargets[slot] = (!ended && p == T) ? g.outcomeFor(T) : 0.0f;
        }
        // The move that ended the game (made at T-1) earns its mover the outcome.
        batch.rewardTargets[slot] = (ended && k >= 1 && p == T) ? g.outcomeFor(T - 1) : 0.0f;
    }
}

} // namespace mzgo
