// One training batch of K-step unrolls: what the replay buffer hands the
// trainer. Position b is observed at unroll step 0, then the K recorded
// actions are applied in turn, with a target at each of the K+1 steps.
#pragma once
#include <vector>

namespace mzgo {

struct TrainBatch {
    int B = 0, K = 0;
    std::vector<float> observations;   // [B][planes][area]
    std::vector<int>   actions;        // [B][K]   actions[b*K+k] leads from unroll step k to k+1
    std::vector<float> policyTargets;  // [B][K+1][actions]
    std::vector<float> policyMask;     // [B][K+1] 1 train, 0 absorbing (no policy loss)
    std::vector<float> valueTargets;   // [B][K+1]
    std::vector<float> rewardTargets;  // [B][K+1], index 0 unused
    // 1 train, 0 no target: past the cap of a game the move cap truncated,
    // nothing is known about the state (game_record.hpp).
    std::vector<float> valueMask;      // [B][K+1]
    std::vector<float> rewardMask;     // [B][K+1], index 0 unused
    // Sizes every vector for the given shape (contents are zero-filled).
    void resize(int B, int K, int planes, int area, int actions);
};

} // namespace mzgo
