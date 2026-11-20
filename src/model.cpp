#include "mzgo/model.hpp"
#include "mzgo/train_batch.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace mzgo {

void softmax(const float* logits, int n, float* probs) {
    float maxLogit = logits[0];
    for (int i = 1; i < n; ++i) maxLogit = std::max(maxLogit, logits[i]);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        probs[i] = std::exp(logits[i] - maxLogit);
        sum += probs[i];
    }
    for (int i = 0; i < n; ++i) probs[i] = static_cast<float>(probs[i] / sum);
}

void TrainBatch::resize(int b, int k, int planes, int area, int numActions) {
    B = b;
    K = k;
    const size_t sb = static_cast<size_t>(b), sk = static_cast<size_t>(k);
    observations.assign(sb * static_cast<size_t>(planes) * static_cast<size_t>(area), 0.0f);
    actions.assign(sb * sk, 0);
    policyTargets.assign(sb * (sk + 1) * static_cast<size_t>(numActions), 0.0f);
    policyMask.assign(sb * (sk + 1), 0.0f);
    valueTargets.assign(sb * (sk + 1), 0.0f);
    rewardTargets.assign(sb * (sk + 1), 0.0f);
}

} // namespace mzgo
