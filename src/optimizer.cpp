#include "mzgo/optimizer.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include "mzgo/layers.hpp"
#include "mzgo/parallel.hpp"

namespace mzgo {

// ---------------------------------------------------------------- SgdMomentum

SgdMomentum::SgdMomentum(const std::vector<ParamView>& params, float momentum, float weightDecay)
    : momentum_(momentum), weightDecay_(weightDecay) {
    for (const ParamView& p : params) velocity_.emplace_back(p.size, 0.0f);
}

void SgdMomentum::step(const std::vector<ParamView>& paramList, float lr, ThreadPool* pool) {
    std::vector<ParamView> params = paramList;   // the chunks below take mutable views
    assert(params.size() == velocity_.size());
    // One list of (parameter, element range) chunks over every tensor, so
    // the whole update is a single run on the pool.
    struct Chunk { size_t param, begin, end; };
    std::vector<Chunk> chunks;
    for (size_t k = 0; k < params.size(); ++k) {
        assert(velocity_[k].size() == params[k].size);
        for (size_t begin = 0; begin < params[k].size; begin += kElementChunk)
            chunks.push_back({k, begin, std::min(params[k].size, begin + kElementChunk)});
    }
    parallelChunks(pool, static_cast<int>(chunks.size()), [&](int c) {
        const Chunk& chunk = chunks[static_cast<size_t>(c)];
        ParamView& p = params[chunk.param];
        std::vector<float>& v = velocity_[chunk.param];
        // Decay only weights: shrinking BN's gamma/beta or a bias toward zero
        // does not make the function smoother, it just fights the fit.
        const float decay = p.decay ? weightDecay_ : 0.0f;
        for (size_t i = chunk.begin; i < chunk.end; ++i) {
            const float g = p.grad[i] + decay * p.value[i];
            v[i] = momentum_ * v[i] + g;
            p.value[i] -= lr * v[i];
        }
    });
}

// ---------------------------------------------------------------- learning rate

float exponentialLr(float lrInit, float decayRate, int decaySteps, long step) {
    const double exponent = static_cast<double>(step) / static_cast<double>(decaySteps);
    return static_cast<float>(static_cast<double>(lrInit) * std::pow(static_cast<double>(decayRate), exponent));
}

} // namespace mzgo
