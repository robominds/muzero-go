// Copied from alphazero-go (Mark Castelluccio), adapted for mzgo.
// How a model learns from a gradient. Plain SGD steps straight down the
// current minibatch's gradient, which is noisy; momentum keeps a running
// "velocity" so consistent directions build up speed while noise cancels.
// Weight decay adds a pull toward zero on the weights (L2 regularization),
// which keeps them small and the model smoother. The learning rate decays
// exponentially with the training step: large early, smaller later.
#pragma once
#include <string>
#include <utility>
#include <vector>
#include "mzgo/params.hpp"

namespace mzgo {

class ThreadPool;   // mzgo/parallel.hpp

// SGD with momentum and L2 weight decay:
//   g' = grad + weightDecay * w   (only for params with decay == true)
//   v  = momentum * v + g'
//   w -= lr * v
class SgdMomentum {
public:
    SgdMomentum(const std::vector<ParamView>& params, float momentum, float weightDecay);
    // Each parameter is updated on its own, so with a pool (chunks of
    // kElementChunk elements per parameter tensor) the bits are unchanged.
    void step(const std::vector<ParamView>& params, float lr, ThreadPool* pool = nullptr);
    std::vector<std::vector<float>>& velocity() { return velocity_; }   // one per param, params() order
    const std::vector<std::vector<float>>& velocity() const { return velocity_; }
private:
    float momentum_, weightDecay_;
    std::vector<std::vector<float>> velocity_;
};

// Exponential learning-rate decay per training step:
//   lr = lrInit * decayRate^(step / decaySteps)
float exponentialLr(float lrInit, float decayRate, int decaySteps, long step);

} // namespace mzgo
