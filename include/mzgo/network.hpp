// Adapted from alphazero-go's network (Mark Castelluccio) for mzgo.
// MuZero's network: three functions learned together.
//   representation h(observation) -> s0          (a Trunk, then min-max normalization)
//   dynamics       g(s_{k-1}, a_k) -> s_k (, r_k) (a Trunk on F + 1 planes, then normalization)
//   prediction     f(s_k) -> (policy logits, value) (AlphaZero's two heads)
// Search only ever sees hidden states s, never real positions below the
// root: the action enters g as one extra plane, all zeros except a 1 on the
// point played (a pass is an all-zero plane).
//
// Training unrolls the model K steps from a real position (one TrainBatch)
// and compares f's predictions at every step with the targets recorded in
// the game. The loss is backpropagated through the whole unroll
// (backpropagation through time): the same g runs K times, each call with
// its own saved activations, and its weight gradients add up over the calls.
// Two scalings from the paper keep the long chain stable: the loss of every
// step k >= 1 is weighted 1/K, and the gradient flowing back into a hidden
// state s_k with k >= 1 from the next dynamics call is halved. That follows
// the pseudocode's update_weights, which applies
// `hidden_state = scale_gradient(hidden_state, 0.5)` AFTER each
// recurrent_inference, so the edge s0 -> g_1 is not scaled. (The paper's
// text says "we also scale the gradient at the start of the dynamics
// function by 1/2", which would include s0; where text and pseudocode
// disagree, this project follows the pseudocode, as for the loss weights.)
#pragma once
#include <cstdint>
#include <memory>
#include <vector>
#include "mzgo/backend.hpp"
#include "mzgo/blocks.hpp"
#include "mzgo/model.hpp"
#include "mzgo/params.hpp"
#include "mzgo/train_batch.hpp"

namespace mzgo {

class ThreadPool;   // mzgo/parallel.hpp

struct NetConfig {
    int boardSize = 9, history = 8, reprBlocks = 6, dynBlocks = 6, filters = 64, valueHidden = 64;
    bool rewardHead = false;
    int planes() const { return 2 * history + 2; }      // observation planes (encode.hpp)
    int area() const { return boardSize * boardSize; }
    int actions() const { return area() + 1; }           // every point, plus pass
    int hiddenSize() const { return filters * area(); }  // one sample's hidden state, [filters][area]
    bool operator==(const NetConfig& o) const;
    bool operator!=(const NetConfig& o) const { return !(*this == o); }
};

// Batch means (sum over b / B) per unroll step, before the step's scale.
struct UnrollLoss {
    std::vector<double> policy, value, reward;   // size K+1; reward[0] = 0 (and all 0 without the head)
    double total = 0;                            // sum_k scale_k (policy + value + reward); no L2
};

// Activations for a whole unroll: one cache per call of h, g and f, sized
// for up to maxBatch samples and exactly K steps. With a pool, every layer
// spreads over its threads; the bits are the same with or without one (see
// layers.hpp). The pool must outlive the workspace's use.
class TrainWorkspace {
public:
    TrainWorkspace(const NetConfig& cfg, int maxBatch, int K, const Backend& backend, ThreadPool* pool = nullptr);
    TrainWorkspace(const TrainWorkspace&) = delete;              // the caches point at scratch_
    TrainWorkspace& operator=(const TrainWorkspace&) = delete;
    int maxBatch() const { return maxBatch_; }
    int unroll() const { return K_; }
    // The last forward's outputs at unroll step k in 0..K, for B samples:
    const float* hidden(int k) const { return hidden_[k].data(); }    // [B][hiddenSize], normalized
    const float* logits(int k) const { return heads_[k].logits.data(); }   // [B][actions]
    const float* values(int k) const { return heads_[k].value.values.data(); }   // [B]
    const float* rewards(int k) const;                                   // [B], k >= 1, reward head only
private:
    friend class MuZeroNetwork;
    NetConfig cfg_;
    int maxBatch_, K_;
    const Backend* backend_;
    ThreadPool* pool_;
    Scratch scratch_;
    std::vector<float> obs_;                      // [planes][B][A]
    TrunkCache reprCache_;
    std::vector<TrunkCache> dynCaches_;           // [K]: call k = 1..K is dynCaches_[k - 1]
    std::vector<HeadCache> heads_;                // [K + 1]
    std::vector<ValueHeadCache> rewardCaches_;    // [K] with the reward head, else empty
    // state_[k] [F + 1][B][A], channel-major: channels 0..F-1 are s_k (the
    // input of f at step k), channel F the action plane of a_{k+1}; the whole
    // tensor is the input of g at step k + 1.
    std::vector<std::vector<float>> state_;       // [K + 1]
    std::vector<std::vector<float>> hidden_;      // [K + 1] [B][hiddenSize]: s_k sample by sample
    std::vector<std::vector<MinMaxCache>> minMax_;   // [K + 1][B]
    std::vector<std::vector<float>> dLogits_, dValues_, dRewards_;   // loss gradients per step
    // Backward temporaries, [F][B][A] except dIn_ [F + 1][B][A].
    std::vector<float> dHidden_, dTmp_, dRaw_, dNext_, dIn_, dSample_;
};

class MuZeroNetwork {
public:
    MuZeroNetwork(const NetConfig& cfg, uint64_t seed);   // He init via Rng(seed)
    const NetConfig& config() const { return cfg_; }

    // Fixed order, stable names: "repr.stem.conv.w", "repr.block0.bn1.gamma",
    // ..., "dyn.stem.conv.w", ..., "pred.policy.conv.w", ..., "pred.policy.fc.b",
    // "pred.value.conv.w", ..., "pred.value.fc2.b", then with the reward head
    // "reward.conv.w", ..., "reward.fc2.b". decay: conv and linear weights.
    std::vector<ParamView> params();
    std::vector<BufferView> buffers();   // every batch norm's runMean, runVar, same order
    void zeroGrad();

    // Training: zeroGrad, forward with batch-statistics BN (running stats
    // updated once per call of each layer, i.e. K times for g per step),
    // loss, backward into the grads. hiddenGradScale: the paper's 0.5 on
    // the gradient that g_{k+1}'s hidden input sends back into s_k, for
    // k >= 1 only (s0 gets g_1's full gradient; see the top of this file);
    // with 1.0 the gradient is the exact gradient of UnrollLoss::total
    // (what the gradient checks use). With K = 1 it changes nothing.
    // batch.K must equal ws.unroll(), batch.B <= ws.maxBatch().
    UnrollLoss trainStep(TrainWorkspace& ws, const TrainBatch& batch, float hiddenGradScale = 0.5f);
    // The same forward and loss in Probe mode (batch statistics, running
    // stats untouched), without backward.
    UnrollLoss computeLoss(TrainWorkspace& ws, const TrainBatch& batch);
    // One unroll in Calibrate mode: every batch norm's running statistics
    // are SET to those of its input in this pass. A layer called several
    // times (g K times, f K + 1 times) keeps its last call's statistics.
    void calibrateBatchNorm(TrainWorkspace& ws, const TrainBatch& batch);

    // Inference (running statistics), for search. The network is only read,
    // so any number of threads may run these at once, each with its own
    // InferWorkspace. Outputs are resized to B.
    struct InferWorkspace {              // opaque per-thread buffers (defined in network.cpp)
        virtual ~InferWorkspace() = default;
    };
    std::unique_ptr<InferWorkspace> newInferWorkspace(int maxBatch, const Backend& backend) const;
    // obs [B][planes][area] -> hidden s0, logits, values; rewards 0.
    void initialInference(InferWorkspace& ws, const float* obs, int B, ModelOutput& out) const;
    // hidden [B][hiddenSize], actions [B] (area = pass) -> s1, logits, values,
    // rewards (0 without the reward head). hidden may point into out.hidden.
    void recurrentInference(InferWorkspace& ws, const float* hidden, const int* actions, int B,
                            ModelOutput& out) const;

private:
    NetConfig cfg_;
    Trunk repr_, dyn_;
    PredictionHeads pred_;
    ValueHead reward_;                   // in params() only with cfg_.rewardHead

    void forward(TrainWorkspace& ws, const TrainBatch& batch, BnMode mode);
    UnrollLoss loss(TrainWorkspace& ws, const TrainBatch& batch, bool withGradients) const;
    void backward(TrainWorkspace& ws, int B, float hiddenGradScale);
};

// The Model interface (model.hpp) over a MuZeroNetwork: each session owns
// an InferWorkspace; the network and the backend are shared read-only and
// must outlive the model.
class NetworkModel : public Model {
public:
    NetworkModel(const MuZeroNetwork& net, const Backend& backend);
    ModelDims dims() const override;
    int history() const override;
    std::unique_ptr<ModelSession> newSession(int maxBatch) const override;
private:
    const MuZeroNetwork& net_;
    const Backend& backend_;
};

} // namespace mzgo
