// Adapted from alphazero-go's network (Mark Castelluccio) for mzgo.
#include "mzgo/network.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>
#include "mzgo/board.hpp"
#include "mzgo/parallel.hpp"
#include "mzgo/rng.hpp"

namespace mzgo {

bool NetConfig::operator==(const NetConfig& o) const {
    return boardSize == o.boardSize && history == o.history && reprBlocks == o.reprBlocks &&
           dynBlocks == o.dynBlocks && filters == o.filters && valueHidden == o.valueHidden &&
           rewardHead == o.rewardHead && unrollSteps == o.unrollSteps;
}

namespace {

// Samples per chunk of a per-sample loop on the pool.
constexpr size_t kSampleChunk = 4;

// The trunk's raw output [F][B][A] -> s, one min-max-normalized hidden
// state per sample: hidden [B][F*A] (sample by sample, as the outside sees
// it) and, if state != null, the same values channel-major in state [F][B][A]
// (as the next layers read it).
void normalizeHidden(const float* raw, int B, int F, int A, float* hidden, MinMaxCache* caches, float* state,
                     ThreadPool* pool) {
    const int n = F * A;
    toSampleMajor(raw, B, F, A, hidden, pool);
    parallelRanges(pool, static_cast<size_t>(B), kSampleChunk, [&](size_t b0, size_t b1) {
        for (size_t b = b0; b < b1; ++b) minMaxForward(hidden + b * n, n, hidden + b * n, caches[b]);
    });
    if (state) toChannelMajor(hidden, B, F, A, state, pool);
}

// The action plane [B][A] (one channel of g's input): all zeros except a 1
// on the point played; a pass (action == A) is all zeros. actions[b * stride].
void writeActionPlane(const int* actions, int stride, int B, int A, float* plane) {
    std::fill(plane, plane + static_cast<size_t>(B) * A, 0.0f);
    for (int b = 0; b < B; ++b) {
        const int a = actions[static_cast<size_t>(b) * stride];
        assert(a >= 0 && a <= A);
        if (a < A) plane[static_cast<size_t>(b) * A + a] = 1.0f;
    }
}

} // namespace

// ---------------------------------------------------------------- TrainWorkspace

TrainWorkspace::TrainWorkspace(const NetConfig& cfg, int maxBatch, int K, const Backend& backend, ThreadPool* pool)
    : cfg_(cfg), maxBatch_(maxBatch), K_(K), backend_(&backend), pool_(pool),
      scratch_(std::max(cfg.planes(), cfg.filters + 1), cfg.filters, maxBatch, cfg.boardSize),
      reprCache_(cfg.filters, cfg.reprBlocks, maxBatch, cfg.boardSize, scratch_) {
    assert(maxBatch >= 1);
    if (K < 0 || K > cfg.unrollSteps)
        throw std::invalid_argument("TrainWorkspace: K = " + std::to_string(K) + " is outside 0.." +
                                    std::to_string(cfg.unrollSteps) + " (the network's unroll_steps)");
    const int n = cfg.boardSize, F = cfg.filters;
    const size_t M = static_cast<size_t>(maxBatch) * cfg.area();
    obs_.resize(static_cast<size_t>(cfg.planes()) * M);
    for (int k = 0; k < K; ++k) dynCaches_.emplace_back(F, cfg.dynBlocks, maxBatch, n, scratch_);
    for (int k = 0; k <= K; ++k) heads_.emplace_back(cfg.valueHidden, maxBatch, n, scratch_);
    if (cfg.rewardHead)
        for (int k = 0; k < K; ++k) rewardCaches_.emplace_back(cfg.valueHidden, maxBatch, cfg.area());
    state_.assign(K + 1, std::vector<float>((F + 1) * M));
    hidden_.assign(K + 1, std::vector<float>(F * M));
    minMax_.assign(K + 1, std::vector<MinMaxCache>(maxBatch));
    dLogits_.assign(K + 1, std::vector<float>(static_cast<size_t>(maxBatch) * cfg.actions()));
    dValues_.assign(K + 1, std::vector<float>(maxBatch));
    dRewards_.assign(K + 1, std::vector<float>(maxBatch));
    for (std::vector<float>* v : {&dHidden_, &dTmp_, &dRaw_, &dNext_, &dSample_}) v->resize(F * M);
    dIn_.resize((F + 1) * M);
}

const float* TrainWorkspace::rewards(int k) const {
    assert(cfg_.rewardHead && k >= 1 && k <= K_);
    return rewardCaches_[k - 1].values.data();
}

// ---------------------------------------------------------------- MuZeroNetwork

MuZeroNetwork::MuZeroNetwork(const NetConfig& cfg, uint64_t seed)
    : cfg_(cfg),
      // Running statistics per call index (see network.hpp): h once, g and
      // the reward head for s1..sK, f for s0..sK.
      repr_(cfg.planes(), cfg.filters, cfg.reprBlocks, 1),
      dyn_(cfg.filters + 1, cfg.filters, cfg.dynBlocks, cfg.unrollSteps),
      pred_(cfg.filters, cfg.boardSize, cfg.valueHidden, cfg.unrollSteps + 1),
      reward_(cfg.rewardHead ? ValueHead(cfg.filters, cfg.area(), cfg.valueHidden, cfg.unrollSteps)
                             : ValueHead()) {
    assert(cfg.boardSize >= kMinBoardSize && cfg.boardSize <= kMaxBoardSize);
    assert(cfg.history >= 1 && cfg.reprBlocks >= 0 && cfg.dynBlocks >= 0 && cfg.filters >= 1 &&
           cfg.valueHidden >= 1 && cfg.unrollSteps >= 1);
    // One generator, layers in params() order, so a seed fixes every weight;
    // the reward head last, so it leaves the other weights of a seed alone.
    Rng rng(seed);
    repr_.init(rng);
    dyn_.init(rng);
    pred_.init(rng);
    if (cfg.rewardHead) reward_.init(rng);
}

std::vector<ParamView> MuZeroNetwork::params() {
    std::vector<ParamView> out = repr_.params("repr");
    for (ParamView& p : dyn_.params("dyn")) out.push_back(p);
    for (ParamView& p : pred_.params("pred")) out.push_back(p);
    if (cfg_.rewardHead)
        for (ParamView& p : reward_.params("reward")) out.push_back(p);
    return out;
}

std::vector<BufferView> MuZeroNetwork::buffers() {
    // Named by call index: h and f start at k0, g and the reward head at k1.
    std::vector<BufferView> out = repr_.buffers("repr", 0);
    for (BufferView& b : dyn_.buffers("dyn", 1)) out.push_back(b);
    for (BufferView& b : pred_.buffers("pred", 0)) out.push_back(b);
    if (cfg_.rewardHead)
        for (BufferView& b : reward_.buffers("reward", 1)) out.push_back(b);
    return out;
}

void MuZeroNetwork::zeroGrad() {
    for (ParamView& p : params()) std::fill(p.grad, p.grad + p.size, 0.0f);
}

UnrollLoss MuZeroNetwork::trainStep(TrainWorkspace& ws, const TrainBatch& batch, float hiddenGradScale) {
    if (ws.K_ != cfg_.unrollSteps)
        throw std::invalid_argument("trainStep: the workspace unrolls " + std::to_string(ws.K_) +
                                    " steps but the network keeps batch norm statistics for " +
                                    std::to_string(cfg_.unrollSteps));
    zeroGrad();
    forward(ws, batch, BnMode::Train);
    UnrollLoss stats = loss(ws, batch, true);
    backward(ws, batch.B, hiddenGradScale);
    return stats;
}

UnrollLoss MuZeroNetwork::computeLoss(TrainWorkspace& ws, const TrainBatch& batch) {
    forward(ws, batch, BnMode::Probe);
    return loss(ws, batch, false);
}

void MuZeroNetwork::calibrateBatchNorm(TrainWorkspace& ws, const TrainBatch& batch) {
    forward(ws, batch, BnMode::Calibrate);
}

// s0 = norm(h(obs)); (p0, v0) = f(s0); for k = 1..K:
// s_k = norm(g(s_{k-1}, plane(a_k))), r_k = reward(s_k), (p_k, v_k) = f(s_k).
// Batch norm statistics of call index k: f's set k, g's and the reward
// head's set k - 1 (their sets serve s1..sK).
void MuZeroNetwork::forward(TrainWorkspace& ws, const TrainBatch& batch, BnMode mode) {
    assert(ws.cfg_ == cfg_ && ws.K_ <= cfg_.unrollSteps);
    assert(batch.B >= 1 && batch.B <= ws.maxBatch_ && batch.K == ws.K_);
    const Backend& be = *ws.backend_;
    ThreadPool* pool = ws.pool_;
    const int B = batch.B, K = batch.K, n = cfg_.boardSize, A = cfg_.area(), F = cfg_.filters;
    const size_t M = static_cast<size_t>(B) * A;

    toChannelMajor(batch.observations.data(), B, cfg_.planes(), A, ws.obs_.data(), pool);
    repr_.forward(be, ws.obs_.data(), B, n, ws.reprCache_, mode, pool);
    normalizeHidden(ws.reprCache_.output(), B, F, A, ws.hidden_[0].data(), ws.minMax_[0].data(),
                    ws.state_[0].data(), pool);
    pred_.forward(be, ws.state_[0].data(), B, n, ws.heads_[0], mode, pool, BnSets{0});

    for (int k = 1; k <= K; ++k) {
        float* in = ws.state_[k - 1].data();     // s_{k-1} in channels 0..F-1
        writeActionPlane(batch.actions.data() + (k - 1), K, B, A, in + F * M);
        dyn_.forward(be, in, B, n, ws.dynCaches_[k - 1], mode, pool, BnSets{k - 1});
        normalizeHidden(ws.dynCaches_[k - 1].output(), B, F, A, ws.hidden_[k].data(), ws.minMax_[k].data(),
                        ws.state_[k].data(), pool);
        // The reward head reads the normalized s_k, like f: the design's
        // "dynamics -> (s_k, r_k)" read as the reward being a function of
        // the state g produced (my reading; the paper does not say which).
        if (cfg_.rewardHead)
            reward_.forward(be, ws.state_[k].data(), B, n, ws.rewardCaches_[k - 1], mode, pool, BnSets{k - 1});
        pred_.forward(be, ws.state_[k].data(), B, n, ws.heads_[k], mode, pool, BnSets{k});
    }
}

// Per step k: policy = mean_b mask * CE(pi, softmax(logits)) (divided by B,
// not by the number of unmasked samples), value = mean_b (z - v)^2, reward =
// mean_b (u - r)^2 for k >= 1 with the head. total = sum_k scale_k * (...),
// scale_0 = 1, scale_k = 1/K. With gradients, the loss's derivatives with
// respect to the logits, values and rewards go into ws.d*_.
UnrollLoss MuZeroNetwork::loss(TrainWorkspace& ws, const TrainBatch& batch, bool withGradients) const {
    const int B = batch.B, K = batch.K, actions = cfg_.actions();
    UnrollLoss out;
    out.policy.assign(K + 1, 0.0);
    out.value.assign(K + 1, 0.0);
    out.reward.assign(K + 1, 0.0);
    std::vector<double> logZ(B);
    for (int k = 0; k <= K; ++k) {
        const double scale = k == 0 ? 1.0 : 1.0 / K;
        const float* logits = ws.heads_[k].logits.data();
        const float* values = ws.heads_[k].value.values.data();
        const float* rewards = cfg_.rewardHead && k >= 1 ? ws.rewardCaches_[k - 1].values.data() : nullptr;
        auto row = [&](size_t b) { return b * (K + 1) + k; };   // index into the [B][K+1] targets
        // Per sample, on the pool: the softmax normalizer and, for training,
        // the gradients, which belong to their sample alone.
        parallelRanges(ws.pool_, static_cast<size_t>(B), kSampleChunk, [&](size_t b0, size_t b1) {
            for (size_t b = b0; b < b1; ++b) {
                const float* l = logits + b * actions;
                const float* target = batch.policyTargets.data() + row(b) * actions;
                // log softmax(l)_a = l_a - logZ; subtracting the largest
                // logit keeps every exponent <= 0.
                const float maxLogit = *std::max_element(l, l + actions);
                double sumExp = 0.0;
                for (int a = 0; a < actions; ++a) sumExp += std::exp(static_cast<double>(l[a]) - maxLogit);
                logZ[b] = maxLogit + std::log(sumExp);
                if (!withGradients) continue;
                // d/dl_a of -sum pi log softmax = softmax_a * sum(pi) - pi_a.
                double sumPi = 0.0;
                for (int a = 0; a < actions; ++a) sumPi += target[a];
                const double w = scale * batch.policyMask[row(b)] / B;
                float* d = ws.dLogits_[k].data() + b * actions;
                for (int a = 0; a < actions; ++a)
                    d[a] = static_cast<float>(w * (std::exp(l[a] - logZ[b]) * sumPi - target[a]));
                ws.dValues_[k][b] = static_cast<float>(scale * 2.0 * (values[b] - batch.valueTargets[row(b)]) / B);
                if (rewards)
                    ws.dRewards_[k][b] = static_cast<float>(scale * 2.0 * (rewards[b] - batch.rewardTargets[row(b)]) / B);
            }
        });
        // The batch sums run serially, sample by sample: one order of
        // additions, so the same bits however many threads there are.
        double policySum = 0.0, valueSum = 0.0, rewardSum = 0.0;
        for (int b = 0; b < B; ++b) {
            const float* l = logits + static_cast<size_t>(b) * actions;
            const float* target = batch.policyTargets.data() + row(b) * actions;
            double ce = 0.0;
            for (int a = 0; a < actions; ++a) ce -= target[a] * (l[a] - logZ[b]);
            policySum += batch.policyMask[row(b)] * ce;
            const double dv = batch.valueTargets[row(b)] - static_cast<double>(values[b]);
            valueSum += dv * dv;
            if (rewards) {
                const double dr = batch.rewardTargets[row(b)] - static_cast<double>(rewards[b]);
                rewardSum += dr * dr;
            }
        }
        out.policy[k] = policySum / B;
        out.value[k] = valueSum / B;
        out.reward[k] = rewardSum / B;
        out.total += scale * (out.policy[k] + out.value[k] + out.reward[k]);
    }
    return out;
}

// Backprop through time, k = K..0. The gradient reaching s_k is f's (and
// the reward head's) input gradient plus, for k < K, the first F channels of
// g's input gradient at step k + 1 (the action plane's share is dropped),
// times hiddenGradScale for k >= 1 and unscaled for s0: the pseudocode
// scales hidden_state after each recurrent_inference, never after
// initial_inference. Through the normalization, then the trunk that
// produced s_k: g at step k (whose input is s_{k-1}) or h for k = 0.
void MuZeroNetwork::backward(TrainWorkspace& ws, int B, float hiddenGradScale) {
    const Backend& be = *ws.backend_;
    ThreadPool* pool = ws.pool_;
    const int K = ws.K_, n = cfg_.boardSize, A = cfg_.area(), F = cfg_.filters;
    const size_t size = static_cast<size_t>(F) * B * A;
    const int hiddenSize = cfg_.hiddenSize();
    float* dHidden = ws.dHidden_.data();
    float* dSample = ws.dSample_.data();
    float* dRaw = ws.dRaw_.data();
    float* dNext = ws.dNext_.data();

    for (int k = K; k >= 0; --k) {
        const float* s = ws.state_[k].data();
        pred_.backward(be, s, ws.dLogits_[k].data(), ws.dValues_[k].data(), B, n, ws.heads_[k], dHidden, pool);
        if (cfg_.rewardHead && k >= 1) {
            reward_.backward(be, s, ws.dRewards_[k].data(), B, n, ws.rewardCaches_[k - 1], ws.dTmp_.data(), pool);
            addInto(dHidden, ws.dTmp_.data(), size, pool);
        }
        if (k < K) addInto(dHidden, dNext, size, pool);

        // Through the min-max normalization, sample by sample.
        toSampleMajor(dHidden, B, F, A, dSample, pool);
        const float* hidden = ws.hidden_[k].data();
        const MinMaxCache* caches = ws.minMax_[k].data();
        parallelRanges(pool, static_cast<size_t>(B), kSampleChunk, [&](size_t b0, size_t b1) {
            for (size_t b = b0; b < b1; ++b)
                minMaxBackward(dSample + b * hiddenSize, hidden + b * hiddenSize, hiddenSize, caches[b],
                               dSample + b * hiddenSize);
        });
        toChannelMajor(dSample, B, F, A, dRaw, pool);

        if (k >= 1) {
            dyn_.backward(be, ws.state_[k - 1].data(), dRaw, B, n, ws.dynCaches_[k - 1], ws.dIn_.data(), pool);
            // scale_gradient(s_{k-1}, 0.5) when s_{k-1} came from g (k >= 2);
            // s0 (k = 1) gets the full gradient. Channels 0..F-1 of g's
            // input are s_{k-1} ([F][B][A] first in [F+1][B][A]); channel F,
            // the action plane, is not learned.
            const float scale = k >= 2 ? hiddenGradScale : 1.0f;
            const float* dIn = ws.dIn_.data();
            parallelRanges(pool, size, kElementChunk, [&](size_t begin, size_t end) {
                for (size_t j = begin; j < end; ++j) dNext[j] = scale * dIn[j];
            });
        } else {
            repr_.backward(be, ws.obs_.data(), dRaw, B, n, ws.reprCache_, nullptr, pool);
        }
    }
}

// ---------------------------------------------------------------- inference

namespace {

struct InferBuffers : MuZeroNetwork::InferWorkspace {
    InferBuffers(const NetConfig& cfg, int maxBatch, const Backend& be)
        : maxBatch(maxBatch), backend(&be),
          scratch(std::max(cfg.planes(), cfg.filters + 1), cfg.filters, maxBatch, cfg.boardSize),
          reprCache(cfg.filters, cfg.reprBlocks, maxBatch, cfg.boardSize, scratch),
          dynCache(cfg.filters, cfg.dynBlocks, maxBatch, cfg.boardSize, scratch),
          heads(cfg.valueHidden, maxBatch, cfg.boardSize, scratch),
          reward(cfg.valueHidden, maxBatch, cfg.area()),
          obs(static_cast<size_t>(cfg.planes()) * maxBatch * cfg.area()),
          state(static_cast<size_t>(cfg.filters + 1) * maxBatch * cfg.area()),
          minMax(maxBatch), dynSets(maxBatch), predSets(maxBatch) {}
    int maxBatch;
    const Backend* backend;
    Scratch scratch;
    TrunkCache reprCache, dynCache;
    HeadCache heads;
    ValueHeadCache reward;
    std::vector<float> obs;        // [planes][B][A]
    std::vector<float> state;      // [F + 1][B][A]: s (channel-major), then the action plane
    std::vector<MinMaxCache> minMax;
    std::vector<int> dynSets, predSets;   // [B]: each row's batch norm set in g (and reward), in f
};

} // namespace

std::unique_ptr<MuZeroNetwork::InferWorkspace> MuZeroNetwork::newInferWorkspace(int maxBatch,
                                                                                const Backend& backend) const {
    assert(maxBatch >= 1);
    return std::make_unique<InferBuffers>(cfg_, maxBatch, backend);
}

// BnMode::Infer only reads the layers (batch norm uses its running
// statistics through the const forwardInfer), so the const_casts below never
// write to the shared network; they let one forward per block serve both
// inference and training.
void MuZeroNetwork::initialInference(InferWorkspace& iws, const float* obs, int B, ModelOutput& out) const {
    InferBuffers& ws = static_cast<InferBuffers&>(iws);
    assert(B >= 1 && B <= ws.maxBatch);
    const Backend& be = *ws.backend;
    const int n = cfg_.boardSize, A = cfg_.area(), F = cfg_.filters;
    toChannelMajor(obs, B, cfg_.planes(), A, ws.obs.data());
    const_cast<Trunk&>(repr_).forward(be, ws.obs.data(), B, n, ws.reprCache, BnMode::Infer);
    out.hidden.resize(static_cast<size_t>(B) * cfg_.hiddenSize());
    normalizeHidden(ws.reprCache.output(), B, F, A, out.hidden.data(), ws.minMax.data(), ws.state.data(), nullptr);
    const_cast<PredictionHeads&>(pred_).forward(be, ws.state.data(), B, n, ws.heads, BnMode::Infer);
    out.logits.assign(ws.heads.logits.begin(), ws.heads.logits.begin() + static_cast<size_t>(B) * cfg_.actions());
    out.values.assign(ws.heads.value.values.begin(), ws.heads.value.values.begin() + B);
    out.rewards.assign(B, 0.0f);
}

void MuZeroNetwork::recurrentInference(InferWorkspace& iws, const float* hidden, const int* actions,
                                       const int* depths, int B, ModelOutput& out) const {
    InferBuffers& ws = static_cast<InferBuffers&>(iws);
    assert(B >= 1 && B <= ws.maxBatch);
    // Depth d is call index min(d, K): f's set d, g's and the reward head's set d - 1.
    for (int b = 0; b < B; ++b) {
        assert(depths[b] >= 1);
        const int k = std::min(depths[b], cfg_.unrollSteps);
        ws.dynSets[b] = k - 1;
        ws.predSets[b] = k;
    }
    const BnSets dynSets{0, ws.dynSets.data()}, predSets{0, ws.predSets.data()};
    const Backend& be = *ws.backend;
    const int n = cfg_.boardSize, A = cfg_.area(), F = cfg_.filters;
    const size_t M = static_cast<size_t>(B) * A;
    // Read hidden (which may live in out.hidden) before out is resized.
    toChannelMajor(hidden, B, F, A, ws.state.data());
    writeActionPlane(actions, 1, B, A, ws.state.data() + F * M);
    const_cast<Trunk&>(dyn_).forward(be, ws.state.data(), B, n, ws.dynCache, BnMode::Infer, nullptr, dynSets);
    out.hidden.resize(static_cast<size_t>(B) * cfg_.hiddenSize());
    normalizeHidden(ws.dynCache.output(), B, F, A, out.hidden.data(), ws.minMax.data(), ws.state.data(), nullptr);
    if (cfg_.rewardHead) {
        const_cast<ValueHead&>(reward_).forward(be, ws.state.data(), B, n, ws.reward, BnMode::Infer, nullptr,
                                                dynSets);
        out.rewards.assign(ws.reward.values.begin(), ws.reward.values.begin() + B);
    } else {
        out.rewards.assign(B, 0.0f);
    }
    const_cast<PredictionHeads&>(pred_).forward(be, ws.state.data(), B, n, ws.heads, BnMode::Infer, nullptr,
                                                predSets);
    out.logits.assign(ws.heads.logits.begin(), ws.heads.logits.begin() + static_cast<size_t>(B) * cfg_.actions());
    out.values.assign(ws.heads.value.values.begin(), ws.heads.value.values.begin() + B);
}

// ---------------------------------------------------------------- NetworkModel

namespace {

class NetworkSession : public ModelSession {
public:
    NetworkSession(const MuZeroNetwork& net, const Backend& be, int maxBatch)
        : net_(net), ws_(net.newInferWorkspace(maxBatch, be)) {}
    void initialInference(const float* observations, int B, ModelOutput& out) override {
        net_.initialInference(*ws_, observations, B, out);
    }
    void recurrentInference(const float* hidden, const int* actions, const int* depths, int B,
                            ModelOutput& out) override {
        net_.recurrentInference(*ws_, hidden, actions, depths, B, out);
    }
private:
    const MuZeroNetwork& net_;
    std::unique_ptr<MuZeroNetwork::InferWorkspace> ws_;
};

} // namespace

NetworkModel::NetworkModel(const MuZeroNetwork& net, const Backend& backend) : net_(net), backend_(backend) {}

ModelDims NetworkModel::dims() const { return ModelDims{net_.config().boardSize, net_.config().filters}; }

int NetworkModel::history() const { return net_.config().history; }

std::unique_ptr<ModelSession> NetworkModel::newSession(int maxBatch) const {
    return std::make_unique<NetworkSession>(net_, backend_, maxBatch);
}

} // namespace mzgo
