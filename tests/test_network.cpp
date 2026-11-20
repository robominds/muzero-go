#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>
#include "mzgo/backend.hpp"
#include "mzgo/checkpoint.hpp"
#include "mzgo/network.hpp"
#include "mzgo/parallel.hpp"
#include "mzgo/rng.hpp"
#include "mzgo/serialize.hpp"

using namespace mzgo;

namespace {

NetConfig smallConfig(bool rewardHead, int unrollSteps = 3, bool passPlane = false) {
    NetConfig cfg;
    cfg.boardSize = 5;
    cfg.history = 2;
    cfg.reprBlocks = 1;
    cfg.dynBlocks = 1;
    cfg.filters = 4;
    cfg.valueHidden = 4;
    cfg.rewardHead = rewardHead;
    cfg.unrollSteps = unrollSteps;
    cfg.passPlane = passPlane;
    return cfg;
}

// A random unroll batch: observation planes in {0, 1}, random actions (some
// of them passes), normalized random policy targets with every mask 1 (policy, value, reward),
// value targets in [-1, 1], reward targets in {-1, 0, 1}.
TrainBatch randomBatch(const NetConfig& cfg, int B, int K, Rng& rng) {
    TrainBatch t;
    t.resize(B, K, cfg.planes(), cfg.area(), cfg.actions());
    for (float& v : t.observations) v = rng.uniform() < 0.5 ? 0.0f : 1.0f;
    for (int& a : t.actions) a = rng.below(cfg.actions());
    for (int i = 0; i < B * (K + 1); ++i) {
        float* row = t.policyTargets.data() + static_cast<size_t>(i) * cfg.actions();
        double sum = 0.0;
        for (int a = 0; a < cfg.actions(); ++a) sum += row[a] = static_cast<float>(rng.uniform());
        for (int a = 0; a < cfg.actions(); ++a) row[a] = static_cast<float>(row[a] / sum);
        t.policyMask[i] = t.valueMask[i] = t.rewardMask[i] = 1.0f;
        t.valueTargets[i] = static_cast<float>(2.0 * rng.uniform() - 1.0);
        t.rewardTargets[i] = static_cast<float>(rng.below(3) - 1);
    }
    return t;
}

// Random batch norm affine parameters and linear biases (see test_blocks):
// exercises their gradients, moves ReLU kinks away from the channel means and
// tanh outputs away from 0.
void randomizeAffine(MuZeroNetwork& net, Rng& rng) {
    for (const ParamView& p : net.params()) {
        const bool isGamma = p.name.find(".gamma") != std::string::npos;
        const bool isBeta = p.name.find(".beta") != std::string::npos;
        const bool isBias = p.name.size() > 2 && p.name.compare(p.name.size() - 2, 2, ".b") == 0;
        for (size_t i = 0; i < p.size; ++i) {
            if (isGamma) p.value[i] = static_cast<float>(0.5 + rng.uniform());
            if (isBeta) p.value[i] = static_cast<float>(rng.uniform() - 0.5);
            if (isBias) p.value[i] = static_cast<float>(rng.uniform() - 0.5);
        }
    }
}

std::vector<std::vector<float>> gradsOf(MuZeroNetwork& net) {
    std::vector<std::vector<float>> out;
    for (const ParamView& p : net.params()) out.emplace_back(p.grad, p.grad + p.size);
    return out;
}

std::vector<float> buffersOf(MuZeroNetwork& net) {
    std::vector<float> all;
    for (const BufferView& b : net.buffers()) all.insert(all.end(), b.value, b.value + b.size);
    return all;
}

// The brief's tolerance: relative error < 2e-2, or absolute error < 1e-4.
bool gradClose(double a, double b) {
    const double diff = std::fabs(a - b);
    return diff < 1e-4 || diff < 2e-2 * std::max(std::fabs(a), std::fabs(b));
}

struct Differences { double central, forward, backward; };

// Central and one-sided differences of loss() in *value with steps of +-h
// (the steps actually taken are measured after float rounding).
template <typename LossFn>
Differences differences(float* value, float h, LossFn loss) {
    const float saved = *value;
    const float up = saved + h, down = saved - h;
    const double mid = loss();
    *value = up;
    const double lossUp = loss();
    *value = down;
    const double lossDown = loss();
    *value = saved;
    return {(lossUp - lossDown) / (static_cast<double>(up) - down), (lossUp - mid) / (static_cast<double>(up) - saved),
            (mid - lossDown) / (static_cast<double>(saved) - down)};
}

// One checked entry, with steps of 1e-2 down to 1e-4. The unrolled loss is
// only piecewise smooth: thousands of ReLUs, and every hidden state's argmin
// and argmax, switch somewhere, and one parameter moves many of them, so a
// step often straddles kinks (the central difference then averages
// different slopes). true: some step's central difference matches. Asserts
// on a mismatch: over the three smallest steps the central differences
// agree with each other and, at each step, the forward and backward
// differences do too (no kink in sight), yet not with the analytic
// gradient -- a smooth region where the gradient is wrong. false otherwise
// (ill-conditioned): the caller re-draws another entry. The loss is summed
// in double from float outputs; its float noise (~1e-7) keeps steps above 1e-4.
template <typename LossFn>
bool checkEntry(const std::string& what, float* value, double analytic, LossFn loss) {
    std::vector<Differences> d;
    for (float h : {1e-2f, 3e-3f, 1e-3f, 3e-4f, 1e-4f}) {
        d.push_back(differences(value, h, loss));
        if (gradClose(analytic, d.back().central)) return true;
    }
    bool smooth = true;
    for (size_t s = d.size() - 3; s < d.size(); ++s)
        smooth = smooth && gradClose(d[s].forward, d[s].backward) && gradClose(d[s].central, d.back().central);
    if (smooth) {
        std::printf("%s: analytic %.7f does not match finite differences %.7f\n", what.c_str(), analytic,
                    d.back().central);
        assert(false);
    }
    return false;
}

std::string tempPath(const std::string& name) {
    return (std::filesystem::temp_directory_path() /
            ("mzgo_test_network_" + std::to_string(getpid()) + "_" + name)).string();
}

void assertNear(const std::vector<float>& a, const float* b, size_t count, double tol) {
    assert(a.size() >= count);
    for (size_t i = 0; i < count; ++i) {
        if (std::fabs(static_cast<double>(a[i]) - b[i]) > tol) {
            std::printf("  entry %zu: %.7f vs %.7f\n", i, a[i], b[i]);
            assert(false);
        }
    }
}

} // namespace

static void test_param_names() {
    for (bool reward : {false, true}) {
        MuZeroNetwork net(smallConfig(reward), 1);
        std::vector<ParamView> params = net.params();
        std::set<std::string> names;
        for (const ParamView& p : params) {
            names.insert(p.name);
            const bool isWeight = p.name.compare(p.name.size() - 2, 2, ".w") == 0;
            assert(p.decay == isWeight);
        }
        assert(names.size() == params.size());   // unique
        for (const char* name : {"repr.stem.conv.w", "repr.block0.conv2.w", "dyn.stem.bn.gamma",
                                 "dyn.block0.bn1.gamma", "pred.policy.fc.b", "pred.value.fc2.w"})
            assert(names.count(name) == 1);
        assert(names.count("reward.fc2.w") == (reward ? 1u : 0u));
        assert(params.front().name == "repr.stem.conv.w");
        // g reads F + 1 planes: the hidden state and the action plane.
        const NetConfig& cfg = net.config();
        for (const ParamView& p : params) {
            if (p.name == "dyn.stem.conv.w") assert(p.size == static_cast<size_t>(cfg.filters * (cfg.filters + 1) * 9));
            if (p.name == "repr.stem.conv.w") assert(p.size == static_cast<size_t>(cfg.filters * cfg.planes() * 9));
        }
        // Running statistics per call index: h one set, g and the reward
        // head K (k1..kK), f K + 1 (k0..kK); K = 3 here.
        const int K = cfg.unrollSteps;
        const size_t sets = (1 + 2) * 1 + (1 + 2) * K + 2 * (K + 1) + (reward ? K : 0);
        std::vector<BufferView> buffers = net.buffers();
        assert(buffers.size() == 2 * sets);
        std::set<std::string> bufferNames;
        for (const BufferView& b : buffers) bufferNames.insert(b.name);
        assert(bufferNames.size() == buffers.size());
        for (const char* name : {"repr.stem.bn.runMean.k0", "dyn.stem.bn.runVar.k1", "dyn.block0.bn1.runMean.k3",
                                 "pred.policy.bn.runMean.k0", "pred.value.bn.runVar.k3"})
            assert(bufferNames.count(name) == 1);
        for (const char* name : {"repr.stem.bn.runMean.k1", "dyn.stem.bn.runMean.k0", "dyn.stem.bn.runMean.k4",
                                 "pred.policy.bn.runMean.k4"})
            assert(bufferNames.count(name) == 0);
        assert(bufferNames.count("reward.bn.runMean.k3") == (reward ? 1u : 0u));
        assert(bufferNames.count("reward.bn.runMean.k0") == 0);
    }
    std::printf("test_param_names passed\n");
}

// trainStep(..., 1.0) against central differences of computeLoss().total.
static void checkUnrollGradient(bool rewardHead) {
    const int B = 3, K = 2;
    const NetConfig cfg = smallConfig(rewardHead, K);
    NaiveBackend be;
    MuZeroNetwork net(cfg, 21);
    Rng rng(22);
    randomizeAffine(net, rng);
    TrainBatch batch = randomBatch(cfg, B, K, rng);
    batch.policyMask[1 * (K + 1) + 2] = 0.0f;           // sample 1 is past the game's end at k = 2
    batch.actions[2 * K + 0] = cfg.area();               // sample 2 passes first
    // Sample 0 is past the cap of a truncated game at k = 2: no value or
    // reward target there (and no policy target).
    batch.policyMask[0 * (K + 1) + 2] = batch.valueMask[0 * (K + 1) + 2] = batch.rewardMask[0 * (K + 1) + 2] = 0.0f;
    batch.valueMask[2 * (K + 1) + 1] = 0.0f;             // and two lone masked entries
    batch.rewardMask[1 * (K + 1) + 1] = 0.0f;
    TrainWorkspace ws(cfg, B, K, be);
    {   // A masked entry's target does not enter the loss.
        TrainBatch wild = batch;
        wild.valueTargets[0 * (K + 1) + 2] = wild.valueTargets[2 * (K + 1) + 1] = 50.0f;
        wild.rewardTargets[0 * (K + 1) + 2] = wild.rewardTargets[1 * (K + 1) + 1] = -50.0f;
        assert(net.computeLoss(ws, wild).total == net.computeLoss(ws, batch).total);
    }

    const UnrollLoss trained = net.trainStep(ws, batch, 1.0f);
    const UnrollLoss probed = net.computeLoss(ws, batch);
    assert(trained.total == probed.total);               // same forward, same batch statistics
    assert(trained.policy.size() == K + 1u && trained.reward[0] == 0.0);
    assert(rewardHead == (trained.reward[1] > 0.0));

    auto loss = [&] { return net.computeLoss(ws, batch).total; };
    // Checks per group of layers, each a random tensor of the group and a
    // random entry of it; an ill-conditioned entry is re-drawn.
    std::vector<std::string> groups = {"repr.", "dyn.", "pred."};
    if (rewardHead) groups.push_back("reward.");
    std::vector<ParamView> params = net.params();
    const std::vector<std::vector<float>> analytic = gradsOf(net);
    int checked = 0, redrawn = 0;
    for (const std::string& group : groups) {
        std::vector<size_t> members;
        for (size_t t = 0; t < params.size(); ++t)
            if (params[t].name.compare(0, group.size(), group) == 0) members.push_back(t);
        assert(!members.empty());
        for (int done = 0; done < 10;) {
            const size_t t = members[static_cast<size_t>(rng.below(static_cast<int>(members.size())))];
            const size_t i = static_cast<size_t>(rng.below(static_cast<int>(params[t].size)));
            if (checkEntry(params[t].name + "[" + std::to_string(i) + "]", params[t].value + i, analytic[t][i], loss)) {
                ++done;
                ++checked;
            } else {
                ++redrawn;
                assert(redrawn <= 10);
            }
        }
    }
    assert(checked >= 30);
}

static void test_unroll_gradient() {
    checkUnrollGradient(false);
    checkUnrollGradient(true);
    std::printf("test_unroll_gradient passed\n");
}

// Every parameter's gradient after trainStep with hidden-gradient scales
// 0, 0.5 and 1 on the same batch (the forward does not depend on the scale).
static std::vector<std::vector<std::vector<float>>> gradsForScales(int K) {
    const NetConfig cfg = smallConfig(false, K);
    const int B = 3;
    NaiveBackend be;
    MuZeroNetwork net(cfg, 31);
    Rng rng(32);
    randomizeAffine(net, rng);
    const TrainBatch batch = randomBatch(cfg, B, K, rng);
    TrainWorkspace ws(cfg, B, K, be);
    std::vector<std::vector<std::vector<float>>> out;
    for (float scale : {0.0f, 0.5f, 1.0f}) {
        net.trainStep(ws, batch, scale);
        out.push_back(gradsOf(net));
    }
    return out;
}

// The pseudocode scales the gradient of s_k for k >= 1 only (after each
// recurrent_inference), so:
// K = 1: s0 -> g_1 is the only dynamics edge and it is unscaled; no
//   gradient depends on the scale, bit for bit.
// K = 2: s1 -> g_2 is scaled. The gradient entering s1 (so g_1's weights)
//   and through it s0 (so h's weights) is affine in the scale; f's weights
//   only see the forward and do not depend on it.
static void test_gradient_scale_half() {
    const auto one = gradsForScales(1);
    assert(one[0] == one[1] && one[2] == one[1]);

    const auto two = gradsForScales(2);
    const std::vector<ParamView> params = MuZeroNetwork(smallConfig(false), 31).params();
    const auto& g0 = two[0];
    const auto& gHalf = two[1];
    const auto& g1 = two[2];
    bool reprDiffers = false, dynDiffers = false;
    for (size_t t = 0; t < params.size(); ++t) {
        const std::string& name = params[t].name;
        if (name.compare(0, 5, "pred.") == 0) {
            assert(g0[t] == gHalf[t] && g1[t] == gHalf[t]);   // bit for bit
            continue;
        }
        assert(name.compare(0, 5, "repr.") == 0 || name.compare(0, 4, "dyn.") == 0);
        // Relative to the tensor's largest gradient: an entry that is a near
        // cancellation of the scaled and unscaled contributions carries the
        // rounding of both terms, which is relative to them, not to the small
        // result.
        double scale = 0.0;
        for (size_t i = 0; i < params[t].size; ++i)
            scale = std::max({scale, std::fabs(static_cast<double>(g0[t][i])), std::fabs(static_cast<double>(g1[t][i]))});
        for (size_t i = 0; i < params[t].size; ++i) {
            const double mean = (static_cast<double>(g0[t][i]) + g1[t][i]) / 2;
            assert(std::fabs(gHalf[t][i] - mean) <= 1e-5 * scale);
            if (std::fabs(static_cast<double>(g1[t][i]) - g0[t][i]) > 1e-3 * scale)
                (name[0] == 'r' ? reprDiffers : dynDiffers) = true;
        }
    }
    assert(reprDiffers && dynDiffers);
    std::printf("test_gradient_scale_half passed\n");
}

// After calibrateBatchNorm on a K-step batch, inference mode on that batch
// computes what the training-mode (batch statistics) forward computed, at
// every unroll step: initialInference reproduces step 0 and
// recurrentInference at depth k reproduces step k (hidden, p, v, r), because
// each call index k keeps its own running statistics.
// Calibration stores the unbiased variance (M / (M - 1) times the batch's,
// what the running average estimates), while the training forward
// normalizes with the biased one; the test rescales runVar by (M - 1) / M.
static void test_inference_matches_probe() {
    const int K = 3;
    NetConfig cfg = smallConfig(true);
    const int B = 4, F = cfg.filters, A = cfg.area();
    const double M = static_cast<double>(B) * A;
    NaiveBackend be;
    MuZeroNetwork net(cfg, 41);
    Rng rng(42);
    randomizeAffine(net, rng);
    const TrainBatch batch = randomBatch(cfg, B, K, rng);
    const size_t hiddenCount = static_cast<size_t>(B) * F * A;
    auto iws = net.newInferWorkspace(B, be);

    TrainWorkspace ws(cfg, B, K, be);
    net.calibrateBatchNorm(ws, batch);
    for (const BufferView& b : net.buffers())
        if (b.name.find(".runVar") != std::string::npos)
            for (size_t i = 0; i < b.size; ++i) b.value[i] = static_cast<float>(b.value[i] * ((M - 1) / M));

    ModelOutput out;
    net.initialInference(*iws, batch.observations.data(), B, out);
    assert(out.hidden.size() == hiddenCount && out.rewards == std::vector<float>(B, 0.0f));
    assertNear(out.hidden, ws.hidden(0), hiddenCount, 1e-4);
    assertNear(out.logits, ws.logits(0), static_cast<size_t>(B) * cfg.actions(), 1e-4);
    assertNear(out.values, ws.values(0), B, 1e-4);
    std::vector<int> actions(B);
    for (int k = 1; k <= K; ++k) {
        for (int b = 0; b < B; ++b) actions[b] = batch.actions[b * K + (k - 1)];
        const std::vector<int> depths(B, k);
        net.recurrentInference(*iws, out.hidden.data(), actions.data(), depths.data(), B, out);
        assertNear(out.hidden, ws.hidden(k), hiddenCount, 1e-4);
        assertNear(out.logits, ws.logits(k), static_cast<size_t>(B) * cfg.actions(), 1e-4);
        assertNear(out.values, ws.values(k), B, 1e-4);
        assertNear(out.rewards, ws.rewards(k), B, 1e-4);
        for (float v : out.hidden) assert(v >= 0.0f && v <= 1.0f);
    }
    std::printf("test_inference_matches_probe passed\n");
}

// Rows of different depths in one recurrent batch: each row gets exactly
// the bits it gets when the whole batch runs at that row's depth (the same
// inputs, so only the batch norm sets differ). Depths past K use set K.
static void test_mixed_depths_bit_identical() {
    const NetConfig cfg = smallConfig(true);   // K = 3
    const int B = 5, K = cfg.unrollSteps, hs = cfg.hiddenSize(), na = cfg.actions();
    NaiveBackend be;
    MuZeroNetwork net(cfg, 45);
    Rng rng(46);
    randomizeAffine(net, rng);
    const TrainBatch batch = randomBatch(cfg, B, K, rng);
    TrainWorkspace ws(cfg, B, K, be);
    net.trainStep(ws, batch);                  // distinct statistics per set
    auto iws = net.newInferWorkspace(B, be);
    ModelOutput root;
    net.initialInference(*iws, batch.observations.data(), B, root);
    const std::vector<int> actions = {0, 3, cfg.area(), 7, 12};
    const std::vector<int> mixed = {3, 1, 2, 7, 1};
    ModelOutput got;
    net.recurrentInference(*iws, root.hidden.data(), actions.data(), mixed.data(), B, got);
    bool differs = false;
    for (int d : {1, 2, 3, 7}) {
        const std::vector<int> uniform(B, d);
        ModelOutput want;
        net.recurrentInference(*iws, root.hidden.data(), actions.data(), uniform.data(), B, want);
        for (int b = 0; b < B; ++b) {
            if (mixed[b] != d) {
                differs = differs || want.logits[static_cast<size_t>(b) * na] != got.logits[static_cast<size_t>(b) * na];
                continue;
            }
            assert(std::equal(want.hidden.begin() + b * hs, want.hidden.begin() + (b + 1) * hs,
                              got.hidden.begin() + b * hs));
            assert(std::equal(want.logits.begin() + b * na, want.logits.begin() + (b + 1) * na,
                              got.logits.begin() + b * na));
            assert(want.values[b] == got.values[b] && want.rewards[b] == got.rewards[b]);
        }
    }
    assert(differs);                           // the depth does change the outputs
    // Depth 7 > K uses set K: the same as depth 3.
    const std::vector<int> three(B, 3), seven(B, 7);
    ModelOutput a, b;
    net.recurrentInference(*iws, root.hidden.data(), actions.data(), three.data(), B, a);
    net.recurrentInference(*iws, root.hidden.data(), actions.data(), seven.data(), B, b);
    assert(a.hidden == b.hidden && a.logits == b.logits && a.values == b.values && a.rewards == b.rewards);
    std::printf("test_mixed_depths_bit_identical passed\n");
}

// The workspace's K is a prefix of the network's unroll_steps; training
// needs the full K (each set of statistics is built for its step).
static void test_unroll_steps_checked() {
    const NetConfig cfg = smallConfig(false, 2);
    NaiveBackend be;
    MuZeroNetwork net(cfg, 47);
    bool threw = false;
    try {
        TrainWorkspace ws(cfg, 2, 3, be);
    } catch (const std::invalid_argument& e) {
        threw = std::string(e.what()).find("unroll_steps") != std::string::npos;
    }
    assert(threw);
    Rng rng(48);
    const TrainBatch batch = randomBatch(cfg, 2, 1, rng);
    TrainWorkspace ws1(cfg, 2, 1, be);
    net.computeLoss(ws1, batch);               // a shorter unroll is fine without training
    net.calibrateBatchNorm(ws1, batch);
    threw = false;
    try {
        net.trainStep(ws1, batch);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    std::printf("test_unroll_steps_checked passed\n");
}

// The training step's bits do not depend on the thread pool.
static void test_pool_bit_identical() {
    const NetConfig cfg = smallConfig(true);
    const int B = 5, K = 3;
    NaiveBackend be;
    Rng rng(51);
    const TrainBatch batch = randomBatch(cfg, B, K, rng);
    MuZeroNetwork plain(cfg, 52);
    TrainWorkspace ws(cfg, B, K, be);
    const UnrollLoss expected = plain.trainStep(ws, batch);
    for (int threads : {2, 3}) {
        ThreadPool pool(threads);
        MuZeroNetwork pooled(cfg, 52);
        TrainWorkspace pws(cfg, B, K, be, &pool);
        const UnrollLoss got = pooled.trainStep(pws, batch);
        assert(got.total == expected.total && got.policy == expected.policy && got.value == expected.value &&
               got.reward == expected.reward);
        assert(gradsOf(pooled) == gradsOf(plain));
        assert(buffersOf(pooled) == buffersOf(plain));
    }
    std::printf("test_pool_bit_identical passed\n");
}

// NetworkModel's sessions give exactly the network's own inference.
static void test_network_model() {
    const NetConfig cfg = smallConfig(true);
    const int B = 3;
    NaiveBackend be;
    MuZeroNetwork net(cfg, 61);
    Rng rng(62);
    const TrainBatch batch = randomBatch(cfg, B, 1, rng);
    NetworkModel model(net, be);
    assert(model.dims().boardSize == cfg.boardSize && model.dims().filters == cfg.filters);
    assert(model.history() == cfg.history);
    auto session = model.newSession(B);
    auto iws = net.newInferWorkspace(B, be);
    ModelOutput a, b;
    session->initialInference(batch.observations.data(), B, a);
    net.initialInference(*iws, batch.observations.data(), B, b);
    assert(a.hidden == b.hidden && a.logits == b.logits && a.values == b.values);
    const std::vector<int> actions = {0, cfg.area(), 7}, depths = {1, 2, 5};
    session->recurrentInference(a.hidden.data(), actions.data(), depths.data(), B, a);   // hidden read from out itself
    net.recurrentInference(*iws, b.hidden.data(), actions.data(), depths.data(), B, b);
    assert(a.hidden == b.hidden && a.logits == b.logits && a.values == b.values && a.rewards == b.rewards);
    std::printf("test_network_model passed\n");
}

static void test_checkpoint_round_trip() {
    const NetConfig cfg = smallConfig(true);
    const int B = 3;
    NaiveBackend be;
    MuZeroNetwork net(cfg, 71);
    Rng rng(72);
    randomizeAffine(net, rng);
    const TrainBatch batch = randomBatch(cfg, B, cfg.unrollSteps, rng);
    TrainWorkspace ws(cfg, B, cfg.unrollSteps, be);
    net.trainStep(ws, batch);                 // every set of running statistics away from its initial values
    const std::string path = tempPath("round_trip.mzn");
    saveNetwork(net, path);
    MuZeroNetwork loaded = loadNetwork(path);
    std::filesystem::remove(path);
    assert(loaded.config() == cfg);
    std::vector<ParamView> pa = net.params(), pb = loaded.params();
    assert(pa.size() == pb.size());
    for (size_t t = 0; t < pa.size(); ++t)
        assert(pa[t].name == pb[t].name && std::equal(pa[t].value, pa[t].value + pa[t].size, pb[t].value));
    assert(buffersOf(net) == buffersOf(loaded));
    std::vector<BufferView> ba = net.buffers(), bb = loaded.buffers();
    for (size_t t = 0; t < ba.size(); ++t) assert(ba[t].name == bb[t].name);
    // Each step's set was updated from its own call: g's k1 and k3 differ.
    const float* dynK1 = nullptr;
    const float* dynK3 = nullptr;
    for (const BufferView& b : bb) {
        if (b.name == "dyn.block0.bn1.runMean.k1") dynK1 = b.value;
        if (b.name == "dyn.block0.bn1.runMean.k3") dynK3 = b.value;
    }
    assert(dynK1 && dynK3 && !std::equal(dynK1, dynK1 + cfg.filters, dynK3));

    auto ia = net.newInferWorkspace(B, be), ib = loaded.newInferWorkspace(B, be);
    ModelOutput a, b;
    net.initialInference(*ia, batch.observations.data(), B, a);
    loaded.initialInference(*ib, batch.observations.data(), B, b);
    assert(a.hidden == b.hidden && a.logits == b.logits && a.values == b.values);
    const std::vector<int> actions = {1, 2, cfg.area()}, depths = {1, 3, 2};
    net.recurrentInference(*ia, a.hidden.data(), actions.data(), depths.data(), B, a);
    loaded.recurrentInference(*ib, b.hidden.data(), actions.data(), depths.data(), B, b);
    assert(a.hidden == b.hidden && a.logits == b.logits && a.values == b.values && a.rewards == b.rewards);
    std::printf("test_checkpoint_round_trip passed\n");
}

static void test_pass_plane_network() {
    const NetConfig cfg = smallConfig(true, 2, true);
    assert(cfg.planes() == 2 * cfg.history + 3);
    assert(smallConfig(true, 2, false).planes() == 2 * cfg.history + 2);
    assert(!(cfg == smallConfig(true, 2, false)));
    const int B = 3;
    NaiveBackend be;
    MuZeroNetwork net(cfg, 91);
    Rng rng(92);
    const TrainBatch batch = randomBatch(cfg, B, cfg.unrollSteps, rng);
    assert(batch.observations.size() == static_cast<size_t>(B) * cfg.planes() * cfg.area());
    auto ws = net.newInferWorkspace(B, be);
    ModelOutput out;
    net.initialInference(*ws, batch.observations.data(), B, out);
    assert(out.values.size() == static_cast<size_t>(B));
    TrainWorkspace tws(cfg, B, cfg.unrollSteps, be);
    const UnrollLoss loss = net.trainStep(tws, batch);
    assert(std::isfinite(loss.total));
    for (const ParamView& p : net.params())
        if (p.name == "repr.stem.conv.w") assert(p.size == static_cast<size_t>(cfg.filters * cfg.planes() * 9));
    std::printf("test_pass_plane_network passed\n");
}

// Version 2 had no pass plane field: the header ended after unrollSteps.
static void test_checkpoint_pass_plane_and_v2() {
    const std::string path = tempPath("pass_plane.mzn");
    {   // round trip with the plane on
        const NetConfig cfg = smallConfig(true, 2, true);
        MuZeroNetwork net(cfg, 93);
        saveNetwork(net, path);
        MuZeroNetwork loaded = loadNetwork(path);
        assert(loaded.config() == cfg && loaded.config().passPlane);
        assert(loaded.config().planes() == 2 * cfg.history + 3);
    }
    {   // a version 2 file: same bytes as a v3 file minus the pass plane field, version number 2
        const NetConfig cfg = smallConfig(true, 2, false);
        MuZeroNetwork net(cfg, 94);
        saveNetwork(net, path);
        std::string bytes;
        {
            std::ifstream in(path, std::ios::binary);
            bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        }
        const size_t flagAt = 8 + 8 * 4;   // magic, version, then eight i32 header fields
        assert(bytes[flagAt] == 0 && bytes[flagAt + 1] == 0 && bytes[flagAt + 2] == 0 && bytes[flagAt + 3] == 0);
        bytes.erase(flagAt, 4);
        bytes[4] = 2;
        {
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        }
        MuZeroNetwork loaded = loadNetwork(path);
        assert(loaded.config() == cfg && !loaded.config().passPlane);
        std::vector<ParamView> pa = net.params(), pb = loaded.params();
        assert(pa.size() == pb.size());
        for (size_t t = 0; t < pa.size(); ++t) assert(std::equal(pa[t].value, pa[t].value + pa[t].size, pb[t].value));
    }
    std::filesystem::remove(path);
    std::printf("test_checkpoint_pass_plane_and_v2 passed\n");
}

static void test_corrupt_checkpoint() {
    MuZeroNetwork net(smallConfig(false), 81);
    const std::string path = tempPath("corrupt.mzn");
    saveNetwork(net, path);
    const auto size = std::filesystem::file_size(path);
    std::filesystem::resize_file(path, size / 2);
    bool threw = false;
    try {
        loadNetwork(path);
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find(path) != std::string::npos;
    }
    assert(threw);

    {   // bad magic
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out << "NOPE and some more bytes";
    }
    threw = false;
    try {
        loadNetwork(path);
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find(path) != std::string::npos;
    }
    assert(threw);

    {   // a version 1 checkpoint (one set of batch norm statistics per layer)
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        BinaryWriter w(out);
        w.u32(0x4E475A4D);
        w.u32(1);
        for (int v : {5, 2, 1, 1, 4, 4, 0}) w.i32(v);
        w.u32(0);
    }
    threw = false;
    try {
        loadNetwork(path);
    } catch (const std::runtime_error& e) {
        const std::string what = e.what();
        threw = what.find(path) != std::string::npos && what.find("version 1") != std::string::npos &&
                what.find("format changed") != std::string::npos;
    }
    assert(threw);
    std::filesystem::remove(path);
    std::printf("test_corrupt_checkpoint passed\n");
}

int main() {
    test_param_names();
    test_unroll_gradient();
    test_gradient_scale_half();
    test_inference_matches_probe();
    test_mixed_depths_bit_identical();
    test_unroll_steps_checked();
    test_pool_bit_identical();
    test_network_model();
    test_checkpoint_round_trip();
    test_pass_plane_network();
    test_checkpoint_pass_plane_and_v2();
    test_corrupt_checkpoint();
    std::printf("test_network: all tests passed\n");
    return 0;
}
