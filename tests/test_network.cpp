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

using namespace mzgo;

namespace {

NetConfig smallConfig(bool rewardHead) {
    NetConfig cfg;
    cfg.boardSize = 5;
    cfg.history = 2;
    cfg.reprBlocks = 1;
    cfg.dynBlocks = 1;
    cfg.filters = 4;
    cfg.valueHidden = 4;
    cfg.rewardHead = rewardHead;
    return cfg;
}

// A random unroll batch: observation planes in {0, 1}, random actions (some
// of them passes), normalized random policy targets with every mask 1,
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
        t.policyMask[i] = 1.0f;
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
        const size_t bnLayers = (1 + 2) + (1 + 2) + 2 + (reward ? 1 : 0);
        assert(net.buffers().size() == 2 * bnLayers);
    }
    std::printf("test_param_names passed\n");
}

// trainStep(..., 1.0) against central differences of computeLoss().total.
static void checkUnrollGradient(bool rewardHead) {
    const NetConfig cfg = smallConfig(rewardHead);
    const int B = 3, K = 2;
    NaiveBackend be;
    MuZeroNetwork net(cfg, 21);
    Rng rng(22);
    randomizeAffine(net, rng);
    TrainBatch batch = randomBatch(cfg, B, K, rng);
    batch.policyMask[1 * (K + 1) + 2] = 0.0f;           // sample 1 is past the game's end at k = 2
    batch.actions[2 * K + 0] = cfg.area();               // sample 2 passes first
    TrainWorkspace ws(cfg, B, K, be);

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

// K = 1: the gradient entering s0 is f's plus scale * g's input gradient,
// so the representation's gradient is affine in the scale and nothing else
// depends on it.
static void test_gradient_scale_half() {
    const NetConfig cfg = smallConfig(false);
    const int B = 3, K = 1;
    NaiveBackend be;
    MuZeroNetwork net(cfg, 31);
    Rng rng(32);
    randomizeAffine(net, rng);
    const TrainBatch batch = randomBatch(cfg, B, K, rng);
    TrainWorkspace ws(cfg, B, K, be);
    net.trainStep(ws, batch, 0.0f);
    const auto g0 = gradsOf(net);
    net.trainStep(ws, batch, 0.5f);
    const auto gHalf = gradsOf(net);
    net.trainStep(ws, batch, 1.0f);
    const auto g1 = gradsOf(net);

    const std::vector<ParamView> params = net.params();
    bool reprDiffers = false;
    for (size_t t = 0; t < params.size(); ++t) {
        if (params[t].name.compare(0, 5, "repr.") != 0) {
            assert(g0[t] == gHalf[t] && g1[t] == gHalf[t]);   // bit for bit
            continue;
        }
        // Relative to the tensor's largest gradient: an entry that is a near
        // cancellation of f's and g's contributions carries the rounding of
        // both terms, which is relative to them, not to the small result.
        double scale = 0.0;
        for (size_t i = 0; i < params[t].size; ++i)
            scale = std::max({scale, std::fabs(static_cast<double>(g0[t][i])), std::fabs(static_cast<double>(g1[t][i]))});
        for (size_t i = 0; i < params[t].size; ++i) {
            const double mean = (static_cast<double>(g0[t][i]) + g1[t][i]) / 2;
            assert(std::fabs(gHalf[t][i] - mean) <= 1e-5 * scale);
            if (std::fabs(static_cast<double>(g1[t][i]) - g0[t][i]) > 1e-3 * scale) reprDiffers = true;
        }
    }
    assert(reprDiffers);
    std::printf("test_gradient_scale_half passed\n");
}

// After calibrateBatchNorm on a batch, inference mode on that batch
// computes what the training-mode (batch statistics) forward computed.
// Two corrections make "the same" exact:
// - Calibration stores the unbiased variance (M / (M - 1) times the batch's,
//   what the running average estimates), while the training forward
//   normalizes with the biased one; the test rescales runVar by (M - 1) / M.
// - A layer called several times keeps its last call's statistics: with
//   K = 1, f's are those of step 1, so the recurrent inference is compared
//   after a K = 1 calibration (s0, then s1, p1, v1, r1) and the initial
//   inference's policy and value after a K = 0 one.
static void test_inference_matches_probe() {
    const NetConfig cfg = smallConfig(true);
    const int B = 4, F = cfg.filters, A = cfg.area();
    const double M = static_cast<double>(B) * A;
    NaiveBackend be;
    MuZeroNetwork net(cfg, 41);
    Rng rng(42);
    randomizeAffine(net, rng);
    const TrainBatch batch = randomBatch(cfg, B, 1, rng);
    auto unbiasToBiased = [&] {
        for (const BufferView& b : net.buffers())
            if (b.name.find(".runVar") != std::string::npos)
                for (size_t i = 0; i < b.size; ++i) b.value[i] = static_cast<float>(b.value[i] * ((M - 1) / M));
    };
    const size_t hiddenCount = static_cast<size_t>(B) * F * A;
    auto iws = net.newInferWorkspace(B, be);

    // K = 1.
    TrainWorkspace ws(cfg, B, 1, be);
    net.calibrateBatchNorm(ws, batch);
    unbiasToBiased();
    ModelOutput out0, out1;
    net.initialInference(*iws, batch.observations.data(), B, out0);
    assert(out0.hidden.size() == hiddenCount && out0.rewards == std::vector<float>(B, 0.0f));
    assertNear(out0.hidden, ws.hidden(0), hiddenCount, 1e-4);
    std::vector<int> actions(B);
    for (int b = 0; b < B; ++b) actions[b] = batch.actions[b * 1 + 0];
    net.recurrentInference(*iws, out0.hidden.data(), actions.data(), B, out1);
    assertNear(out1.hidden, ws.hidden(1), hiddenCount, 1e-4);
    assertNear(out1.logits, ws.logits(1), static_cast<size_t>(B) * cfg.actions(), 1e-4);
    assertNear(out1.values, ws.values(1), B, 1e-4);
    assertNear(out1.rewards, ws.rewards(1), B, 1e-4);
    for (float v : out1.hidden) assert(v >= 0.0f && v <= 1.0f);

    // K = 0: the same observations, f last called on s0.
    TrainBatch initial = batch;
    initial.resize(B, 0, cfg.planes(), A, cfg.actions());
    initial.observations = batch.observations;
    TrainWorkspace ws0(cfg, B, 0, be);
    net.calibrateBatchNorm(ws0, initial);
    unbiasToBiased();
    net.initialInference(*iws, batch.observations.data(), B, out0);
    assertNear(out0.hidden, ws0.hidden(0), hiddenCount, 1e-4);
    assertNear(out0.logits, ws0.logits(0), static_cast<size_t>(B) * cfg.actions(), 1e-4);
    assertNear(out0.values, ws0.values(0), B, 1e-4);
    std::printf("test_inference_matches_probe passed\n");
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
    const std::vector<int> actions = {0, cfg.area(), 7};
    session->recurrentInference(a.hidden.data(), actions.data(), B, a);   // hidden read from out itself
    net.recurrentInference(*iws, b.hidden.data(), actions.data(), B, b);
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
    const TrainBatch batch = randomBatch(cfg, B, 2, rng);
    TrainWorkspace ws(cfg, B, 2, be);
    net.trainStep(ws, batch);                 // running statistics away from their initial values
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

    auto ia = net.newInferWorkspace(B, be), ib = loaded.newInferWorkspace(B, be);
    ModelOutput a, b;
    net.initialInference(*ia, batch.observations.data(), B, a);
    loaded.initialInference(*ib, batch.observations.data(), B, b);
    assert(a.hidden == b.hidden && a.logits == b.logits && a.values == b.values);
    const std::vector<int> actions = {1, 2, cfg.area()};
    net.recurrentInference(*ia, a.hidden.data(), actions.data(), B, a);
    loaded.recurrentInference(*ib, b.hidden.data(), actions.data(), B, b);
    assert(a.hidden == b.hidden && a.logits == b.logits && a.values == b.values && a.rewards == b.rewards);
    std::printf("test_checkpoint_round_trip passed\n");
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
    std::filesystem::remove(path);
    std::printf("test_corrupt_checkpoint passed\n");
}

int main() {
    test_param_names();
    test_unroll_gradient();
    test_gradient_scale_half();
    test_inference_matches_probe();
    test_pool_bit_identical();
    test_network_model();
    test_checkpoint_round_trip();
    test_corrupt_checkpoint();
    std::printf("test_network: all tests passed\n");
    return 0;
}
