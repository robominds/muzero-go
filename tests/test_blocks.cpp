#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include "mzgo/backend.hpp"
#include "mzgo/blocks.hpp"
#include "mzgo/rng.hpp"

using namespace mzgo;

namespace {

std::vector<float> randomVector(Rng& rng, size_t count, double lo = -1.0, double hi = 1.0) {
    std::vector<float> v(count);
    for (float& x : v) x = static_cast<float>(lo + (hi - lo) * rng.uniform());
    return v;
}

// L = sum_i r_i * y_i in double: its gradient with respect to y is exactly
// r, which is what backward is fed as dy.
double weightedSum(const std::vector<float>& r, const float* y) {
    double s = 0.0;
    for (size_t i = 0; i < r.size(); ++i) s += static_cast<double>(r[i]) * y[i];
    return s;
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

// One checked entry, with steps of 1e-2 down to 1e-4. A network's loss is
// only piecewise smooth: a step that straddles a ReLU switching (or a
// min-max argmin/argmax changing) makes the central difference average two
// slopes. true: some step's central difference matches. Asserts on a
// mismatch: over the three smallest steps the central differences agree
// with each other and, at each step, the forward and backward differences
// do too (no kink in sight), yet not with the analytic gradient -- a smooth
// region where the gradient is wrong. false otherwise (ill-conditioned):
// the caller re-draws another entry. The loss is summed in double from
// float outputs; its float noise (~1e-7) keeps steps above 1e-4.
template <typename LossFn>
bool checkEntry(const char* what, float* value, double analytic, LossFn loss) {
    std::vector<Differences> d;
    for (float h : {1e-2f, 3e-3f, 1e-3f, 3e-4f, 1e-4f}) {
        d.push_back(differences(value, h, loss));
        if (gradClose(analytic, d.back().central)) return true;
    }
    bool smooth = true;
    for (size_t s = d.size() - 3; s < d.size(); ++s)
        smooth = smooth && gradClose(d[s].forward, d[s].backward) && gradClose(d[s].central, d.back().central);
    if (smooth) {
        std::printf("%s: analytic %.7f does not match finite differences %.7f\n", what, analytic, d.back().central);
        assert(false);
    }
    return false;
}

// Checks `count` random entries of the given tensors (value, analytic
// gradient snapshot). An ill-conditioned entry is re-drawn; re-draws must
// stay a minority.
struct Tensor {
    std::string name;
    float* value;
    std::vector<float> grad;   // analytic, snapshot before any probe
};

template <typename LossFn>
void checkRandomEntries(std::vector<Tensor>& tensors, int count, Rng& rng, LossFn loss) {
    int checked = 0, redrawn = 0;
    while (checked < count) {
        // A random tensor, then a random entry of it: small tensors (batch
        // norm, biases) get checked as often as the big weight matrices.
        const size_t t = static_cast<size_t>(rng.below(static_cast<int>(tensors.size())));
        const size_t i = static_cast<size_t>(rng.below(static_cast<int>(tensors[t].grad.size())));
        const std::string what = tensors[t].name + "[" + std::to_string(i) + "]";
        if (checkEntry(what.c_str(), tensors[t].value + i, tensors[t].grad[i], loss)) {
            ++checked;
        } else {
            ++redrawn;
            assert(redrawn <= count);
        }
    }
}

std::vector<Tensor> paramTensors(const std::vector<ParamView>& params) {
    std::vector<Tensor> out;
    for (const ParamView& p : params) out.push_back({p.name, p.value, std::vector<float>(p.grad, p.grad + p.size)});
    return out;
}

// Random batch norm affine parameters and linear biases: gamma != 1 and
// beta != 0 exercise their gradients and move every ReLU's kink away from
// the channel mean, where pre-activations are densest; nonzero biases push
// the tanh outputs away from 0, where tanh' = 1 - v^2 is ~1 and a missing
// tanh' would hide.
void randomizeAffine(const std::vector<ParamView>& params, Rng& rng) {
    for (const ParamView& p : params) {
        const bool isGamma = p.name.find(".gamma") != std::string::npos;
        const bool isBeta = p.name.find(".beta") != std::string::npos;
        const bool isBias = p.name.size() > 2 && p.name.compare(p.name.size() - 2, 2, ".b") == 0;
        for (size_t i = 0; i < p.size; ++i) {
            if (isGamma) p.value[i] = static_cast<float>(0.5 + rng.uniform());
            if (isBeta) p.value[i] = static_cast<float>(rng.uniform() - 0.5);
            if (isBias) p.value[i] = static_cast<float>(2.0 * rng.uniform() - 1.0);
        }
    }
}

} // namespace

static void test_minmax_forward() {
    const float x[3] = {3.0f, 1.0f, 2.0f};
    float y[3];
    MinMaxCache c;
    minMaxForward(x, 3, y, c);
    assert(y[0] == 1.0f && y[1] == 0.0f && y[2] == 0.5f);
    assert(c.argmin == 1 && c.argmax == 0 && c.min == 1.0f && c.range == 2.0f && !c.floored);
    std::printf("test_minmax_forward passed\n");
}

static void test_minmax_gradient() {
    Rng rng(7);
    const int n = 50;
    const double eps = 1e-3;
    for (int trial = 0; trial < 20; ++trial) {
        // A unique min and max, with a margin wider than the probe, so the
        // argmin and argmax do not change under a perturbation of eps.
        std::vector<float> x;
        for (;;) {
            x = randomVector(rng, n);
            std::vector<float> sorted = x;
            std::sort(sorted.begin(), sorted.end());
            if (sorted[1] - sorted[0] > 10 * eps && sorted[n - 1] - sorted[n - 2] > 10 * eps) break;
        }
        const std::vector<float> r = randomVector(rng, n);
        std::vector<float> y(n), dx(n);
        MinMaxCache c;
        minMaxForward(x.data(), n, y.data(), c);
        minMaxBackward(r.data(), y.data(), n, c, dx.data());
        auto loss = [&](const std::vector<float>& in) {
            std::vector<float> out(n);
            MinMaxCache cc;
            minMaxForward(in.data(), n, out.data(), cc);
            return weightedSum(r, out.data());
        };
        for (int i = 0; i < n; ++i) {
            std::vector<float> up = x, down = x;
            up[i] += static_cast<float>(eps);
            down[i] -= static_cast<float>(eps);
            const double numeric = (loss(up) - loss(down)) / (static_cast<double>(up[i]) - down[i]);
            // Relative error, against the gradient's size or 0.1 for an
            // entry whose gradient happens to be near zero.
            const double rel = std::fabs(numeric - dx[i]) / std::max({std::fabs(numeric), std::fabs(dx[i]) * 1.0, 0.1});
            assert(rel < 1e-2);
        }
    }
    std::printf("test_minmax_gradient passed\n");
}

static void test_minmax_constant_input() {
    const int n = 10;
    std::vector<float> x(n, 0.7f), y(n), dx(n);
    MinMaxCache c;
    minMaxForward(x.data(), n, y.data(), c);
    assert(c.floored);
    for (float v : y) assert(std::isfinite(v) && v == 0.0f);
    Rng rng(3);
    const std::vector<float> dy = randomVector(rng, n);
    minMaxBackward(dy.data(), y.data(), n, c, dx.data());
    double sum = 0.0;
    for (float v : dx) {
        assert(std::isfinite(v));
        sum += v;
    }
    // y does not change when every x moves by the same amount, so the
    // gradient sums to zero (the min term cancels the direct one).
    double scale = 0.0;
    for (float v : dx) scale = std::max(scale, std::fabs(static_cast<double>(v)));
    assert(std::fabs(sum) <= 1e-5 * scale);
    std::printf("test_minmax_constant_input passed\n");
}

static void test_trunk_gradient() {
    const int n = 5, A = n * n, B = 2, cin = 3, F = 4;
    const size_t M = static_cast<size_t>(B) * A;
    NaiveBackend be;
    Rng rng(11);
    Trunk trunk(cin, F, 1);
    trunk.init(rng);
    std::vector<ParamView> params = trunk.params("t");
    randomizeAffine(params, rng);
    Scratch scratch(cin, F, B, n);
    TrunkCache cache(F, 1, B, n, scratch);
    std::vector<float> x = randomVector(rng, cin * M);
    const std::vector<float> r = randomVector(rng, F * M);

    for (ParamView& p : params) std::fill(p.grad, p.grad + p.size, 0.0f);
    trunk.forward(be, x.data(), B, n, cache, BnMode::Probe);
    std::vector<float> dy = r, dx(cin * M);
    trunk.backward(be, x.data(), dy.data(), B, n, cache, dx.data());

    auto loss = [&] {
        trunk.forward(be, x.data(), B, n, cache, BnMode::Probe);
        return weightedSum(r, cache.output());
    };
    std::vector<Tensor> paramTs = paramTensors(params);
    checkRandomEntries(paramTs, 20, rng, loss);
    std::vector<Tensor> inputTs = {{"x", x.data(), dx}};
    checkRandomEntries(inputTs, 20, rng, loss);
    std::printf("test_trunk_gradient passed\n");
}

static void test_heads_gradient() {
    const int n = 5, A = n * n, B = 2, F = 4, V = 4;
    const size_t M = static_cast<size_t>(B) * A;
    NaiveBackend be;
    Rng rng(12);
    PredictionHeads heads(F, n, V);
    heads.init(rng);
    std::vector<ParamView> params = heads.params("pred");
    randomizeAffine(params, rng);
    Scratch scratch(F, F, B, n);
    HeadCache cache(V, B, n, scratch);
    std::vector<float> x = randomVector(rng, F * M, 0.0, 1.0);   // a normalized hidden state
    const std::vector<float> rLogits = randomVector(rng, static_cast<size_t>(B) * (A + 1));
    const std::vector<float> rValues = randomVector(rng, B);

    for (ParamView& p : params) std::fill(p.grad, p.grad + p.size, 0.0f);
    heads.forward(be, x.data(), B, n, cache, BnMode::Probe);
    std::vector<float> dx(F * M);
    heads.backward(be, x.data(), rLogits.data(), rValues.data(), B, n, cache, dx.data());

    auto loss = [&] {
        heads.forward(be, x.data(), B, n, cache, BnMode::Probe);
        return weightedSum(rLogits, cache.logits.data()) + weightedSum(rValues, cache.value.values.data());
    };
    std::vector<Tensor> paramTs = paramTensors(params);
    checkRandomEntries(paramTs, 20, rng, loss);
    std::vector<Tensor> inputTs = {{"x", x.data(), dx}};
    checkRandomEntries(inputTs, 20, rng, loss);
    std::printf("test_heads_gradient passed\n");
}

int main() {
    test_minmax_forward();
    test_minmax_gradient();
    test_minmax_constant_input();
    test_trunk_gradient();
    test_heads_gradient();
    std::printf("test_blocks: all tests passed\n");
    return 0;
}
