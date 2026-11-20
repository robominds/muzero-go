// Copied from alphazero-go (Mark Castelluccio), adapted for mzgo.
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include "mzgo/optimizer.hpp"

using namespace mzgo;

static bool near(double a, double b, double tol = 1e-5) { return std::fabs(a - b) <= tol; }

// A hand-made two-parameter list: a decayed weight tensor and an undecayed bias.
struct TwoParams {
    std::vector<float> wv{1.0f, 2.0f}, wg{0.5f, 0.0f}, bv{1.0f}, bg{0.5f};
    std::vector<ParamView> list() {
        return {{"w", wv.data(), wg.data(), wv.size(), true}, {"b", bv.data(), bg.data(), bv.size(), false}};
    }
};

static void test_momentum_and_weight_decay_hand_values() {
    TwoParams t;
    std::vector<ParamView> params = t.list();
    SgdMomentum opt(params, /*momentum=*/0.9f, /*weightDecay=*/1e-4f);
    opt.step(params, 0.1f);
    // Weight[0]: g' = 0.5 + 1e-4 * 1 = 0.5001; v = 0.5001; w = 1 - 0.05001 = 0.94999.
    // Weight[1]: g' = 0 + 1e-4 * 2 = 2e-4;     v = 2e-4;   w = 2 - 2e-5   = 1.99998.
    // Bias:      g' = 0.5 (no decay);          v = 0.5;    b = 1 - 0.05   = 0.95.
    assert(near(t.wv[0], 0.94999));
    assert(near(t.wv[1], 1.99998));
    assert(near(t.bv[0], 0.95));
    assert(near(opt.velocity()[0][0], 0.5001));
    assert(near(opt.velocity()[1][0], 0.5));

    opt.step(params, 0.1f);   // step() leaves the gradients in place
    // Weight[0]: g' = 0.5 + 1e-4 * 0.94999 = 0.500095; v = 0.9 * 0.5001 + 0.500095 = 0.950185;
    //            w = 0.94999 - 0.0950185 = 0.8549715.
    // Bias:      v = 0.9 * 0.5 + 0.5 = 0.95; b = 0.95 - 0.095 = 0.855.
    assert(near(t.wv[0], 0.8549715, 1e-6));
    assert(near(t.bv[0], 0.855, 1e-6));
    assert(near(opt.velocity()[1][0], 0.95));
}

static void test_velocity_shapes_and_restore() {
    TwoParams t;
    std::vector<ParamView> params = t.list();
    SgdMomentum opt(params, 0.9f, 1e-4f);
    assert(opt.velocity().size() == 2);
    assert(opt.velocity()[0].size() == 2 && opt.velocity()[1].size() == 1);
    for (const auto& v : opt.velocity())
        for (float x : v) assert(x == 0.0f);

    // A restored velocity makes the next step match the original optimizer's.
    opt.step(params, 0.1f);
    TwoParams copy = t;
    std::vector<ParamView> cp = copy.list();
    SgdMomentum restored(cp, 0.9f, 1e-4f);
    restored.velocity() = opt.velocity();
    opt.step(params, 0.1f);
    restored.step(cp, 0.1f);
    assert(t.wv == copy.wv && t.bv == copy.bv);
}

static void test_exponential_lr() {
    assert(exponentialLr(0.02f, 0.1f, 1000, 0) == 0.02f);
    assert(near(exponentialLr(0.02f, 0.1f, 1000, 1000), 0.002, 1e-7));
    assert(near(exponentialLr(0.02f, 0.1f, 1000, 500), 0.02 * std::pow(0.1, 0.5), 1e-7));
}

int main() {
    test_momentum_and_weight_decay_hand_values();
    test_velocity_shapes_and_restore();
    test_exponential_lr();
    std::printf("test_optimizer: all tests passed\n");
    return 0;
}
