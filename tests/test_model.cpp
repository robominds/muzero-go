#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include "mzgo/model.hpp"
#include "mzgo/train_batch.hpp"

using namespace mzgo;

static void test_dims() {
    ModelDims d{5, 8};
    assert(d.area() == 25 && d.actions() == 26 && d.hiddenSize() == 200);
}

static void test_softmax() {
    const float logits[3] = {0.0f, 1.0f, 2.0f};
    float p[3];
    softmax(logits, 3, p);
    assert(std::fabs(p[0] + p[1] + p[2] - 1.0f) < 1e-6f);
    assert(p[0] < p[1] && p[1] < p[2]);
    assert(std::fabs(p[1] / p[0] - std::exp(1.0f)) < 1e-4f);

    const float big[2] = {1000.0f, 1001.0f};   // naive exp() would overflow
    float q[2];
    softmax(big, 2, q);
    assert(std::isfinite(q[0]) && std::isfinite(q[1]));
    assert(std::fabs(q[0] + q[1] - 1.0f) < 1e-6f);
    assert(std::fabs(q[0] - 0.26894142f) < 1e-5f);
}

static void test_train_batch_resize() {
    TrainBatch t;
    t.resize(3, 5, 4, 25, 26);
    assert(t.B == 3 && t.K == 5);
    assert(t.observations.size() == 3u * 4 * 25);
    assert(t.actions.size() == 3u * 5);
    assert(t.policyTargets.size() == 3u * 6 * 26);
    assert(t.policyMask.size() == 3u * 6);
    assert(t.valueTargets.size() == 3u * 6);
    assert(t.rewardTargets.size() == 3u * 6);
}

int main() {
    test_dims();
    test_softmax();
    test_train_batch_resize();
    std::printf("test_model: all tests passed\n");
    return 0;
}
