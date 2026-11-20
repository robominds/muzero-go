// Adapted from alphazero-go's network (Mark Castelluccio) for mzgo.
#include "mzgo/blocks.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include "mzgo/parallel.hpp"

namespace mzgo {

namespace {

// Samples per chunk of a per-sample loop on the pool.
constexpr size_t kSampleChunk = 16;

// Batch norm forward in the given mode (see BnMode).
void bnForward(BatchNorm2d& bn, const float* x, int M, float* y, float* xhat, float* invStd, BnMode mode,
               ThreadPool* pool) {
    switch (mode) {
        case BnMode::Infer: bn.forwardInfer(x, M, y, pool); break;
        case BnMode::Train: bn.forwardTrain(x, M, y, xhat, invStd, true, pool); break;
        case BnMode::Probe: bn.forwardTrain(x, M, y, xhat, invStd, false, pool); break;
        case BnMode::Calibrate: bn.forwardTrain(x, M, y, xhat, invStd, true, pool, 1.0f); break;
    }
}

void convParam(std::vector<ParamView>& out, const std::string& prefix, Conv2d& c) {
    out.push_back({prefix + ".w", c.w.data(), c.gw.data(), c.w.size(), true});
}
void bnParams(std::vector<ParamView>& out, const std::string& prefix, BatchNorm2d& b) {
    out.push_back({prefix + ".gamma", b.gamma.data(), b.ggamma.data(), b.gamma.size(), false});
    out.push_back({prefix + ".beta", b.beta.data(), b.gbeta.data(), b.beta.size(), false});
}
void linearParams(std::vector<ParamView>& out, const std::string& prefix, Linear& l) {
    out.push_back({prefix + ".w", l.w.data(), l.gw.data(), l.w.size(), true});
    out.push_back({prefix + ".b", l.b.data(), l.gb.data(), l.b.size(), false});
}
void bnBuffers(std::vector<BufferView>& out, const std::string& prefix, BatchNorm2d& b) {
    out.push_back({prefix + ".runMean", b.runMean.data(), b.runMean.size()});
    out.push_back({prefix + ".runVar", b.runVar.data(), b.runVar.size()});
}

} // namespace

void addInto(float* dst, const float* src, size_t count, ThreadPool* pool) {
    parallelRanges(pool, count, kElementChunk, [&](size_t begin, size_t end) {
        for (size_t j = begin; j < end; ++j) dst[j] += src[j];
    });
}

// ---------------------------------------------------------------- min-max

void minMaxForward(const float* x, int n, float* y, MinMaxCache& c) {
    assert(n >= 1);
    int lo = 0, hi = 0;
    for (int i = 1; i < n; ++i) {
        if (x[i] < x[lo]) lo = i;
        if (x[i] > x[hi]) hi = i;
    }
    c.argmin = lo;
    c.argmax = hi;
    c.min = x[lo];
    const float range = x[hi] - x[lo];
    c.floored = range < kMinMaxFloor;
    c.range = c.floored ? kMinMaxFloor : range;
    for (int i = 0; i < n; ++i) y[i] = (x[i] - c.min) / c.range;
}

void minMaxBackward(const float* dy, const float* y, int n, const MinMaxCache& c, float* dx) {
    // y_i = (x_i - m) / r with m = min x and r = max x - m. Differentiating:
    //   dy_i/dx_i = 1/r (every element, directly)
    //   dy_i/dm   = (y_i - 1)/r   -> lands on the argmin element
    //   dy_i/dmax = -y_i / r      -> lands on the argmax element
    // When floored, r is the constant 1e-5 and dy_i/dm = -1/r.
    double sumDy = 0.0, sumDyY = 0.0;
    for (int i = 0; i < n; ++i) {
        sumDy += dy[i];
        sumDyY += static_cast<double>(dy[i]) * y[i];
    }
    const double r = c.range;
    for (int i = 0; i < n; ++i) dx[i] = static_cast<float>(dy[i] / r);
    if (c.floored) {
        dx[c.argmin] = static_cast<float>(dx[c.argmin] - sumDy / r);
    } else {
        dx[c.argmin] = static_cast<float>(dx[c.argmin] + (sumDyY - sumDy) / r);
        dx[c.argmax] = static_cast<float>(dx[c.argmax] - sumDyY / r);
    }
}

// ---------------------------------------------------------------- layouts

void toChannelMajor(const float* src, int B, int C, int A, float* dst, ThreadPool* pool) {
    parallelChunks(pool, C, [&](int c) {
        for (int b = 0; b < B; ++b)
            std::copy(src + (static_cast<size_t>(b) * C + c) * A, src + (static_cast<size_t>(b) * C + c + 1) * A,
                      dst + (static_cast<size_t>(c) * B + b) * A);
    });
}

void toSampleMajor(const float* src, int B, int C, int A, float* dst, ThreadPool* pool) {
    parallelChunks(pool, C, [&](int c) {
        for (int b = 0; b < B; ++b)
            std::copy(src + (static_cast<size_t>(c) * B + b) * A, src + (static_cast<size_t>(c) * B + b + 1) * A,
                      dst + (static_cast<size_t>(b) * C + c) * A);
    });
}

// ---------------------------------------------------------------- scratch

Scratch::Scratch(int maxCin, int filters, int maxBatch, int n) {
    const size_t M = static_cast<size_t>(maxBatch) * n * n;
    // One im2col buffer serves every 3x3 conv on this thread (stems read
    // maxCin channels, blocks `filters`); backward rebuilds a conv's columns
    // right before using them.
    col.resize(static_cast<size_t>(std::max(maxCin, filters)) * 9 * M);
    dcol.resize(col.size());
    convOut.resize(std::max<size_t>(filters, 2) * M);   // the policy conv has 2 outputs even if filters == 1
    dA.resize(static_cast<size_t>(filters) * M);
    dB.resize(static_cast<size_t>(filters) * M);
}

// ---------------------------------------------------------------- trunk

Trunk::Trunk(int cin_, int filters_, int nblocks)
    : cin(cin_), filters(filters_), stemConv(cin_, filters_, 3), stemBn(filters_) {
    assert(cin >= 1 && filters >= 1 && nblocks >= 0);
    for (int i = 0; i < nblocks; ++i)
        blocks.push_back({Conv2d(filters, filters, 3), BatchNorm2d(filters), Conv2d(filters, filters, 3),
                          BatchNorm2d(filters)});
}

void Trunk::init(Rng& rng) {
    stemConv.init(rng);
    for (ResidualBlock& b : blocks) {
        b.conv1.init(rng);
        b.conv2.init(rng);
    }
}

TrunkCache::TrunkCache(int filters, int nblocks, int maxBatch, int n, Scratch& s) : scratch(&s) {
    const size_t F = static_cast<size_t>(filters);
    const size_t size = F * static_cast<size_t>(maxBatch) * n * n;
    stemXhat.resize(size);
    stemInvStd.resize(F);
    stemOut.resize(size);
    blocks.resize(nblocks);
    for (Block& b : blocks) {
        b.xhat1.resize(size);
        b.invStd1.resize(F);
        b.act1.resize(size);
        b.xhat2.resize(size);
        b.invStd2.resize(F);
        b.out.resize(size);
    }
}

void Trunk::forward(const Backend& be, const float* x, int B, int n, TrunkCache& c, BnMode mode,
                    ThreadPool* pool) {
    assert(c.blocks.size() == blocks.size());
    assert(static_cast<size_t>(filters) * B * n * n <= c.stemOut.size());
    const int M = B * n * n;
    const size_t size = static_cast<size_t>(filters) * M;
    float* col = c.scratch->col.data();
    float* convOut = c.scratch->convOut.data();

    // Stem: conv3x3 -> BN -> ReLU.
    stemConv.forward(be, x, B, n, col, convOut, pool);
    bnForward(stemBn, convOut, M, c.stemOut.data(), c.stemXhat.data(), c.stemInvStd.data(), mode, pool);
    reluForward(c.stemOut.data(), size, pool);

    // Residual blocks: conv -> BN -> ReLU -> conv -> BN -> + input -> ReLU.
    const float* in = c.stemOut.data();
    for (size_t i = 0; i < blocks.size(); ++i) {
        ResidualBlock& blk = blocks[i];
        TrunkCache::Block& buf = c.blocks[i];
        blk.conv1.forward(be, in, B, n, col, convOut, pool);
        bnForward(blk.bn1, convOut, M, buf.act1.data(), buf.xhat1.data(), buf.invStd1.data(), mode, pool);
        reluForward(buf.act1.data(), size, pool);
        blk.conv2.forward(be, buf.act1.data(), B, n, col, convOut, pool);
        bnForward(blk.bn2, convOut, M, buf.out.data(), buf.xhat2.data(), buf.invStd2.data(), mode, pool);
        addInto(buf.out.data(), in, size, pool);                 // the skip connection
        reluForward(buf.out.data(), size, pool);
        in = buf.out.data();
    }
}

void Trunk::backward(const Backend& be, const float* x, float* dy, int B, int n, TrunkCache& c, float* dx,
                     ThreadPool* pool) {
    const int M = B * n * n;
    const size_t size = static_cast<size_t>(filters) * M;
    float* col = c.scratch->col.data();
    float* dcol = c.scratch->dcol.data();
    float* dTmp = c.scratch->dA.data();
    float* dTmp2 = c.scratch->dB.data();

    // Residual blocks in reverse; dy holds dL/d(block output). The shared
    // col buffer was overwritten by later convs (and later calls), so each
    // 3x3 conv's columns are rebuilt from its saved input before its backward.
    for (size_t i = blocks.size(); i-- > 0;) {
        ResidualBlock& blk = blocks[i];
        TrunkCache::Block& buf = c.blocks[i];
        const float* in = i == 0 ? c.stemOut.data() : c.blocks[i - 1].out.data();
        reluBackward(buf.out.data(), dy, size, pool);            // now dL/d(bn2 output + skip)
        blk.bn2.backward(dy, buf.xhat2.data(), buf.invStd2.data(), M, dTmp, pool);
        im2colChannels(be, pool, buf.act1.data(), filters, B, n, 3, col);
        blk.conv2.backward(be, buf.act1.data(), col, dTmp, B, n, dcol, dTmp2, pool);
        reluBackward(buf.act1.data(), dTmp2, size, pool);
        blk.bn1.backward(dTmp2, buf.xhat1.data(), buf.invStd1.data(), M, dTmp2, pool);
        im2colChannels(be, pool, in, filters, B, n, 3, col);
        blk.conv1.backward(be, in, col, dTmp2, B, n, dcol, dTmp, pool);
        // The skip path passes dy to the block input unchanged; the conv
        // path adds its own contribution.
        addInto(dy, dTmp, size, pool);
    }

    // Stem.
    reluBackward(c.stemOut.data(), dy, size, pool);
    stemBn.backward(dy, c.stemXhat.data(), c.stemInvStd.data(), M, dy, pool);
    im2colChannels(be, pool, x, cin, B, n, 3, col);
    stemConv.backward(be, x, col, dy, B, n, dcol, dx, pool);
}

std::vector<ParamView> Trunk::params(const std::string& prefix) {
    std::vector<ParamView> out;
    convParam(out, prefix + ".stem.conv", stemConv);
    bnParams(out, prefix + ".stem.bn", stemBn);
    for (size_t i = 0; i < blocks.size(); ++i) {
        const std::string p = prefix + ".block" + std::to_string(i);
        convParam(out, p + ".conv1", blocks[i].conv1);
        bnParams(out, p + ".bn1", blocks[i].bn1);
        convParam(out, p + ".conv2", blocks[i].conv2);
        bnParams(out, p + ".bn2", blocks[i].bn2);
    }
    return out;
}

std::vector<BufferView> Trunk::buffers(const std::string& prefix) {
    std::vector<BufferView> out;
    bnBuffers(out, prefix + ".stem.bn", stemBn);
    for (size_t i = 0; i < blocks.size(); ++i) {
        const std::string p = prefix + ".block" + std::to_string(i);
        bnBuffers(out, p + ".bn1", blocks[i].bn1);
        bnBuffers(out, p + ".bn2", blocks[i].bn2);
    }
    return out;
}

// ---------------------------------------------------------------- value head

ValueHead::ValueHead(int filters, int area, int hidden)
    : conv(filters, 1, 1), bn(1), fc1(area, hidden), fc2(hidden, 1) {}

void ValueHead::init(Rng& rng) {
    conv.init(rng);
    fc1.init(rng);
    fc2.init(rng);
}

ValueHeadCache::ValueHeadCache(int hiddenSize, int maxBatch, int area) {
    const size_t M = static_cast<size_t>(maxBatch) * area;
    const size_t H = static_cast<size_t>(maxBatch) * hiddenSize;
    xhat.resize(M);
    invStd.resize(1);
    act.resize(M);
    hidden.resize(H);
    values.resize(maxBatch);
    dHidden.resize(H);
    dAct.resize(M);
    dU.resize(maxBatch);
}

void ValueHead::forward(const Backend& be, const float* x, int B, int n, ValueHeadCache& c, BnMode mode,
                        ThreadPool* pool) {
    const int M = B * n * n;
    // conv1x1 (F -> 1) -> BN -> ReLU -> Linear(A -> V) -> ReLU -> Linear(V -> 1) -> tanh.
    // A 1x1 conv needs no im2col; its output is written straight into act
    // and normalized in place (forwardTrain reads x before writing y).
    conv.forward(be, x, B, n, nullptr, c.act.data(), pool);
    bnForward(bn, c.act.data(), M, c.act.data(), c.xhat.data(), c.invStd.data(), mode, pool);
    reluForward(c.act.data(), M, pool);
    // With a single channel, [1][B][A] already is [B][A]: one row per sample.
    fc1.forward(be, c.act.data(), B, c.hidden.data());
    reluForward(c.hidden.data(), static_cast<size_t>(B) * fc1.out, pool);
    fc2.forward(be, c.hidden.data(), B, c.values.data());
    for (int b = 0; b < B; ++b) c.values[b] = std::tanh(c.values[b]);
}

void ValueHead::backward(const Backend& be, const float* x, const float* dValues, int B, int n,
                         ValueHeadCache& c, float* dx, ThreadPool* pool) {
    const int M = B * n * n;
    for (int b = 0; b < B; ++b) c.dU[b] = dValues[b] * (1.0f - c.values[b] * c.values[b]);   // tanh'
    fc2.backward(be, c.hidden.data(), c.dU.data(), B, c.dHidden.data());
    reluBackward(c.hidden.data(), c.dHidden.data(), static_cast<size_t>(B) * fc1.out, pool);
    fc1.backward(be, c.act.data(), c.dHidden.data(), B, c.dAct.data());
    reluBackward(c.act.data(), c.dAct.data(), M, pool);
    bn.backward(c.dAct.data(), c.xhat.data(), c.invStd.data(), M, c.dAct.data(), pool);
    conv.backward(be, x, nullptr, c.dAct.data(), B, n, nullptr, dx, pool);
}

std::vector<ParamView> ValueHead::params(const std::string& prefix) {
    std::vector<ParamView> out;
    convParam(out, prefix + ".conv", conv);
    bnParams(out, prefix + ".bn", bn);
    linearParams(out, prefix + ".fc1", fc1);
    linearParams(out, prefix + ".fc2", fc2);
    return out;
}

std::vector<BufferView> ValueHead::buffers(const std::string& prefix) {
    std::vector<BufferView> out;
    bnBuffers(out, prefix + ".bn", bn);
    return out;
}

// ---------------------------------------------------------------- prediction heads

PredictionHeads::PredictionHeads(int filters_, int n, int valueHidden)
    : filters(filters_), area(n * n), actions(n * n + 1), policyConv(filters_, 2, 1), policyBn(2),
      policyFc(2 * n * n, n * n + 1), value(filters_, n * n, valueHidden) {}

void PredictionHeads::init(Rng& rng) {
    policyConv.init(rng);
    policyFc.init(rng);
    value.init(rng);
}

HeadCache::HeadCache(int valueHidden, int maxBatch, int n, Scratch& s)
    : value(valueHidden, maxBatch, n * n), scratch(&s) {
    const size_t M = static_cast<size_t>(maxBatch) * n * n;
    policyXhat.resize(2 * M);
    policyInvStd.resize(2);
    policyAct.resize(2 * M);
    policyFlat.resize(2 * M);
    logits.resize(static_cast<size_t>(maxBatch) * (n * n + 1));
    dPolicyFlat.resize(2 * M);
    dPolicyAct.resize(2 * M);
}

void PredictionHeads::forward(const Backend& be, const float* x, int B, int n, HeadCache& c, BnMode mode,
                              ThreadPool* pool) {
    const int A = area, M = B * A;
    // Policy: conv1x1 (F -> 2) -> BN -> ReLU -> per-sample flatten -> Linear.
    policyConv.forward(be, x, B, n, nullptr, c.policyAct.data(), pool);
    bnForward(policyBn, c.policyAct.data(), M, c.policyAct.data(), c.policyXhat.data(), c.policyInvStd.data(),
              mode, pool);
    reluForward(c.policyAct.data(), 2 * static_cast<size_t>(M), pool);
    // [2][B][A] -> [B][2A], feature index ch * A + p.
    parallelRanges(pool, static_cast<size_t>(B), kSampleChunk, [&](size_t b0, size_t b1) {
        for (size_t b = b0; b < b1; ++b)
            for (int ch = 0; ch < 2; ++ch)
                std::copy(c.policyAct.data() + (static_cast<size_t>(ch) * B + b) * A,
                          c.policyAct.data() + (static_cast<size_t>(ch) * B + b + 1) * A,
                          c.policyFlat.data() + b * 2 * A + static_cast<size_t>(ch) * A);
    });
    policyFc.forward(be, c.policyFlat.data(), B, c.logits.data());

    value.forward(be, x, B, n, c.value, mode, pool);
}

void PredictionHeads::backward(const Backend& be, const float* x, const float* dLogits, const float* dValues,
                               int B, int n, HeadCache& c, float* dx, ThreadPool* pool) {
    const int A = area, M = B * A;
    const size_t size = static_cast<size_t>(filters) * M;
    float* dTmp = c.scratch->dA.data();

    value.backward(be, x, dValues, B, n, c.value, dx, pool);

    policyFc.backward(be, c.policyFlat.data(), dLogits, B, c.dPolicyFlat.data());
    parallelRanges(pool, static_cast<size_t>(B), kSampleChunk, [&](size_t b0, size_t b1) {
        for (size_t b = b0; b < b1; ++b)          // [B][2A] -> [2][B][A], undoing the flatten
            for (int ch = 0; ch < 2; ++ch)
                std::copy(c.dPolicyFlat.data() + b * 2 * A + static_cast<size_t>(ch) * A,
                          c.dPolicyFlat.data() + b * 2 * A + static_cast<size_t>(ch + 1) * A,
                          c.dPolicyAct.data() + (static_cast<size_t>(ch) * B + b) * A);
    });
    reluBackward(c.policyAct.data(), c.dPolicyAct.data(), 2 * static_cast<size_t>(M), pool);
    policyBn.backward(c.dPolicyAct.data(), c.policyXhat.data(), c.policyInvStd.data(), M, c.dPolicyAct.data(),
                      pool);
    policyConv.backward(be, x, nullptr, c.dPolicyAct.data(), B, n, nullptr, dTmp, pool);
    addInto(dx, dTmp, size, pool);                // the hidden state feeds both heads
}

std::vector<ParamView> PredictionHeads::params(const std::string& prefix) {
    std::vector<ParamView> out;
    convParam(out, prefix + ".policy.conv", policyConv);
    bnParams(out, prefix + ".policy.bn", policyBn);
    linearParams(out, prefix + ".policy.fc", policyFc);
    for (ParamView& p : value.params(prefix + ".value")) out.push_back(p);
    return out;
}

std::vector<BufferView> PredictionHeads::buffers(const std::string& prefix) {
    std::vector<BufferView> out;
    bnBuffers(out, prefix + ".policy.bn", policyBn);
    for (BufferView& b : value.buffers(prefix + ".value")) out.push_back(b);
    return out;
}

} // namespace mzgo
