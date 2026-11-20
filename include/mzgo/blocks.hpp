// Adapted from alphazero-go's network (Mark Castelluccio) for mzgo.
// The pieces MuZero's three functions are built from. Representation h and
// dynamics g are both a Trunk (a 3x3 stem conv, then residual blocks, as in
// AlphaZero) followed by min-max normalization of the hidden state;
// prediction f is AlphaZero's policy and value heads on top of a hidden
// state. The optional reward head has the value head's shape.
//
// Backprop through the unroll (BPTT) runs the SAME dynamics layers once per
// unroll step, so a layer's weights cannot hold the activations its backward
// needs: every call gets its own cache (TrunkCache, HeadCache, ...), filled
// by forward and read by backward, while the weights and their gradient
// accumulators stay in the layer. Gradients from the K calls simply add up
// in the accumulators. Scratch is per thread and shared by every call:
// buffers that only live inside one forward or backward.
//
// Layouts: like the layers, activations are channel-major [C][B][n*n].
#pragma once
#include <cstddef>
#include <string>
#include <vector>
#include "mzgo/backend.hpp"
#include "mzgo/layers.hpp"
#include "mzgo/params.hpp"
#include "mzgo/rng.hpp"

namespace mzgo {

class ThreadPool;   // mzgo/parallel.hpp

// How batch norm normalizes in a forward pass.
// Infer: running statistics (search; only reads the layers).
// Train: batch statistics, running statistics updated (a training step).
// Probe: batch statistics, running statistics untouched (loss evaluation,
//        gradient checks: the same outputs as Train without side effects).
// Calibrate: batch statistics, running statistics SET to them (momentum 1).
enum class BnMode { Infer, Train, Probe, Calibrate };

// Which set of running statistics every batch norm of a block uses (Infer)
// or updates (Train, Calibrate); Probe touches none. `set` serves the whole
// batch; in Infer mode, `rows` (if not null) gives one set per sample
// instead, so one batch can mix search depths. Blocks are built with
// `statSets` sets (see layers.hpp's BatchNorm2d); the network decides which
// call index uses which set (network.hpp).
struct BnSets {
    int set = 0;
    const int* rows = nullptr;   // [B], Infer only
};

// ---------------------------------------------------------------- min-max

// Hidden-state normalization (App. G): y = (x - min) / max(max - min, 1e-5)
// over one sample's n values, so every hidden state lies in [0, 1] whatever
// scale the dynamics drift to over an unroll.
struct MinMaxCache { int argmin, argmax; float min, range; bool floored; };
constexpr float kMinMaxFloor = 1e-5f;
// y may alias x (in-place normalization).
void minMaxForward(const float* x, int n, float* y, MinMaxCache& c);
// Accumulate-free: writes dx from dy. min and max are functions of x too, so
// the exact gradient is dy / range everywhere plus two correction terms, one
// on the argmin element and one on the argmax element. When floored (range
// < 1e-5, the denominator is the constant 1e-5), only the min term remains.
// dx may alias dy.
void minMaxBackward(const float* dy, const float* y, int n, const MinMaxCache& c, float* dx);

// ---------------------------------------------------------------- layouts

// [B][C][A] (one sample after another, as observations and hidden states
// are stored outside the network) <-> [C][B][A] (channel-major, as layers
// compute). One channel per chunk on the pool: plain copies, same bits.
void toChannelMajor(const float* src, int B, int C, int A, float* dst, ThreadPool* pool = nullptr);
void toSampleMajor(const float* src, int B, int C, int A, float* dst, ThreadPool* pool = nullptr);

// ---------------------------------------------------------------- scratch

// Per-thread buffers that live only inside one forward or backward, shared
// by every block on that thread. Sized for inputs of up to maxCin channels,
// `filters` channels inside, maxBatch samples on an n x n board.
struct Scratch {
    Scratch(int maxCin, int filters, int maxBatch, int n);
    std::vector<float> col, dcol;     // im2col columns and their gradient, any 3x3 conv
    std::vector<float> convOut;       // a conv's output before its batch norm
    std::vector<float> dA, dB;        // [filters][M] backward temporaries
};

// ---------------------------------------------------------------- trunk

// Stem conv3x3 + BN + ReLU (cin -> filters), then `blocks` residual blocks
// (conv-BN-ReLU-conv-BN, plus the block's input, ReLU): alphazero-go's trunk.
struct ResidualBlock {
    Conv2d conv1;
    BatchNorm2d bn1;
    Conv2d conv2;
    BatchNorm2d bn2;
};

// One call's activations, sized for (maxBatch, n). Holds a pointer to the
// thread's Scratch, which must outlive it.
struct TrunkCache {
    struct Block {
        std::vector<float> xhat1, invStd1, act1;   // bn1 cache; act1 = relu(bn1) = conv2 input
        std::vector<float> xhat2, invStd2, out;    // bn2 cache; out = relu(bn2 + skip)
    };
    TrunkCache(int filters, int blocks, int maxBatch, int n, Scratch& scratch);
    std::vector<float> stemXhat, stemInvStd, stemOut;
    std::vector<Block> blocks;
    Scratch* scratch;
    // The trunk's output [filters][B][A] after the last forward.
    const float* output() const { return blocks.empty() ? stemOut.data() : blocks.back().out.data(); }
};

struct Trunk {
    int cin = 0, filters = 0;
    Conv2d stemConv;
    BatchNorm2d stemBn;
    std::vector<ResidualBlock> blocks;
    Trunk() = default;
    Trunk(int cin, int filters, int blocks, int statSets = 1);
    void init(Rng& rng);              // He init, stem then blocks in order
    // x [cin][B][A] -> cache.output() [filters][B][A].
    void forward(const Backend& be, const float* x, int B, int n, TrunkCache& cache, BnMode mode,
                 ThreadPool* pool = nullptr, BnSets sets = {});
    // x: the forward's input, unchanged. dy: dL/d(output) [filters][B][A],
    // used as working space (overwritten). Accumulates the gradients; writes
    // dx [cin][B][A] (overwrite) unless dx == nullptr. dy and dx must not
    // be the scratch's buffers.
    void backward(const Backend& be, const float* x, float* dy, int B, int n, TrunkCache& cache,
                  float* dx, ThreadPool* pool = nullptr);
    // "<prefix>.stem.conv.w", "<prefix>.block0.bn1.gamma", ...
    std::vector<ParamView> params(const std::string& prefix);
    // Per batch norm, per set s: "<prefix>.stem.bn.runMean.k<firstK + s>",
    // then its runVar; firstK names set 0 by the call index it serves.
    std::vector<BufferView> buffers(const std::string& prefix, int firstK);
};

// ---------------------------------------------------------------- heads

// Value head (also the reward head's shape): 1x1 conv (filters -> 1), BN,
// ReLU, FC (area -> hidden), ReLU, FC (hidden -> 1), tanh.
struct ValueHeadCache {
    ValueHeadCache(int hidden, int maxBatch, int area);
    std::vector<float> xhat, invStd, act;   // act [1][B][A] = [B][A]
    std::vector<float> hidden;              // [B][hidden], after ReLU
    std::vector<float> values;              // [B], after tanh: the head's output
    std::vector<float> dHidden, dAct, dU;   // backward temporaries
};

struct ValueHead {
    Conv2d conv;
    BatchNorm2d bn;
    Linear fc1, fc2;
    ValueHead() = default;
    ValueHead(int filters, int area, int hidden, int statSets = 1);
    void init(Rng& rng);
    // x [filters][B][A] -> cache.values [B].
    void forward(const Backend& be, const float* x, int B, int n, ValueHeadCache& cache, BnMode mode,
                 ThreadPool* pool = nullptr, BnSets sets = {});
    // dValues: dL/d(values) [B] (after tanh). Writes dx [filters][B][A] (overwrite).
    void backward(const Backend& be, const float* x, const float* dValues, int B, int n,
                  ValueHeadCache& cache, float* dx, ThreadPool* pool = nullptr);
    // "<prefix>.conv.w", "<prefix>.bn.gamma", "<prefix>.fc1.w", ..., "<prefix>.fc2.b"
    std::vector<ParamView> params(const std::string& prefix);
    std::vector<BufferView> buffers(const std::string& prefix, int firstK);   // as Trunk's
};

// Prediction f: the policy head (1x1 conv -> 2, BN, ReLU, FC -> actions
// logits) and the value head, both reading the same hidden state.
struct HeadCache {
    HeadCache(int valueHidden, int maxBatch, int n, Scratch& scratch);
    std::vector<float> policyXhat, policyInvStd, policyAct;   // policyAct [2][B][A]
    std::vector<float> policyFlat;                            // [B][2A]
    std::vector<float> logits;                                // [B][actions]: the policy output
    std::vector<float> dPolicyFlat, dPolicyAct;               // backward temporaries
    ValueHeadCache value;                                     // value.values [B]: the value output
    Scratch* scratch;
};

struct PredictionHeads {
    int filters = 0, area = 0, actions = 0;
    Conv2d policyConv;
    BatchNorm2d policyBn;
    Linear policyFc;
    ValueHead value;
    PredictionHeads() = default;
    PredictionHeads(int filters, int n, int valueHidden, int statSets = 1);
    void init(Rng& rng);              // policy conv, policy fc, then the value head
    // x [filters][B][A] -> cache.logits, cache.value.values.
    void forward(const Backend& be, const float* x, int B, int n, HeadCache& cache, BnMode mode,
                 ThreadPool* pool = nullptr, BnSets sets = {});
    // dLogits [B][actions], dValues [B] (after tanh). Writes dx (overwrite),
    // the sum of both heads' input gradients; dx must not be a scratch buffer.
    void backward(const Backend& be, const float* x, const float* dLogits, const float* dValues, int B,
                  int n, HeadCache& cache, float* dx, ThreadPool* pool = nullptr);
    // "<prefix>.policy.conv.w", ..., "<prefix>.policy.fc.b", "<prefix>.value.conv.w", ...
    std::vector<ParamView> params(const std::string& prefix);
    std::vector<BufferView> buffers(const std::string& prefix, int firstK);   // as Trunk's
};

// dst[j] += src[j] for j < count, in fixed chunks on the pool.
void addInto(float* dst, const float* src, size_t count, ThreadPool* pool = nullptr);

} // namespace mzgo
