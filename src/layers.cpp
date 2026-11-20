// Copied from alphazero-go (Mark Castelluccio), adapted for mzgo.
#include "mzgo/layers.hpp"
#include "mzgo/parallel.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>

namespace mzgo {

namespace {

// He initialization: weights ~ N(0, 2 / fanIn) keep the variance of a ReLU
// layer's output close to its input's, so a deep stack starts neither
// exploding nor vanishing.
void heNormal(std::vector<float>& w, int fanIn, Rng& rng) {
    const double stddev = std::sqrt(2.0 / fanIn);
    for (float& v : w) v = static_cast<float>(rng.normal() * stddev);
}

} // namespace

// ---------------------------------------------------------------- Conv2d

Conv2d::Conv2d(int cin_, int cout_, int k_) : cin(cin_), cout(cout_), k(k_) {
    assert(cin > 0 && cout > 0 && (k == 1 || k == 3));
    w.assign(static_cast<size_t>(cout) * cin * k * k, 0.0f);
    gw.assign(w.size(), 0.0f);
}

void Conv2d::init(Rng& rng) { heNormal(w, cin * k * k, rng); }

size_t Conv2d::colSize(int B, int n) const {
    return k == 1 ? 0 : static_cast<size_t>(cin) * k * k * B * n * n;
}

void Conv2d::forward(const Backend& be, const float* x, int B, int n, float* col, float* y,
                     ThreadPool* pool) const {
    const int M = B * n * n;
    // A 1x1 convolution needs no unfolding: x [cin][M] already is the column matrix.
    const float* columns = x;
    if (k != 1) {
        im2colChannels(be, pool, x, cin, B, n, k, col);
        columns = col;
    }
    // y [cout][M] = w [cout][cin*k*k] . columns [cin*k*k][M]
    gemmColumns(be, pool, false, false, cout, M, cin * k * k, 1.0f, w.data(), columns, 0.0f, y);
}

void Conv2d::backward(const Backend& be, const float* x, const float* col, const float* dy,
                      int B, int n, float* dcol, float* dx, ThreadPool* pool) {
    const int M = B * n * n;
    const int K = cin * k * k;
    const float* columns = k == 1 ? x : col;
    // dL/dx = w^T . dy: for a 1x1 conv directly (the columns ARE x); for a
    // 3x3 conv into dcol, whose columns col2im then folds back onto the
    // pixels they were copied from (summing overlapping patches).
    float* dIn = k == 1 ? dx : dcol;
    const int rows = k == 1 ? cin : K;
    const int blocks = dx == nullptr ? 0 : (pool ? gemmColumnBlockCount(rows, M) : 1);
    // y = w . columns, so dL/dw = dy . columns^T (summed over all M pixels),
    // accumulated into gw (beta = 1). It is one call, chunk 0 of the run
    // below beside the column blocks of dL/dx, unless the backend can cut it
    // into row bands on the pool itself. Without a pool: the two gemms in
    // that order.
    const bool dWDone = pool != nullptr &&
                        be.gemmRowsOnPool(*pool, false, true, cout, K, M, 1.0f, dy, columns, 1.0f, gw.data());
    const int first = dWDone ? 0 : 1;
    parallelChunks(pool, first + blocks, [&](int chunk) {
        if (chunk < first) {
            be.gemm(false, true, cout, K, M, 1.0f, dy, columns, 1.0f, gw.data());
        } else {
            gemmColumnBlock(be, chunk - first, blocks, true, false, rows, M, cout, 1.0f, w.data(), dy, 0.0f, dIn);
        }
    });
    if (dx != nullptr && k != 1) col2imChannels(be, pool, dcol, cin, B, n, k, dx);
}

// ---------------------------------------------------------------- BatchNorm2d

BatchNorm2d::BatchNorm2d(int c_, int sets_)
    : c(c_), sets(sets_), gamma(c_, 1.0f), beta(c_, 0.0f), ggamma(c_, 0.0f), gbeta(c_, 0.0f),
      runMean(static_cast<size_t>(sets_) * c_, 0.0f), runVar(static_cast<size_t>(sets_) * c_, 1.0f) {
    assert(c > 0 && sets > 0);
}

namespace {

// Inference batch norm of one channel folded into y = scale * x + shift.
// forwardInfer and forwardInferRows both go through these two helpers, so a
// value gets the same bits from either.
inline void inferAffine(const BatchNorm2d& bn, int ch, int set, float& scale, float& shift) {
    const size_t i = static_cast<size_t>(set) * bn.c + ch;
    scale = bn.gamma[ch] / std::sqrt(bn.runVar[i] + BatchNorm2d::kEps);
    shift = bn.beta[ch] - scale * bn.runMean[i];
}
inline void applyAffine(const float* x, size_t count, float scale, float shift, float* y) {
    for (size_t i = 0; i < count; ++i) y[i] = scale * x[i] + shift;
}

} // namespace

// Each channel is one chunk: its statistics, sums and outputs involve no
// other channel, so the loops below run unchanged whichever thread has it.

void BatchNorm2d::forwardInfer(const float* x, int M, float* y, ThreadPool* pool, int set) const {
    assert(set >= 0 && set < sets);
    parallelChunks(pool, c, [&](int ch) {
        // Fold the fixed statistics into one scale and shift per channel.
        float scale, shift;
        inferAffine(*this, ch, set, scale, shift);
        applyAffine(x + static_cast<size_t>(ch) * M, static_cast<size_t>(M), scale, shift,
                    y + static_cast<size_t>(ch) * M);
    });
}

void BatchNorm2d::forwardInferRows(const float* x, int B, int A, const int* rowSets, float* y,
                                   ThreadPool* pool) const {
    parallelChunks(pool, c, [&](int ch) {
        for (int b = 0; b < B; ++b) {
            assert(rowSets[b] >= 0 && rowSets[b] < sets);
            float scale, shift;
            inferAffine(*this, ch, rowSets[b], scale, shift);
            const size_t offset = (static_cast<size_t>(ch) * B + b) * A;
            applyAffine(x + offset, static_cast<size_t>(A), scale, shift, y + offset);
        }
    });
}

void BatchNorm2d::forwardTrain(const float* x, int M, float* y, float* xhat, float* invStd,
                               bool updateRunning, ThreadPool* pool, float momentum, int set) {
    assert(M > 1 && set >= 0 && set < sets);
    parallelChunks(pool, c, [&](int ch) {
        const float* xc = x + static_cast<size_t>(ch) * M;
        // Two passes in double: mean first, then the variance around it
        // (the one-pass E[x^2] - E[x]^2 loses precision when the mean is large).
        double sum = 0.0;
        for (int i = 0; i < M; ++i) sum += xc[i];
        const double mean = sum / M;
        double sq = 0.0;
        for (int i = 0; i < M; ++i) sq += (xc[i] - mean) * (xc[i] - mean);
        const double var = sq / M;                                  // biased: used to normalize
        const float inv = static_cast<float>(1.0 / std::sqrt(var + kEps));
        invStd[ch] = inv;

        float* xh = xhat + static_cast<size_t>(ch) * M;
        float* yc = y + static_cast<size_t>(ch) * M;
        const float m = static_cast<float>(mean);
        for (int i = 0; i < M; ++i) {
            const float h = (xc[i] - m) * inv;                      // read x before y (they may alias)
            xh[i] = h;
            yc[i] = gamma[ch] * h + beta[ch];
        }
        if (updateRunning) {
            // Inference sees one position at a time, so it needs population
            // estimates: unbiased variance, exponential moving averages.
            const double unbiased = var * M / (M - 1);
            const size_t i = static_cast<size_t>(set) * c + ch;
            runMean[i] = (1.0f - momentum) * runMean[i] + momentum * static_cast<float>(mean);
            runVar[i] = (1.0f - momentum) * runVar[i] + momentum * static_cast<float>(unbiased);
        }
    });
}

void BatchNorm2d::backward(const float* dy, const float* xhat, const float* invStd, int M,
                           float* dx, ThreadPool* pool) {
    parallelChunks(pool, c, [&](int ch) {
        const float* dyc = dy + static_cast<size_t>(ch) * M;
        const float* xh = xhat + static_cast<size_t>(ch) * M;
        double sumDy = 0.0, sumDyXhat = 0.0;
        for (int i = 0; i < M; ++i) {
            sumDy += dyc[i];
            sumDyXhat += static_cast<double>(dyc[i]) * xh[i];
        }
        ggamma[ch] += static_cast<float>(sumDyXhat);                // y = gamma * xhat + beta
        gbeta[ch] += static_cast<float>(sumDy);
        // Every x in the channel moved the batch mean and variance, so each
        // dx collects a share of every dy:
        //   dx = gamma * invStd / M * (M * dy - sum(dy) - xhat * sum(dy * xhat))
        const float scale = gamma[ch] * invStd[ch] / static_cast<float>(M);
        const float fM = static_cast<float>(M);
        const float sDy = static_cast<float>(sumDy), sDyXhat = static_cast<float>(sumDyXhat);
        float* dxc = dx + static_cast<size_t>(ch) * M;
        for (int i = 0; i < M; ++i) dxc[i] = scale * (fM * dyc[i] - sDy - xh[i] * sDyXhat);
    });
}

// ---------------------------------------------------------------- Linear

Linear::Linear(int in_, int out_)
    : in(in_), out(out_), w(static_cast<size_t>(in_) * out_, 0.0f), b(out_, 0.0f),
      gw(w.size(), 0.0f), gb(out_, 0.0f) {
    assert(in > 0 && out > 0);
}

void Linear::init(Rng& rng) {
    heNormal(w, in, rng);
    std::fill(b.begin(), b.end(), 0.0f);
}

void Linear::forward(const Backend& be, const float* x, int B, float* y) const {
    // y [B][out] = x [B][in] . w^T (w is stored [out][in]), then + b per row.
    be.gemm(false, true, B, out, in, 1.0f, x, w.data(), 0.0f, y);
    for (int s = 0; s < B; ++s) {
        float* row = y + static_cast<size_t>(s) * out;
        for (int o = 0; o < out; ++o) row[o] += b[o];
    }
}

void Linear::backward(const Backend& be, const float* x, const float* dy, int B, float* dx) {
    // gw [out][in] += dy^T [out][B] . x [B][in]
    be.gemm(true, false, out, in, B, 1.0f, dy, x, 1.0f, gw.data());
    for (int s = 0; s < B; ++s) {
        const float* row = dy + static_cast<size_t>(s) * out;
        for (int o = 0; o < out; ++o) gb[o] += row[o];
    }
    // dx [B][in] = dy [B][out] . w [out][in]
    if (dx != nullptr) be.gemm(false, false, B, in, out, 1.0f, dy, w.data(), 0.0f, dx);
}

// ---------------------------------------------------------------- ReLU

void reluForward(float* x, size_t count, ThreadPool* pool) {
    parallelRanges(pool, count, kElementChunk, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) x[i] = x[i] > 0.0f ? x[i] : 0.0f;
    });
}

void reluBackward(const float* y, float* dy, size_t count, ThreadPool* pool) {
    // relu'(x) is 1 where the output is positive, 0 elsewhere; the output
    // tells us which, so the pre-activation need not be kept. Loading dy
    // unconditionally lets the compiler turn the choice into a vector select
    // (as a conditional load it stayed a mispredicted branch per element).
    parallelRanges(pool, count, kElementChunk, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            const float g = dy[i];
            dy[i] = y[i] > 0.0f ? g : 0.0f;
        }
    });
}

} // namespace mzgo
