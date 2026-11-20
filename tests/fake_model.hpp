// A deterministic stand-in for the network, for tests of search, self-play
// and anything else written against the Model interface.
//
// MuZero's search never looks at the real rules below the root: everything
// it knows comes from the two inferences. A fake Model therefore lets a test
// script exactly what the "network" believes (which move it likes, what a
// state is worth) and check that the search does the right thing with it.
//
// Every output starts from a deterministic default, and an optional hook may
// then overwrite any part of it:
//   - initial inference: hidden = a hash of the observation, logits all 0
//     (uniform policy), value 0, reward 0. RootHook(observation, out).
//   - recurrent inference: hidden = a hash of (parent hidden, action), so
//     different imagined paths reach different hidden states; logits all 0,
//     value 0, reward 0. StepHook(parentHidden, action, out).
// Hidden values lie in [0, 1), like the real model's normalized hidden state.
// Each batch row is computed on its own, so outputs never depend on batch
// size or composition. Hooks are called from whichever thread runs the
// session; give sessions to several threads only with thread-safe hooks
// (pure functions of their arguments are).
#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>
#include "mzgo/encode.hpp"
#include "mzgo/model.hpp"
#include "mzgo/rng.hpp"

namespace mzgo {

// What the fake returns for one state; a hook receives it pre-filled with
// the default and may change any field (vector sizes must stay the same).
struct FakeOutput {
    std::vector<float> hidden;   // [hiddenSize]
    std::vector<float> logits;   // [actions]
    float value = 0.0f, reward = 0.0f;
};

class FakeModel : public Model {
public:
    using RootHook = std::function<void(const float* observation, FakeOutput& out)>;
    using StepHook = std::function<void(const float* parentHidden, int action, FakeOutput& out)>;

    explicit FakeModel(const ModelDims& dims, int history = 1, RootHook root = nullptr,
                       StepHook step = nullptr)
        : dims_(dims), history_(history), root_(std::move(root)), step_(std::move(step)) {}

    ModelDims dims() const override { return dims_; }
    int history() const override { return history_; }
    std::unique_ptr<ModelSession> newSession(int maxBatch) const override {
        return std::make_unique<Session>(*this, maxBatch);
    }

    // Fills `hs` floats in [0, 1) from a seed (splitmix64 stream).
    static void fillHidden(uint64_t seed, int hs, float* out) {
        for (int i = 0; i < hs; ++i)
            out[i] = static_cast<float>(splitmix64(seed) >> 40) / static_cast<float>(1 << 24);
    }
    // Hash of n floats' bit patterns (FNV-1a over 32-bit words, then mixed).
    static uint64_t hashFloats(const float* x, int n, uint64_t h = 1469598103934665603ULL) {
        for (int i = 0; i < n; ++i) {
            uint32_t bits;
            std::memcpy(&bits, &x[i], sizeof bits);
            h = (h ^ bits) * 1099511628211ULL;
        }
        return splitmix64(h);
    }

private:
    class Session : public ModelSession {
    public:
        Session(const FakeModel& m, int maxBatch) : m_(m), maxBatch_(maxBatch) {}

        void initialInference(const float* observations, int B, ModelOutput& out) override {
            const int obsSize = numPlanes(m_.history_) * m_.dims_.area();
            run(B, out, [&](int b, FakeOutput& o) {
                const float* obs = observations + static_cast<size_t>(b) * obsSize;
                fillHidden(hashFloats(obs, obsSize), m_.dims_.hiddenSize(), o.hidden.data());
                if (m_.root_) m_.root_(obs, o);
            });
        }

        void recurrentInference(const float* hidden, const int* actions, int B,
                                ModelOutput& out) override {
            const int hs = m_.dims_.hiddenSize();
            run(B, out, [&](int b, FakeOutput& o) {
                const float* parent = hidden + static_cast<size_t>(b) * hs;
                fillHidden(hashFloats(parent, hs, deriveSeed(static_cast<uint64_t>(actions[b]))), hs,
                           o.hidden.data());
                if (m_.step_) m_.step_(parent, actions[b], o);
            });
        }

    private:
        template <class Fill>
        void run(int B, ModelOutput& out, Fill fill) {
            assert(B >= 1 && B <= maxBatch_);
            const int hs = m_.dims_.hiddenSize(), na = m_.dims_.actions();
            out.hidden.resize(static_cast<size_t>(B) * hs);
            out.logits.resize(static_cast<size_t>(B) * na);
            out.values.resize(B);
            out.rewards.resize(B);
            FakeOutput o;
            for (int b = 0; b < B; ++b) {
                o.hidden.assign(hs, 0.0f);
                o.logits.assign(na, 0.0f);
                o.value = 0.0f;
                o.reward = 0.0f;
                fill(b, o);
                assert(static_cast<int>(o.hidden.size()) == hs && static_cast<int>(o.logits.size()) == na);
                std::copy(o.hidden.begin(), o.hidden.end(), out.hidden.begin() + static_cast<size_t>(b) * hs);
                std::copy(o.logits.begin(), o.logits.end(), out.logits.begin() + static_cast<size_t>(b) * na);
                out.values[b] = o.value;
                out.rewards[b] = o.reward;
            }
        }

        const FakeModel& m_;
        int maxBatch_;
    };

    ModelDims dims_;
    int history_;
    RootHook root_;
    StepHook step_;
};

} // namespace mzgo
