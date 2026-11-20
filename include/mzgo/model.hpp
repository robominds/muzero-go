// What search and self-play see of a network. A Model is the shared,
// read-only weights; each worker thread takes a ModelSession from it for its
// own scratch space. MuZero's two inferences: initialInference maps an
// observation to a hidden state plus policy and value (representation then
// prediction); recurrentInference maps a hidden state and an action to the
// next hidden state plus reward, policy and value (dynamics then prediction).
#pragma once
#include <memory>
#include <vector>

namespace mzgo {

struct ModelDims {
    int boardSize = 0, filters = 0;
    int area() const { return boardSize * boardSize; }
    int actions() const { return area() + 1; }          // every point, plus pass
    int hiddenSize() const { return filters * area(); }
};

struct ModelOutput {               // resized by the session to B
    std::vector<float> hidden;     // [B][hiddenSize], normalized to [0,1]
    std::vector<float> logits;     // [B][actions]
    std::vector<float> values;     // [B], side to move at that state, in [-1,1]
    std::vector<float> rewards;    // [B], 0 for initial inference and without a reward head
};

class ModelSession {               // one per worker thread; not thread-safe
public:
    virtual ~ModelSession() = default;
    virtual void initialInference(const float* observations /*[B][planes][area]*/, int B, ModelOutput& out) = 0;
    virtual void recurrentInference(const float* hidden /*[B][hiddenSize]*/, const int* actions, int B,
                                    ModelOutput& out) = 0;
};

class Model {                      // shared read-only across threads
public:
    virtual ~Model() = default;
    virtual ModelDims dims() const = 0;
    virtual int history() const = 0;   // H for the observation planes
    virtual std::unique_ptr<ModelSession> newSession(int maxBatch) const = 0;
};

// Numerically stable softmax of n logits into probs (subtracts the maximum).
void softmax(const float* logits, int n, float* probs);

} // namespace mzgo
