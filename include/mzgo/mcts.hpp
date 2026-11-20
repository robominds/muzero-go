// MuZero's search: Monte Carlo tree search inside the learned model.
//
// AlphaZero searches over real board positions. MuZero reads the real board
// exactly once, at the root: the representation function turns the
// observation into a hidden state, and from then on the tree is built by the
// dynamics function alone, one imagined step per edge. Each simulation
//   1. selects: walks down from the root, at each node taking the child with
//      the best pUCT score  Q̄ + P·√N_parent/(1+N_child)·(c1 + log((N_parent+c2+1)/c2));
//   2. expands: asks the model for the leaf (dynamics on the parent's hidden
//      state and the action, then prediction), giving the leaf's hidden
//      state, reward, policy prior over its children and value;
//   3. backs up: adds the leaf value along the path, negating it at every
//      ply because the players alternate.
// The rules only enter at the root, where the children are limited to the
// real legal moves. Below it every action (each point, and pass) is allowed
// and nothing is terminal: the model has to learn what an illegal move or a
// game that already ended "means".
//
// Signs: a node's value is from the view of the player to move there; the
// parent sees an edge as Q = r + γ·(−v_child). Q is mapped to [0, 1] by
// MinMaxStats before it is weighed against the prior term; an unvisited
// child scores Q̄ = 0 there.
//
// The search is split into beginRoot / selectLeaf / expandLeaf so a caller
// can run one simulation in each of many games and evaluate all their leaves
// in one batched model call; runSearch is the plain one-game loop.
#pragma once
#include <utility>
#include <vector>
#include "mzgo/game.hpp"
#include "mzgo/model.hpp"
#include "mzgo/rng.hpp"

namespace mzgo {

struct MctsConfig {
    int simulations = 200;
    float pbCInit = 1.25f, pbCBase = 19652.0f, discount = 1.0f;
    // Root Dirichlet noise (self-play only), mixed over the legal moves.
    bool addNoise = false; float dirichletAlpha = 0.12f, noiseFraction = 0.25f;
};

// Range of the edge Q values seen in one tree, for normalizing Q to [0, 1].
// Starts at the known bounds of a board game's value, [-1, 1] (the
// pseudocode's choice, "identical behaviour to AlphaZero in board games");
// rewards could push Q outside, so update widens the range.
class MinMaxStats {
public:
    MinMaxStats(float lo = -1.f, float hi = 1.f);
    void update(float q);
    float normalize(float q) const;    // (q - lo)/(hi - lo) if hi > lo else q
    float lo() const { return lo_; }
    float hi() const { return hi_; }
private:
    float lo_, hi_;
};

// What a pending leaf needs from the model: dynamics on (parentHidden, action).
// parentHidden points into the search's own storage and stays valid until
// the next beginRoot. depth: the leaf's depth (root child = 1), which the
// model uses to pick its per-unroll-step batch norm statistics.
struct LeafRequest { const float* parentHidden; int action; int depth; };

struct SearchResult {
    std::vector<int> visits;           // [actions], illegal-at-root actions are 0
    std::vector<float> policy;         // visits / sum
    float rootValue;                   // root mean value, side to move's view
    int maxDepth; float meanDepth;     // depth of each simulation's new leaf (root child = 1)
    // Simulations whose imagined path goes on BEYOND two consecutive passes
    // (counting the real previous move when it was a pass): the double pass
    // is not the leaf, so the search expanded a state after the point where
    // the real game would have ended. A leaf that is the double pass itself
    // does not count.
    float pastEndFraction;
};

// Statistics of one root edge, for tests and analysis output.
struct RootEdge {
    int action;
    float prior;       // after masking (and noise)
    int visits;
    float meanValue;   // child's mean value, from the child's side to move; 0 if unvisited
    float q;           // r + γ·(−meanValue), from the root's side to move; 0 if unvisited
};

// One ply of the principal variation.
struct PvStep {
    int action;
    float value;       // the child's mean value, from the child's side to move
    float reward;      // predicted reward on the edge, from the mover's view
    float q;           // r + γ·(−value), from the mover's view
};

class Search {                          // one game's tree; reused across moves (cleared by beginRoot)
public:
    Search(const MctsConfig&, const ModelDims&);
    // From the root's initial inference: priors = softmax(logits) restricted to `legal`
    // (ascending, non-empty) and renormalized; noise mixed over the legal actions when
    // cfg.addNoise. The root's own value is not backed up (as in the pseudocode).
    void beginRoot(const float* hidden, const float* logits, float value,
                   const std::vector<int>& legal, bool previousMoveWasPass, Rng& rng);
    bool finished() const;              // simulations done
    LeafRequest selectLeaf();           // descends from the root; asserts !finished()
    // Answers the last selectLeaf: stores the leaf, gives it children over all
    // actions with prior softmax(logits), and backs `value` up the path.
    void expandLeaf(const float* hidden, const float* logits, float value, float reward);
    SearchResult result() const;
    int selectMove(const SearchResult&, bool sample, Rng& rng) const;  // sample ∝ visits, else argmax (ties: lowest index)
    std::vector<RootEdge> rootEdges() const;   // ascending action order
    const MinMaxStats& minMaxStats() const { return minmax_; }   // this tree's Q range so far
    // The principal variation, read from the tree: from the root, repeatedly
    // the most visited child (ties: lowest action), up to maxLen plies, while
    // that child has visits. Each entry is the move, that child's mean
    // value, and the edge's reward and Q: the line the search believes in,
    // entirely in imagined space below the root.
    std::vector<PvStep> principalVariation(int maxLen) const;

private:
    struct Node {
        int action;              // edge from the parent
        float prior;
        float reward = 0.0f;     // on the edge into this node, parent's view
        int visits = 0;
        float valueSum = 0.0f;   // this node's side to move
        int firstChild = -1;     // children are contiguous in nodes_; -1 = not expanded
        int numChildren = 0;
        int hiddenSlot = -1;     // index into hidden_, in units of hiddenSize
    };
    float mean(const Node& n) const { return n.visits ? n.valueSum / static_cast<float>(n.visits) : 0.0f; }
    float edgeQ(const Node& child) const { return child.reward + cfg_.discount * -mean(child); }
    int selectChild(const Node& parent) const;
    void storeHidden(Node& n, const float* hidden);

    MctsConfig cfg_;
    ModelDims dims_;
    std::vector<Node> nodes_;           // the arena; nodes_[0] is the root
    std::vector<float> hidden_;         // one hiddenSize block per expanded node
    std::vector<float> probs_;          // softmax scratch
    std::vector<int> path_;             // node indices of the pending simulation
    MinMaxStats minmax_;
    int expanded_ = 0, simulations_ = 0;
    bool pending_ = false, previousMoveWasPass_ = false;
    int maxDepth_ = 0, depthSum_ = 0, pastEnd_ = 0;
};

// Convenience for one game, batch 1: begin + loop of select/recurrentInference/expand.
// Asserts !game.isTerminal().
SearchResult runSearch(Search&, ModelSession&, const Game& game, int history, Rng& rng);

} // namespace mzgo
