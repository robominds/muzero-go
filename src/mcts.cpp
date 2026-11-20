#include "mzgo/mcts.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include "mzgo/encode.hpp"

namespace mzgo {

MinMaxStats::MinMaxStats(float lo, float hi) : lo_(lo), hi_(hi) {}

void MinMaxStats::update(float q) {
    lo_ = std::min(lo_, q);
    hi_ = std::max(hi_, q);
}

float MinMaxStats::normalize(float q) const { return hi_ > lo_ ? (q - lo_) / (hi_ - lo_) : q; }

Search::Search(const MctsConfig& cfg, const ModelDims& dims) : cfg_(cfg), dims_(dims) {
    assert(cfg_.simulations > 0);
    // One hidden block for the root and one per simulation, allocated once,
    // so a LeafRequest's pointer never moves during a search. The node arena
    // gets the same treatment: each simulation adds `actions` children.
    hidden_.assign(static_cast<size_t>(cfg_.simulations + 1) * dims_.hiddenSize(), 0.0f);
    nodes_.reserve(static_cast<size_t>(cfg_.simulations + 1) * dims_.actions() + 1);
    probs_.resize(dims_.actions());
}

void Search::storeHidden(Node& n, const float* hidden) {
    assert(expanded_ <= cfg_.simulations);
    const size_t hs = static_cast<size_t>(dims_.hiddenSize());
    n.hiddenSlot = expanded_++;
    std::memcpy(hidden_.data() + static_cast<size_t>(n.hiddenSlot) * hs, hidden, hs * sizeof(float));
}

void Search::beginRoot(const float* hidden, const float* logits, float /*value*/,
                       const std::vector<int>& legal, bool previousMoveWasPass, Rng& rng) {
    assert(!legal.empty());
    for (size_t i = 0; i < legal.size(); ++i) {
        assert(legal[i] >= 0 && legal[i] < dims_.actions());
        assert(i == 0 || legal[i] > legal[i - 1]);
    }
    nodes_.clear();
    path_.clear();
    minmax_ = MinMaxStats();
    expanded_ = simulations_ = 0;
    pending_ = false;
    previousMoveWasPass_ = previousMoveWasPass;
    maxDepth_ = depthSum_ = pastEnd_ = 0;

    nodes_.push_back(Node{-1, 1.0f});
    storeHidden(nodes_[0], hidden);

    // The only place the rules enter: children for the legal moves only,
    // prior renormalized over them (uniform if the model gave them all 0).
    softmax(logits, dims_.actions(), probs_.data());
    double sum = 0.0;
    for (int a : legal) sum += probs_[a];
    std::vector<float> priors(legal.size());
    for (size_t i = 0; i < legal.size(); ++i)
        priors[i] = sum > 0.0 ? static_cast<float>(probs_[legal[i]] / sum) : 1.0f / static_cast<float>(legal.size());
    if (cfg_.addNoise) {
        const std::vector<double> eta = dirichlet(rng, cfg_.dirichletAlpha, static_cast<int>(legal.size()));
        const float f = cfg_.noiseFraction;
        for (size_t i = 0; i < legal.size(); ++i)
            priors[i] = (1.0f - f) * priors[i] + f * static_cast<float>(eta[i]);
    }
    nodes_[0].firstChild = 1;
    nodes_[0].numChildren = static_cast<int>(legal.size());
    for (size_t i = 0; i < legal.size(); ++i) nodes_.push_back(Node{legal[i], priors[i]});
}

bool Search::finished() const { return simulations_ >= cfg_.simulations; }

int Search::selectChild(const Node& parent) const {
    const float np = static_cast<float>(parent.visits);
    const float pbC = std::log((np + cfg_.pbCBase + 1.0f) / cfg_.pbCBase) + cfg_.pbCInit;
    const float sqrtN = std::sqrt(np);
    int best = -1;
    float bestScore = 0.0f, bestPrior = 0.0f;
    for (int i = parent.firstChild; i < parent.firstChild + parent.numChildren; ++i) {
        const Node& c = nodes_[i];
        // Unvisited: Q̄ = 0 on the normalized scale (as bad as a loss), so
        // only the prior term can draw the search there.
        const float q = c.visits > 0 ? minmax_.normalize(edgeQ(c)) : 0.0f;
        const float score = q + c.prior * sqrtN / (1.0f + static_cast<float>(c.visits)) * pbC;
        // Ties (every score is 0 on the root's first simulation): larger
        // prior, then lower action (children are in ascending action order).
        if (best < 0 || score > bestScore || (score == bestScore && c.prior > bestPrior)) {
            best = i;
            bestScore = score;
            bestPrior = c.prior;
        }
    }
    assert(best >= 0);
    return best;
}

LeafRequest Search::selectLeaf() {
    assert(!finished() && !pending_ && !nodes_.empty());
    path_.clear();
    int node = 0;
    path_.push_back(node);
    while (nodes_[node].firstChild >= 0) {
        node = selectChild(nodes_[node]);
        path_.push_back(node);
    }
    pending_ = true;
    const Node& parent = nodes_[path_[path_.size() - 2]];
    return {hidden_.data() + static_cast<size_t>(parent.hiddenSlot) * dims_.hiddenSize(), nodes_[node].action};
}

void Search::expandLeaf(const float* hidden, const float* logits, float value, float reward) {
    assert(pending_);
    pending_ = false;
    const int leaf = path_.back();
    storeHidden(nodes_[leaf], hidden);
    nodes_[leaf].reward = reward;

    // Below the root: every action, prior straight from the model, unmasked.
    softmax(logits, dims_.actions(), probs_.data());
    const int first = static_cast<int>(nodes_.size());
    for (int a = 0; a < dims_.actions(); ++a) nodes_.push_back(Node{a, probs_[a]});
    nodes_[leaf].firstChild = first;          // re-indexed: push_back may have moved nodes_
    nodes_[leaf].numChildren = dims_.actions();

    // Backup. `v` is always from the view of the side to move at the node
    // being updated; handing it to the parent collects the edge's reward
    // and flips the sign.
    float v = value;
    for (int i = static_cast<int>(path_.size()) - 1; i >= 0; --i) {
        Node& n = nodes_[path_[i]];
        n.valueSum += v;
        n.visits += 1;
        if (i > 0) minmax_.update(edgeQ(n));
        v = n.reward + cfg_.discount * -v;
    }

    // Statistics: leaf depth, and whether the imagined path (after the real
    // previous move) holds two passes in a row.
    const int depth = static_cast<int>(path_.size()) - 1;
    maxDepth_ = std::max(maxDepth_, depth);
    depthSum_ += depth;
    bool prevPass = previousMoveWasPass_, pastEnd = false;
    for (size_t i = 1; i < path_.size(); ++i) {
        const bool isPass = nodes_[path_[i]].action == dims_.area();
        pastEnd = pastEnd || (isPass && prevPass);
        prevPass = isPass;
    }
    pastEnd_ += pastEnd;
    ++simulations_;
}

SearchResult Search::result() const {
    assert(simulations_ > 0);
    const Node& root = nodes_[0];
    SearchResult r;
    r.visits.assign(dims_.actions(), 0);
    r.policy.assign(dims_.actions(), 0.0f);
    int total = 0;
    for (int i = root.firstChild; i < root.firstChild + root.numChildren; ++i) {
        r.visits[nodes_[i].action] = nodes_[i].visits;
        total += nodes_[i].visits;
    }
    for (int a = 0; a < dims_.actions(); ++a)
        r.policy[a] = static_cast<float>(r.visits[a]) / static_cast<float>(total);
    r.rootValue = mean(root);
    r.maxDepth = maxDepth_;
    r.meanDepth = static_cast<float>(depthSum_) / static_cast<float>(simulations_);
    r.pastEndFraction = static_cast<float>(pastEnd_) / static_cast<float>(simulations_);
    return r;
}

int Search::selectMove(const SearchResult& r, bool sample, Rng& rng) const {
    int total = 0;
    for (int v : r.visits) total += v;
    assert(total > 0);
    if (sample) {
        // Temperature 1: probability proportional to visit count.
        const double x = rng.uniform() * total;
        double run = 0.0;
        int last = 0;
        for (int a = 0; a < static_cast<int>(r.visits.size()); ++a) {
            if (r.visits[a] == 0) continue;
            run += r.visits[a];
            last = a;
            if (x < run) return a;
        }
        return last;
    }
    int best = 0;
    for (int a = 1; a < static_cast<int>(r.visits.size()); ++a)
        if (r.visits[a] > r.visits[best]) best = a;
    return best;
}

std::vector<RootEdge> Search::rootEdges() const {
    std::vector<RootEdge> edges;
    const Node& root = nodes_[0];
    for (int i = root.firstChild; i < root.firstChild + root.numChildren; ++i) {
        const Node& c = nodes_[i];
        edges.push_back({c.action, c.prior, c.visits, mean(c), c.visits > 0 ? edgeQ(c) : 0.0f});
    }
    return edges;
}

SearchResult runSearch(Search& search, ModelSession& session, const Game& game, int history, Rng& rng) {
    assert(!game.isTerminal());
    const int n = game.size();
    std::vector<float> obs(static_cast<size_t>(numPlanes(history)) * n * n);
    encodeGame(game, history, 0, obs.data());
    ModelOutput root, step;
    session.initialInference(obs.data(), 1, root);
    const bool previousMoveWasPass = !game.moves().empty() && game.moves().back() == game.board().passAction();
    search.beginRoot(root.hidden.data(), root.logits.data(), root.values[0], game.legalActions(),
                     previousMoveWasPass, rng);
    while (!search.finished()) {
        const LeafRequest req = search.selectLeaf();
        session.recurrentInference(req.parentHidden, &req.action, 1, step);
        search.expandLeaf(step.hidden.data(), step.logits.data(), step.values[0], step.rewards[0]);
    }
    return search.result();
}

} // namespace mzgo
