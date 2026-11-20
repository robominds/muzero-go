#include "mzgo/players.hpp"
#include <cassert>
#include <cmath>

namespace mzgo {

bool fillsOwnEye(const Board& b, int point) {
    if (b.at(point) != Color::Empty) return false;
    const int n = b.size();
    const Color me = b.toMove();
    const int r = point / n, c = point % n;
    const int dr[4] = {-1, 1, 0, 0}, dc[4] = {0, 0, -1, 1};
    for (int k = 0; k < 4; ++k) {
        const int rr = r + dr[k], cc = c + dc[k];
        if (rr < 0 || rr >= n || cc < 0 || cc >= n) continue;
        const int q = rr * n + cc;
        if (b.at(q) != me) return false;
        if (b.libertiesAt(q) <= 1) return false;
    }
    return true;
}

int RandomPlayer::selectMove(const Game& game, Rng& rng) {
    const Board& b = game.board();
    int cand[kMaxBoardSize * kMaxBoardSize];
    int count = 0;
    for (int p = 0; p < b.area(); ++p)
        if (b.at(p) == Color::Empty && game.isLegal(p) && !fillsOwnEye(b, p)) cand[count++] = p;
    if (count == 0) return b.passAction();
    return cand[rng.below(count)];
}

RolloutMctsPlayer::RolloutMctsPlayer(int simulations) : simulations_(simulations) {
    assert(simulations > 0);
}

std::string RolloutMctsPlayer::name() const {
    return "rollout-mcts-" + std::to_string(simulations_);
}

namespace {

struct Node {
    int action = -1;              // move that led here
    bool blackMoved = false;      // who played it
    int visits = 0;
    double wins = 0;              // sum of results for the player who moved into this node, in [0, 1]
    std::vector<int> children;    // indices into the node pool
    std::vector<int> untried;     // legal actions not yet expanded
    bool expanded = false;
};

// Result in [0, 1] for the given side: 1 win, 0.5 tie, 0 loss.
double resultFor(float resultForBlack, bool black) {
    const double r = black ? resultForBlack : -resultForBlack;
    return (r + 1.0) / 2.0;
}

} // namespace

int RolloutMctsPlayer::selectMove(const Game& root, Rng& rng) {
    assert(!root.isTerminal());
    RandomPlayer policy;
    std::vector<Node> pool;
    pool.reserve(simulations_ + 1);
    pool.emplace_back();
    const double c = std::sqrt(2.0);

    for (int sim = 0; sim < simulations_; ++sim) {
        Game g = root;            // replay the path on a copy of the root
        int cur = 0;
        std::vector<int> path{0};
        while (!g.isTerminal()) {
            if (!pool[cur].expanded) {
                pool[cur].untried = g.legalActions();
                pool[cur].expanded = true;
            }
            if (!pool[cur].untried.empty()) {
                auto& u = pool[cur].untried;
                const int i = rng.below(static_cast<int>(u.size()));
                const int a = u[i];
                u[i] = u.back();
                u.pop_back();
                Node child;
                child.action = a;
                child.blackMoved = g.board().toMove() == Color::Black;
                g.play(a);
                pool.push_back(child);
                const int idx = static_cast<int>(pool.size()) - 1;
                pool[cur].children.push_back(idx);
                cur = idx;
                path.push_back(cur);
                break;
            }
            // Fully expanded: pick the child with the best UCB1 score.
            int best = -1;
            double bestScore = -1e30;
            const double logN = std::log(static_cast<double>(pool[cur].visits));
            for (int ch : pool[cur].children) {
                const Node& nd = pool[ch];
                const double score = nd.wins / nd.visits + c * std::sqrt(logN / nd.visits);
                if (score > bestScore) { bestScore = score; best = ch; }
            }
            assert(best >= 0);
            g.play(pool[best].action);
            cur = best;
            path.push_back(cur);
        }
        while (!g.isTerminal()) g.play(policy.selectMove(g, rng));
        const float res = g.resultForBlack();
        for (int idx : path) {
            ++pool[idx].visits;
            if (idx != 0) pool[idx].wins += resultFor(res, pool[idx].blackMoved);
        }
    }

    int bestAction = -1, bestVisits = -1;
    for (int ch : pool[0].children)
        if (pool[ch].visits > bestVisits) { bestVisits = pool[ch].visits; bestAction = pool[ch].action; }
    assert(bestAction >= 0);
    return bestAction;
}

} // namespace mzgo
