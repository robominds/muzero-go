#include "mzgo/selfplay.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <utility>
#include "mzgo/encode.hpp"
#include "mzgo/parallel.hpp"

namespace mzgo {

namespace {

// One game of a group, with its own tree and random stream.
struct LiveGame {
    LiveGame(const SelfPlayConfig& cfg, const ModelDims& dims, uint64_t seed)
        : game(cfg.boardSize, cfg.komi, cfg.maxMoves), search(cfg.mcts, dims), rng(seed) {}
    Game game;
    Search search;
    Rng rng;
    std::vector<float> policies, rootValues;   // per move, for recordFromGame
    bool atRoot = true;                        // the next step begins a new search
};

// Per-game search statistics, reduced in game order at the end.
struct GameStats {
    int searches = 0, maxDepth = 0;
    double depthSum = 0, pastEndSum = 0;
};

// Plays games [begin, end) in lockstep with one session.
void playGroup(const Model& model, const SelfPlayConfig& cfg, uint64_t seed, int begin, int end,
               ModelSession& session, std::vector<GameRecord>& records, std::vector<GameStats>& stats) {
    const ModelDims dims = model.dims();
    const int history = model.history();
    const int obsSize = numPlanes(history) * dims.area(), hs = dims.hiddenSize();

    std::vector<std::unique_ptr<LiveGame>> live;
    for (int i = begin; i < end; ++i)
        live.push_back(std::make_unique<LiveGame>(cfg, dims, deriveSeed(seed, static_cast<uint64_t>(i))));
    std::vector<int> index(live.size());           // game index of each live entry
    for (size_t k = 0; k < live.size(); ++k) index[k] = begin + static_cast<int>(k);

    std::vector<float> obs, parents;
    std::vector<int> actions, depths, batch;       // batch: positions in `live` of this call's rows
    ModelOutput out;
    while (!live.empty()) {
        // 1. Games starting a move: one batched initial inference for their roots.
        batch.clear();
        for (size_t k = 0; k < live.size(); ++k)
            if (live[k]->atRoot) batch.push_back(static_cast<int>(k));
        if (!batch.empty()) {
            obs.resize(batch.size() * static_cast<size_t>(obsSize));
            for (size_t b = 0; b < batch.size(); ++b)
                encodeGame(live[batch[b]]->game, history, 0, obs.data() + b * obsSize);
            session.initialInference(obs.data(), static_cast<int>(batch.size()), out);
            for (size_t b = 0; b < batch.size(); ++b) {
                LiveGame& g = *live[batch[b]];
                const auto& moves = g.game.moves();
                const bool previousMoveWasPass = !moves.empty() && moves.back() == g.game.board().passAction();
                g.search.beginRoot(out.hidden.data() + b * hs, out.logits.data() + b * dims.actions(),
                                   out.values[b], g.game.legalActions(), previousMoveWasPass, g.rng);
                g.atRoot = false;
            }
        }

        // 2. One simulation in every unfinished search: one batched recurrent inference.
        batch.clear();
        actions.clear();
        depths.clear();
        parents.clear();
        for (size_t k = 0; k < live.size(); ++k) {
            if (live[k]->search.finished()) continue;
            const LeafRequest req = live[k]->search.selectLeaf();
            batch.push_back(static_cast<int>(k));
            actions.push_back(req.action);
            depths.push_back(req.depth);         // the games' leaves sit at different depths
            parents.insert(parents.end(), req.parentHidden, req.parentHidden + hs);
        }
        if (!batch.empty()) {
            session.recurrentInference(parents.data(), actions.data(), depths.data(), static_cast<int>(batch.size()),
                                       out);
            for (size_t b = 0; b < batch.size(); ++b)
                live[batch[b]]->search.expandLeaf(out.hidden.data() + b * hs, out.logits.data() + b * dims.actions(),
                                                  out.values[b], out.rewards[b]);
        }

        // 3. Finished searches: record the search, play the move, retire ended games.
        for (size_t k = 0; k < live.size();) {
            LiveGame& g = *live[k];
            if (!g.search.finished()) { ++k; continue; }
            const SearchResult r = g.search.result();
            GameStats& s = stats[index[k]];
            ++s.searches;
            s.maxDepth = std::max(s.maxDepth, r.maxDepth);
            s.depthSum += r.meanDepth;
            s.pastEndSum += r.pastEndFraction;
            g.policies.insert(g.policies.end(), r.policy.begin(), r.policy.end());
            g.rootValues.push_back(r.rootValue);
            const bool sample = g.game.moveCount() < cfg.temperatureMoves;
            g.game.play(g.search.selectMove(r, sample, g.rng));
            g.atRoot = true;
            if (g.game.isTerminal()) {
                records[index[k]] = recordFromGame(g.game, g.policies, g.rootValues);
                live.erase(live.begin() + static_cast<std::ptrdiff_t>(k));
                index.erase(index.begin() + static_cast<std::ptrdiff_t>(k));
            } else {
                ++k;
            }
        }
    }
}

} // namespace

std::vector<GameRecord> playSelfPlay(const Model& model, const SelfPlayConfig& cfg, int games, uint64_t seed,
                                     int workers, SelfPlayStats* stats) {
    assert(cfg.group >= 1 && games >= 0);
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<GameRecord> records(games);
    std::vector<GameStats> perGame(games);
    const int groups = (games + cfg.group - 1) / cfg.group;
    const int threads = std::max(1, std::min(resolveWorkers(workers), groups));
    // One session per worker thread, created by that thread on first use.
    std::vector<std::unique_ptr<ModelSession>> sessions(threads);
    parallelFor(groups, threads, [&](int grp, int worker) {
        if (!sessions[worker]) sessions[worker] = model.newSession(cfg.group);
        const int begin = grp * cfg.group, end = std::min(games, begin + cfg.group);
        playGroup(model, cfg, seed, begin, end, *sessions[worker], records, perGame);
    });

    if (stats) {
        // Reduced in game order, so the numbers do not depend on `workers` either.
        int searches = 0, maxDepth = 0, capped = 0;
        double depthSum = 0, pastEndSum = 0, moves = 0;
        for (int i = 0; i < games; ++i) {
            searches += perGame[i].searches;
            maxDepth = std::max(maxDepth, perGame[i].maxDepth);
            depthSum += perGame[i].depthSum;
            pastEndSum += perGame[i].pastEndSum;
            moves += records[i].length();
            capped += !records[i].endedByPasses();
        }
        stats->seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        stats->meanLength = games ? moves / games : 0.0;
        stats->meanDepth = searches ? depthSum / searches : 0.0;
        stats->maxDepth = maxDepth;
        stats->pastEndFraction = searches ? pastEndSum / searches : 0.0;
        stats->cappedGames = capped;
    }
    return records;
}

MuZeroPlayer::MuZeroPlayer(const Model& model, const MctsConfig& cfg, int temperatureMoves, std::string name)
    : model_(model), temperatureMoves_(temperatureMoves), name_(std::move(name)), session_(model.newSession(1)),
      search_([&] { MctsConfig c = cfg; c.addNoise = false; return c; }(), model.dims()) {}

int MuZeroPlayer::selectMove(const Game& game, Rng& rng) {
    last_ = runSearch(search_, *session_, game, model_.history(), rng);
    return search_.selectMove(last_, game.moveCount() < temperatureMoves_, rng);
}

SearchResult MuZeroPlayer::lastSearch() const { return last_; }

} // namespace mzgo
