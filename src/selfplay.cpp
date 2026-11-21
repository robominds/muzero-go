#include "mzgo/selfplay.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <numeric>
#include <utility>
#include "mzgo/encode.hpp"
#include "mzgo/parallel.hpp"

namespace mzgo {

namespace {

// One game of a group, with its own tree and random stream.
struct LiveGame {
    LiveGame(const SelfPlayConfig& cfg, const ModelDims& dims, uint64_t seed, bool learnerBlack)
        : game(cfg.boardSize, cfg.komi, cfg.maxMoves), search(cfg.mcts, dims), rng(seed),
          learnerBlack(learnerBlack) {}
    Game game;
    Search search;                             // either side's search (all models share dims)
    Rng rng;
    std::vector<float> policies, rootValues;   // per move, for recordFromGame
    std::vector<uint8_t> mask;                 // per move, league games only: 1 = the learner's move
    bool learnerBlack;                         // league games only
    bool atRoot = true;                        // the next step begins a new search
};

// Per-game search statistics (the learner's searches), reduced in game order at the end.
struct GameStats {
    int searches = 0, maxDepth = 0;
    double depthSum = 0, pastEndSum = 0;
};

// Who plays a group's games. In self-play the learner plays both sides. In a
// league group it plays one side (Black in even games) and the opponent the
// other: a network through its own session, or the random bot when null.
struct GroupPlayers {
    ModelSession* learner;
    bool league = false;
    ModelSession* opponent = nullptr;
};

// Plays the games `games` (game indices) in lockstep.
void playGroup(const Model& model, const SelfPlayConfig& cfg, uint64_t seed, const std::vector<int>& games,
               const GroupPlayers& players, std::vector<GameRecord>& records, std::vector<GameStats>& stats) {
    const ModelDims dims = model.dims();
    const int history = model.history();
    const bool passPlane = model.passPlane();
    const int obsSize = numPlanes(history, passPlane) * dims.area(), hs = dims.hiddenSize();

    std::vector<std::unique_ptr<LiveGame>> live;
    for (int i : games)
        live.push_back(std::make_unique<LiveGame>(cfg, dims, deriveSeed(seed, static_cast<uint64_t>(i)), i % 2 == 0));
    std::vector<int> index(games);                 // game index of each live entry

    const auto learnerToMove = [&](const LiveGame& g) {
        return !players.league || (g.game.moveCount() % 2 == 0) == g.learnerBlack;
    };
    // Records the move chosen at live[k]'s current position and plays it; a game
    // that ends is retired (true: live[k] is now the next game).
    const auto playMove = [&](size_t k, int move, const float* policy, float rootValue, bool learnersMove) {
        LiveGame& g = *live[k];
        g.policies.insert(g.policies.end(), policy, policy + dims.actions());
        g.rootValues.push_back(rootValue);
        if (players.league) g.mask.push_back(learnersMove ? 1 : 0);
        g.game.play(move);
        g.atRoot = true;
        if (!g.game.isTerminal()) return false;
        GameRecord& r = records[index[k]];
        r = recordFromGame(g.game, g.policies, g.rootValues);
        r.policyMask = std::move(g.mask);
        live.erase(live.begin() + static_cast<std::ptrdiff_t>(k));
        index.erase(index.begin() + static_cast<std::ptrdiff_t>(k));
        return true;
    };
    const bool bot = players.league && !players.opponent;
    RandomPlayer randomBot;
    const std::vector<float> zeroPolicy(dims.actions(), 0.0f);
    ModelSession* const sessions[2] = {players.learner, players.opponent};   // by side: learner, opponent

    std::vector<float> obs, parents;
    std::vector<int> actions, depths, batch;       // batch: positions in `live` of this call's rows
    ModelOutput out;
    while (!live.empty()) {
        // 0. The random bot moves at once: no search, a zero policy row and root value 0.
        if (bot) {
            for (size_t k = 0; k < live.size();) {
                LiveGame& g = *live[k];
                if (g.atRoot && !learnerToMove(g) &&
                    playMove(k, randomBot.selectMove(g.game, g.rng), zeroPolicy.data(), 0.0f, false))
                    continue;
                ++k;
            }
        }

        // 1. Games starting a move: per side, one batched initial inference for their roots.
        for (int side = 0; side < 2; ++side) {
            if (!sessions[side]) continue;
            batch.clear();
            for (size_t k = 0; k < live.size(); ++k)
                if (live[k]->atRoot && learnerToMove(*live[k]) == (side == 0)) batch.push_back(static_cast<int>(k));
            if (batch.empty()) continue;
            obs.resize(batch.size() * static_cast<size_t>(obsSize));
            for (size_t b = 0; b < batch.size(); ++b)
                encodeGame(live[batch[b]]->game, history, 0, obs.data() + b * obsSize, passPlane);
            sessions[side]->initialInference(obs.data(), static_cast<int>(batch.size()), out);
            for (size_t b = 0; b < batch.size(); ++b) {
                LiveGame& g = *live[batch[b]];
                const auto& moves = g.game.moves();
                const bool previousMoveWasPass = !moves.empty() && moves.back() == g.game.board().passAction();
                g.search.beginRoot(out.hidden.data() + b * hs, out.logits.data() + b * dims.actions(),
                                   out.values[b], rootLegalActions(g.game, cfg.noPassBefore), previousMoveWasPass, g.rng);
                g.atRoot = false;
            }
        }

        // 2. One simulation in every unfinished search: per side, one batched recurrent inference.
        for (int side = 0; side < 2; ++side) {
            if (!sessions[side]) continue;
            batch.clear();
            actions.clear();
            depths.clear();
            parents.clear();
            for (size_t k = 0; k < live.size(); ++k) {
                if (live[k]->search.finished() || learnerToMove(*live[k]) != (side == 0)) continue;
                const LeafRequest req = live[k]->search.selectLeaf();
                batch.push_back(static_cast<int>(k));
                actions.push_back(req.action);
                depths.push_back(req.depth);     // the games' leaves sit at different depths
                parents.insert(parents.end(), req.parentHidden, req.parentHidden + hs);
            }
            if (batch.empty()) continue;
            sessions[side]->recurrentInference(parents.data(), actions.data(), depths.data(),
                                               static_cast<int>(batch.size()), out);
            for (size_t b = 0; b < batch.size(); ++b)
                live[batch[b]]->search.expandLeaf(out.hidden.data() + b * hs, out.logits.data() + b * dims.actions(),
                                                  out.values[b], out.rewards[b]);
        }

        // 3. Finished searches: record the search, play the move, retire ended games.
        for (size_t k = 0; k < live.size();) {
            LiveGame& g = *live[k];
            if (!g.search.finished()) { ++k; continue; }
            const SearchResult r = g.search.result();
            const bool learnersMove = learnerToMove(g);
            if (learnersMove) {
                GameStats& s = stats[index[k]];
                ++s.searches;
                s.maxDepth = std::max(s.maxDepth, r.maxDepth);
                s.depthSum += r.meanDepth;
                s.pastEndSum += r.pastEndFraction;
            }
            const bool sample = g.game.moveCount() < cfg.temperatureMoves;
            if (!playMove(k, g.search.selectMove(r, sample, g.rng), r.policy.data(), r.rootValue, learnersMove)) ++k;
        }
    }
}

// Reduces the per-game numbers in game order, so they do not depend on `workers` either.
void reduceStats(const std::vector<GameRecord>& records, const std::vector<GameStats>& perGame,
                 std::chrono::steady_clock::time_point t0, SelfPlayStats& stats) {
    const int games = static_cast<int>(records.size());
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
    stats.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    stats.meanLength = games ? moves / games : 0.0;
    stats.meanDepth = searches ? depthSum / searches : 0.0;
    stats.maxDepth = maxDepth;
    stats.pastEndFraction = searches ? pastEndSum / searches : 0.0;
    stats.cappedGames = capped;
}

} // namespace

std::vector<int> rootLegalActions(const Game& game, int noPassBefore) {
    std::vector<int> legal = game.legalActions();
    if (game.moveCount() >= noPassBefore || legal.size() < 2) return legal;
    const int pass = game.board().passAction();
    legal.erase(std::remove(legal.begin(), legal.end(), pass), legal.end());
    return legal;
}

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
        std::vector<int> group(end - begin);
        std::iota(group.begin(), group.end(), begin);
        playGroup(model, cfg, seed, group, GroupPlayers{sessions[worker].get()}, records, perGame);
    });
    if (stats) reduceStats(records, perGame, t0, *stats);
    return records;
}

std::vector<GameRecord> playLeagueGames(const Model& learner, const std::vector<LeagueOpponent>& opponents,
                                        const std::vector<int>& opponentOf, const SelfPlayConfig& cfg,
                                        uint64_t seed, int workers, SelfPlayStats* stats) {
    assert(cfg.group >= 1);
    for (const LeagueOpponent& o : opponents) {
        assert(!o.model || (o.model->dims().boardSize == learner.dims().boardSize &&
                            o.model->dims().filters == learner.dims().filters &&
                            o.model->history() == learner.history() && o.model->passPlane() == learner.passPlane()));
        (void)o;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const int games = static_cast<int>(opponentOf.size());
    std::vector<GameRecord> records(games);
    std::vector<GameStats> perGame(games);

    for (int o : opponentOf) {
        assert(o >= 0 && o < static_cast<int>(opponents.size()));
        (void)o;
    }
    // Game indices sorted by opponent (stable), cut into groups of at most cfg.group with one opponent each.
    std::vector<int> order(games);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return opponentOf[a] < opponentOf[b]; });
    std::vector<std::vector<int>> groups;
    for (int i : order) {
        if (groups.empty() || static_cast<int>(groups.back().size()) == cfg.group ||
            opponentOf[groups.back().front()] != opponentOf[i])
            groups.emplace_back();
        groups.back().push_back(i);
    }

    const int numGroups = static_cast<int>(groups.size());
    const int threads = std::max(1, std::min(resolveWorkers(workers), numGroups));
    // Per worker thread: a learner session and one per network opponent it meets, created on first use.
    struct WorkerSessions {
        std::unique_ptr<ModelSession> learner;
        std::vector<std::unique_ptr<ModelSession>> opponents;
    };
    std::vector<WorkerSessions> sessions(threads);
    parallelFor(numGroups, threads, [&](int grp, int worker) {
        WorkerSessions& w = sessions[worker];
        if (!w.learner) {
            w.learner = learner.newSession(cfg.group);
            w.opponents.resize(opponents.size());
        }
        const int opp = opponentOf[groups[grp].front()];
        GroupPlayers players{w.learner.get(), true};
        if (const Model* m = opponents[opp].model) {
            if (!w.opponents[opp]) w.opponents[opp] = m->newSession(cfg.group);
            players.opponent = w.opponents[opp].get();
        }
        playGroup(learner, cfg, seed, groups[grp], players, records, perGame);
    });
    if (stats) reduceStats(records, perGame, t0, *stats);
    return records;
}

MuZeroPlayer::MuZeroPlayer(const Model& model, const MctsConfig& cfg, int temperatureMoves, std::string name)
    : model_(model), temperatureMoves_(temperatureMoves), name_(std::move(name)), session_(model.newSession(1)),
      search_([&] { MctsConfig c = cfg; c.addNoise = false; return c; }(), model.dims()) {}

int MuZeroPlayer::selectMove(const Game& game, Rng& rng) {
    last_ = runSearch(search_, *session_, game, model_.history(), model_.passPlane(), rng);
    return search_.selectMove(last_, game.moveCount() < temperatureMoves_, rng);
}

SearchResult MuZeroPlayer::lastSearch() const { return last_; }

} // namespace mzgo
