#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>
#include "fake_model.hpp"
#include "mzgo/game.hpp"
#include "mzgo/match.hpp"
#include "mzgo/players.hpp"
#include "mzgo/selfplay.hpp"

using namespace mzgo;

// 5x5 board with 2 filters: 26 actions (pass = 25), hidden size 50.
static const ModelDims kDims{5, 2};
static const int kActions = 26;

// A fake whose policy and value vary with the state and, below the root,
// with the depth the search passes (pure functions of the inputs, so
// thread-safe): searches then differ from move to move and from game to
// game, which makes the equality tests below meaningful, and a lockstep
// batch that handed a row another row's depth would change the games.
static void varyOutput(const float* hidden, FakeOutput& o) {
    for (int a = 0; a < kActions; ++a) o.logits[a] = 2.0f * o.hidden[a % kDims.hiddenSize()];
    o.value = std::max(-1.0f, std::min(1.0f, 2.0f * hidden[0] - 1.0f + 0.1f * static_cast<float>(o.depth % 3)));
}
static FakeModel varyingModel() {
    return FakeModel(kDims, 2, [](const float*, FakeOutput& o) { varyOutput(o.hidden.data(), o); },
                     [](const float* parent, int, FakeOutput& o) { varyOutput(parent, o); });
}

static SelfPlayConfig config(int sims, int group) {
    SelfPlayConfig cfg;
    cfg.mcts.simulations = sims;
    cfg.mcts.addNoise = true;
    cfg.mcts.dirichletAlpha = 10.0f / 25.0f;
    cfg.boardSize = 5;
    cfg.maxMoves = defaultMaxMoves(5);
    cfg.temperatureMoves = 8;
    cfg.komi = 24.5f;
    cfg.group = group;
    return cfg;
}

static bool sameRecord(const GameRecord& a, const GameRecord& b) {
    return a.boardSize == b.boardSize && a.komi == b.komi && a.moves == b.moves && a.stones == b.stones &&
           a.policies.size() == b.policies.size() &&
           std::memcmp(a.policies.data(), b.policies.data(), a.policies.size() * sizeof(float)) == 0 &&
           a.rootValues.size() == b.rootValues.size() &&
           std::memcmp(a.rootValues.data(), b.rootValues.data(), a.rootValues.size() * sizeof(float)) == 0 &&
           a.resultForBlack == b.resultForBlack;
}

static void test_selfplay_finishes() {
    int deepest = 0;                         // one worker: the hook runs on one thread
    FakeModel model(kDims, 1, nullptr, [&](const float*, int, FakeOutput& o) {
        assert(o.depth >= 1);
        deepest = std::max(deepest, o.depth);
    });
    const SelfPlayConfig cfg = config(16, 8);
    SelfPlayStats stats;
    const auto recs = playSelfPlay(model, cfg, 4, 11, 1, &stats);
    assert(recs.size() == 4);
    int capped = 0, totalMoves = 0;
    for (const GameRecord& r : recs) {
        assert(r.boardSize == 5 && r.komi == 24.5f);
        assert(r.length() >= 1 && r.length() <= cfg.maxMoves);
        assert(static_cast<int>(r.policies.size()) == r.length() * kActions);
        assert(static_cast<int>(r.rootValues.size()) == r.length());
        // Replay: every move legal, every policy a distribution over the root's legal moves.
        Game g(5, cfg.komi, cfg.maxMoves);
        for (int t = 0; t < r.length(); ++t) {
            assert(!g.isTerminal());
            const float* pi = r.policies.data() + static_cast<size_t>(t) * kActions;
            double sum = 0;
            for (int a = 0; a < kActions; ++a) {
                assert(pi[a] >= 0.0f);
                if (!g.isLegal(a)) assert(pi[a] == 0.0f);
                sum += pi[a];
            }
            assert(std::fabs(sum - 1.0) < 1e-5);
            assert(pi[r.moves[t]] > 0.0f);   // the move played was searched
            g.play(r.moves[t]);
        }
        assert(g.isTerminal());              // two passes or the cap
        assert(r.resultForBlack == g.resultForBlack());
        capped += g.consecutivePasses() < 2;
        totalMoves += r.length();
    }
    assert(stats.cappedGames == capped);
    assert(std::fabs(stats.meanLength - totalMoves / 4.0) < 1e-9);
    assert(stats.maxDepth >= 1 && stats.meanDepth >= 1.0 && stats.meanDepth <= stats.maxDepth);
    assert(stats.maxDepth == deepest);       // each leaf's depth reached the model
    assert(stats.pastEndFraction >= 0.0 && stats.pastEndFraction <= 1.0);
    assert(stats.seconds >= 0.0);
    std::printf("test_selfplay_finishes passed\n");
}

static void test_selfplay_workers_independent() {
    const FakeModel model = varyingModel();
    const SelfPlayConfig cfg = config(12, 2);       // 7 games -> 4 groups
    const auto one = playSelfPlay(model, cfg, 7, 23, 1);
    const auto three = playSelfPlay(model, cfg, 7, 23, 3);
    assert(one.size() == 7 && three.size() == 7);
    for (int i = 0; i < 7; ++i) assert(sameRecord(one[i], three[i]));
    assert(!(one[0].moves == one[1].moves));         // games really differ
    std::printf("test_selfplay_workers_independent passed\n");
}

static void test_lockstep_equals_sequential() {
    const FakeModel model = varyingModel();
    const auto lockstep = playSelfPlay(model, config(12, 4), 6, 31, 1);
    const auto sequential = playSelfPlay(model, config(12, 1), 6, 31, 1);
    for (int i = 0; i < 6; ++i) assert(sameRecord(lockstep[i], sequential[i]));
    std::printf("test_lockstep_equals_sequential passed\n");
}

static void test_muzero_player_legal() {
    const FakeModel model = varyingModel();
    MctsConfig mcts;
    mcts.simulations = 16;
    mcts.addNoise = true;                            // the player must switch it off

    // A wrapper that checks every move against the rules before playing it.
    class Checked : public Player {
    public:
        explicit Checked(const Model& m, const MctsConfig& c) : p_(m, c, 4) {}
        std::string name() const override { return p_.name(); }
        int selectMove(const Game& game, Rng& rng) override {
            const int m = p_.selectMove(game, rng);
            assert(game.isLegal(m));
            const SearchResult r = p_.lastSearch();
            assert(static_cast<int>(r.visits.size()) == kActions && r.visits[m] > 0);
            return m;
        }
    private:
        MuZeroPlayer p_;
    };

    // No noise and argmax (temperatureMoves 0): the search must not depend on the Rng.
    MuZeroPlayer probe(model, mcts, 0);
    assert(probe.name() == "muzero");
    Game start(5, 24.5f, defaultMaxMoves(5));
    Rng r1(1), r2(2);
    const int m1 = probe.selectMove(start, r1);
    const std::vector<int> v1 = probe.lastSearch().visits;
    assert(probe.selectMove(start, r2) == m1 && probe.lastSearch().visits == v1);
    assert(MuZeroPlayer(model, mcts, 4, "mz-a").name() == "mz-a");

    const MatchResult res = playMatch([&] { return std::make_unique<Checked>(model, mcts); },
                                      [] { return std::make_unique<RandomPlayer>(); }, 3, 5, 24.5f, 0, 7, 1);
    assert(res.games.size() == 3 && res.winsA + res.winsB + res.draws == 3);
    for (const auto& g : res.games) assert(!g.moves.empty());
    std::printf("test_muzero_player_legal passed\n");
}

// League games against one random-bot opponent (shared by the next two tests).
static std::vector<GameRecord> leagueVsRandom(SelfPlayStats* stats = nullptr) {
    const FakeModel learner = varyingModel();
    return playLeagueGames(learner, {LeagueOpponent{}}, std::vector<int>(6, 0), config(12, 4), 41, 1, stats);
}

// The learner's turn at position p of game i: Black (even p) in even games.
static bool learnerAt(int i, int p) { return (p % 2 == 0) == (i % 2 == 0); }

static void test_league_random_opponent_masks() {
    SelfPlayStats stats;
    const auto recs = leagueVsRandom(&stats);
    assert(recs.size() == 6);
    int capped = 0, totalMoves = 0;
    for (int i = 0; i < 6; ++i) {
        const GameRecord& r = recs[i];
        const int T = r.length();
        assert(T >= 1 && static_cast<int>(r.policyMask.size()) == T);
        assert(static_cast<int>(r.policies.size()) == T * kActions && static_cast<int>(r.rootValues.size()) == T);
        for (int p = 0; p < T; ++p) {
            assert((r.policyMask[p] == 1) == learnerAt(i, p));
            assert(r.policyMask[p] == 0 || r.policyMask[p] == 1);
            const float* pi = r.policies.data() + static_cast<size_t>(p) * kActions;
            double sum = 0;
            for (int a = 0; a < kActions; ++a) sum += pi[a];
            if (learnerAt(i, p)) {
                assert(std::fabs(sum - 1.0) < 1e-5);
            } else {
                assert(sum == 0.0 && r.rootValues[p] == 0.0f);
            }
        }
        capped += !r.endedByPasses();
        totalMoves += T;
    }
    assert(stats.cappedGames == capped);
    assert(std::fabs(stats.meanLength - totalMoves / 6.0) < 1e-9);
    assert(stats.maxDepth >= 1 && stats.meanDepth >= 1.0 && stats.meanDepth <= stats.maxDepth);
    std::printf("test_league_random_opponent_masks passed\n");
}

static void test_league_random_bot_never_passes_early() {
    const auto recs = leagueVsRandom();
    int botMoves = 0, botStones = 0;
    for (int i = 0; i < 6; ++i) {
        const GameRecord& r = recs[i];
        Game g(5, r.komi, defaultMaxMoves(5));
        for (int t = 0; t < r.length(); ++t) {
            assert(!g.isTerminal() && g.isLegal(r.moves[t]));
            if (!learnerAt(i, t)) {
                ++botMoves;
                // RandomPlayer's rule: a pass only when every legal non-pass move fills an own eye.
                bool other = false;
                for (int a : g.legalActions())
                    if (a != g.board().passAction() && !fillsOwnEye(g.board(), a)) other = true;
                if (r.moves[t] == g.board().passAction()) assert(!other);
                else ++botStones;
            }
            g.play(r.moves[t]);
        }
        assert(g.isTerminal() && r.resultForBlack == g.resultForBlack());
    }
    assert(botMoves > 0 && botStones > 0);
    std::printf("test_league_random_bot_never_passes_early passed\n");
}

// A root hook putting (almost) all prior mass on the lowest (or highest) empty point.
static FakeModel::RootHook preferEmpty(bool lowest) {
    return [lowest](const float* obs, FakeOutput& o) {
        const int area = kDims.area();
        for (int k = 0; k < area; ++k) {
            const int p = lowest ? k : area - 1 - k;
            if (obs[p] == 0.0f && obs[area + p] == 0.0f) { o.logits[p] = 30.0f; return; }
        }
    };
}
static int preferredPoint(const Board& b, bool lowest) {
    for (int k = 0; k < b.area(); ++k) {
        const int p = lowest ? k : b.area() - 1 - k;
        if (b.at(p) == Color::Empty) return p;
    }
    return -1;
}

static void test_league_network_opponent_used() {
    const FakeModel learner(kDims, 1, preferEmpty(true));
    const FakeModel opponent(kDims, 1, preferEmpty(false));
    SelfPlayConfig cfg = config(8, 4);
    cfg.mcts.addNoise = false;
    cfg.temperatureMoves = 0;
    const auto recs = playLeagueGames(learner, {LeagueOpponent{&opponent}}, {0, 0, 0}, cfg, 5, 1);
    assert(recs.size() == 3);
    int checked[2] = {0, 0};                 // [learner, opponent] moves at a legal preferred point
    for (int i = 0; i < 3; ++i) {
        const GameRecord& r = recs[i];
        assert(static_cast<int>(r.policyMask.size()) == r.length());
        Game g(5, r.komi, cfg.maxMoves);
        for (int t = 0; t < r.length(); ++t) {
            const bool mine = learnerAt(i, t);
            assert(r.policyMask[t] == (mine ? 1 : 0));
            // Both sides' rows are search distributions.
            const float* pi = r.policies.data() + static_cast<size_t>(t) * kActions;
            double sum = 0;
            for (int a = 0; a < kActions; ++a) sum += pi[a];
            assert(std::fabs(sum - 1.0) < 1e-5 && pi[r.moves[t]] > 0.0f);
            const int want = preferredPoint(g.board(), mine);
            if (want >= 0 && g.isLegal(want)) {
                assert(r.moves[t] == want);
                ++checked[mine ? 0 : 1];
            }
            g.play(r.moves[t]);
        }
    }
    assert(checked[0] >= 6 && checked[1] >= 6);
    std::printf("test_league_network_opponent_used passed\n");
}

// Games against the random bot that the move cap stops keep a full mask with the parity pattern.
static void test_league_capped_games_masks() {
    const FakeModel learner = varyingModel();
    SelfPlayConfig cfg = config(8, 3);
    cfg.maxMoves = 7;
    const auto recs = playLeagueGames(learner, {LeagueOpponent{}}, std::vector<int>(5, 0), cfg, 67, 2);
    assert(recs.size() == 5);
    int capped = 0;
    for (int i = 0; i < 5; ++i) {
        const GameRecord& r = recs[i];
        const int T = r.length();
        assert(T >= 1 && T <= cfg.maxMoves && static_cast<int>(r.policyMask.size()) == T);
        for (int p = 0; p < T; ++p) assert((r.policyMask[p] == 1) == learnerAt(i, p));
        capped += !r.endedByPasses();
    }
    assert(capped >= 1);
    std::printf("test_league_capped_games_masks passed\n");
}

// Root hook shared by both sides: prior mass split between the lowest and the
// highest empty point, whose indices it also writes into hidden[0] and hidden[1]
// so a recurrent hook at depth 1 can recognize them.
static void splitPrior(const float* obs, FakeOutput& o) {
    const int area = kDims.area();
    int lo = -1, hi = -1;
    for (int p = 0; p < area; ++p)
        if (obs[p] == 0.0f && obs[area + p] == 0.0f) { if (lo < 0) lo = p; hi = p; }
    if (lo < 0) return;
    o.logits[lo] = o.logits[hi] = 30.0f;
    o.hidden[0] = (static_cast<float>(lo) + 0.5f) / 32.0f;
    o.hidden[1] = (static_cast<float>(hi) + 0.5f) / 32.0f;
}
// Recurrent hook: reward 1 for the root move to the point stored at hidden[slot].
static FakeModel::StepHook rewardPoint(int slot) {
    return [slot](const float* parent, int action, FakeOutput& o) {
        if (o.depth == 1 && action == static_cast<int>(parent[slot] * 32.0f)) o.reward = 1.0f;
    };
}

static void test_league_opponent_recurrent_model_used() {
    // Same root inference for both: only the dynamics decide between the two points.
    const FakeModel learner(kDims, 1, splitPrior, rewardPoint(0));    // likes the lowest empty point
    const FakeModel opponent(kDims, 1, splitPrior, rewardPoint(1));   // likes the highest
    SelfPlayConfig cfg = config(16, 4);
    cfg.mcts.addNoise = false;
    cfg.temperatureMoves = 0;
    const auto recs = playLeagueGames(learner, {LeagueOpponent{&opponent}}, {0, 0, 0}, cfg, 13, 1);
    int checked[2] = {0, 0};                 // [learner, opponent]
    for (int i = 0; i < 3; ++i) {
        const GameRecord& r = recs[i];
        Game g(5, r.komi, cfg.maxMoves);
        for (int t = 0; t < r.length(); ++t) {
            const bool mine = learnerAt(i, t);
            const int lo = preferredPoint(g.board(), true), hi = preferredPoint(g.board(), false);
            if (lo >= 0 && lo != hi && g.isLegal(lo) && g.isLegal(hi)) {
                assert(r.moves[t] == (mine ? lo : hi));
                ++checked[mine ? 0 : 1];
            }
            g.play(r.moves[t]);
        }
    }
    assert(checked[0] >= 6 && checked[1] >= 6);
    std::printf("test_league_opponent_recurrent_model_used passed\n");
}

static void test_league_workers_independent() {
    const FakeModel learner = varyingModel();
    const FakeModel net(kDims, 2, [](const float*, FakeOutput& o) { varyOutput(o.hidden.data(), o); o.value *= -1; },
                        [](const float* parent, int, FakeOutput& o) { varyOutput(parent, o); o.logits[3] += 1.0f; });
    const std::vector<LeagueOpponent> opponents{LeagueOpponent{}, LeagueOpponent{&net}};
    const std::vector<int> opponentOf{1, 0, 0, 1, 1, 0, 1, 0, 0, 1};
    const SelfPlayConfig cfg = config(12, 3);
    SelfPlayStats s1, s3;
    const auto one = playLeagueGames(learner, opponents, opponentOf, cfg, 59, 1, &s1);
    const auto three = playLeagueGames(learner, opponents, opponentOf, cfg, 59, 3, &s3);
    assert(one.size() == 10 && three.size() == 10);
    for (int i = 0; i < 10; ++i) assert(sameRecord(one[i], three[i]) && one[i].policyMask == three[i].policyMask);
    assert(s1.meanLength == s3.meanLength && s1.meanDepth == s3.meanDepth && s1.maxDepth == s3.maxDepth &&
           s1.pastEndFraction == s3.pastEndFraction && s1.cappedGames == s3.cappedGames);
    // Against the network opponent, the opponent's rows are search distributions.
    for (int i = 0; i < 10; ++i) {
        const float* pi = one[i].policies.data() + static_cast<size_t>(learnerAt(i, 0) ? 1 : 0) * kActions;
        double sum = 0;
        for (int a = 0; a < kActions; ++a) sum += pi[a];
        assert(std::fabs(sum - (opponentOf[i] == 1 ? 1.0 : 0.0)) < 1e-5);
    }
    assert(!(one[0].moves == one[3].moves));          // games really differ
    std::printf("test_league_workers_independent passed\n");
}

int main() {
    test_selfplay_finishes();
    test_selfplay_workers_independent();
    test_lockstep_equals_sequential();
    test_muzero_player_legal();
    test_league_random_opponent_masks();
    test_league_random_bot_never_passes_early();
    test_league_network_opponent_used();
    test_league_workers_independent();
    test_league_capped_games_masks();
    test_league_opponent_recurrent_model_used();
    std::printf("test_selfplay: all tests passed\n");
    return 0;
}
