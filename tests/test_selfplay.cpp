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

// A fake whose policy and value vary with the state (pure functions of the
// inputs, so thread-safe): searches then differ from move to move and from
// game to game, which makes the equality tests below meaningful.
static void varyOutput(const float* hidden, FakeOutput& o) {
    for (int a = 0; a < kActions; ++a) o.logits[a] = 2.0f * o.hidden[a % kDims.hiddenSize()];
    o.value = 2.0f * hidden[0] - 1.0f;
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
    FakeModel model(kDims);
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

int main() {
    test_selfplay_finishes();
    test_selfplay_workers_independent();
    test_lockstep_equals_sequential();
    test_muzero_player_legal();
    std::printf("test_selfplay: all tests passed\n");
    return 0;
}
