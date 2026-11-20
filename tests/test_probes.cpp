#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include "mzgo/backend.hpp"
#include "mzgo/encode.hpp"
#include "mzgo/game.hpp"
#include "mzgo/game_record.hpp"
#include "mzgo/mcts.hpp"
#include "mzgo/network.hpp"
#include "mzgo/probes.hpp"

using namespace mzgo;

namespace {

// A small randomly initialized 5x5 network (never trained).
NetConfig smallConfig() {
    NetConfig cfg;
    cfg.boardSize = 5;
    cfg.history = 2;
    cfg.reprBlocks = 1;
    cfg.dynBlocks = 1;
    cfg.filters = 16;
    cfg.valueHidden = 8;
    return cfg;
}

} // namespace

static void test_probe_games() {
    const std::vector<GameRecord> one = probeGames(5, 24.5f, 3, 17, 1);
    const std::vector<GameRecord> many = probeGames(5, 24.5f, 3, 17, 3);
    assert(one.size() == 3 && many.size() == 3);
    for (size_t i = 0; i < one.size(); ++i) {
        const GameRecord& r = one[i];
        assert(r.moves == many[i].moves && r.stones == many[i].stones);
        assert(r.boardSize == 5 && r.komi == 24.5f && r.length() > 0);
        assert(r.stones.size() == static_cast<size_t>(r.length() + 1) * r.area());
        // One-hot policy of the move played, root values 0.
        for (int p = 0; p < r.length(); ++p) {
            const float* pol = r.policies.data() + static_cast<size_t>(p) * r.numActions();
            float sum = 0;
            for (int a = 0; a < r.numActions(); ++a) sum += pol[a];
            assert(sum == 1.0f && pol[r.moves[p]] == 1.0f && r.rootValues[p] == 0.0f);
        }
    }
    std::printf("test_probe_games passed\n");
}

static void test_latent_probe_k0_is_zero() {
    NaiveBackend be;
    MuZeroNetwork net(smallConfig(), 3);
    NetworkModel model(net, be);
    const std::vector<GameRecord> games = probeGames(5, 24.5f, 4, 5, 0);
    const std::vector<LatentRow> rows = latentProbe(model, games, 3);
    assert(rows.size() == 4);
    long positions = 0;
    for (const GameRecord& g : games) positions += g.length();
    assert(rows[0].k == 0 && rows[0].samples == positions);
    assert(rows[0].valueError == 0.0 && rows[0].policyTv == 0.0);   // imagined = fresh
    for (int k = 1; k <= 3; ++k) {
        assert(rows[k].k == k && rows[k].samples > 0 && rows[k].samples < rows[k - 1].samples);
        assert(rows[k].valueError > 0.0 && rows[k].valueError <= 2.0);
        assert(rows[k].policyTv > 0.0 && rows[k].policyTv <= 1.0);
    }
    std::printf("test_latent_probe_k0_is_zero passed\n");
}

static void test_rules_probe_counts() {
    NaiveBackend be;
    const NetConfig cfg = smallConfig();
    MuZeroNetwork net(cfg, 7);
    NetworkModel model(net, be);

    // Two black stones on the board; Black plays C3, so the position after
    // one move has exactly 3 occupied points. White then passes, Black passes.
    const Board start = boardFromRows({".....", ".X...", ".....", "...X.", "....."}, Color::Black);
    Game game(start, 24.5f, defaultMaxMoves(5));
    const std::vector<int> moves = {parseAction("C3", 5), 25, 25};
    for (int m : moves) game.play(m);
    assert(game.isTerminal());
    std::vector<float> policies(moves.size() * 26, 0.0f);
    for (size_t p = 0; p < moves.size(); ++p) policies[p * 26 + moves[p]] = 1.0f;
    const GameRecord rec = recordFromGame(game, policies, std::vector<float>(moves.size(), 0.0f));

    const std::vector<RulesRow> rows = rulesProbe(model, {rec}, 2);
    assert(rows.size() == 3);
    assert(rows[0].samples == 3 && rows[1].samples == 2 && rows[2].samples == 1);

    // Direct computation for k = 1 from t = 0 (the only start reaching position 1
    // is t = 0; t = 1 reaches position 2): mass on the 3 stones of position 1.
    std::vector<float> obs(static_cast<size_t>(cfg.planes()) * 25);
    auto session = model.newSession(1);
    ModelOutput out;
    std::vector<float> probs(26);
    double occupied0to1 = 0;
    encodeRecordPosition(rec, 0, cfg.history, 0, obs.data());
    session->initialInference(obs.data(), 1, out);
    session->recurrentInference(out.hidden.data(), &rec.moves[0], 1, out);
    softmax(out.logits.data(), 26, probs.data());
    for (int pt : {parseAction("B4", 5), parseAction("D2", 5), parseAction("C3", 5)}) occupied0to1 += probs[pt];
    // From t = 1 to position 2 (White passed): the same 3 stones.
    double occupied1to2 = 0;
    encodeRecordPosition(rec, 1, cfg.history, 0, obs.data());
    session->initialInference(obs.data(), 1, out);
    session->recurrentInference(out.hidden.data(), &rec.moves[1], 1, out);
    softmax(out.logits.data(), 26, probs.data());
    for (int pt : {parseAction("B4", 5), parseAction("D2", 5), parseAction("C3", 5)}) occupied1to2 += probs[pt];

    const double expected = (occupied0to1 + occupied1to2) / 2.0;
    assert(std::fabs(rows[1].occupiedMass - expected) < 1e-5);
    // Occupied points are illegal; nothing was captured, no superko on this board.
    assert(std::fabs(rows[1].illegalMass - rows[1].occupiedMass) < 1e-9);
    assert(rows[1].capturedPointMass == 0.0 && rows[1].koSuperkoMass == 0.0);
    std::printf("test_rules_probe_counts passed\n");
}

static void test_rules_probe_captured_points() {
    NaiveBackend be;
    MuZeroNetwork net(smallConfig(), 9);
    NetworkModel model(net, be);
    // A white stone in the corner (A5) with Black on B5 has one liberty, A4;
    // Black plays there and captures it, so A5 is freed after one move.
    const Board start = boardFromRows({"OX...", ".....", ".....", ".....", "....."}, Color::Black);
    Game game(start, 24.5f, defaultMaxMoves(5));
    const std::vector<int> moves = {parseAction("A4", 5), 25, 25};
    for (int m : moves) game.play(m);
    assert(game.boardAt(1).at(0) == Color::Empty);   // the white stone is gone
    std::vector<float> policies(moves.size() * 26, 0.0f);
    for (size_t p = 0; p < moves.size(); ++p) policies[p * 26 + moves[p]] = 1.0f;
    const GameRecord rec = recordFromGame(game, policies, std::vector<float>(moves.size(), 0.0f));
    const std::vector<RulesRow> rows = rulesProbe(model, {rec}, 1);
    assert(rows[0].capturedPointMass == 0.0);
    assert(rows[1].capturedPointMass > 0.0 && rows[1].capturedPointMass < 1.0);
    std::printf("test_rules_probe_captured_points passed\n");
}

static void test_board_probe_learns_k0() {
    NaiveBackend be;
    MuZeroNetwork net(smallConfig(), 11);
    NetworkModel model(net, be);
    const std::vector<GameRecord> games = probeGames(5, 24.5f, 40, 13, 0);
    const BoardProbe probe = fitBoardProbe(model, games, 10, 21);
    assert(probe.w.size() == 3u * 16 && probe.b.size() == 3);

    // Majority-class rate over every point of every position (the k = 0 samples).
    long counts[3] = {0, 0, 0}, points = 0;
    for (const GameRecord& g : games) {
        for (int p = 0; p <= g.length(); ++p) {
            for (int i = 0; i < g.area(); ++i) {
                const Color c = static_cast<Color>(g.stones[static_cast<size_t>(p) * g.area() + i]);
                counts[c == Color::Empty ? 0 : (c == g.toMoveAt(p) ? 1 : 2)]++;
                ++points;
            }
        }
    }
    const double majority = static_cast<double>(*std::max_element(counts, counts + 3)) / points;

    const std::vector<BoardRow> rows = boardProbe(model, probe, games, 2);
    assert(rows.size() == 3 && rows[0].k == 0 && rows[0].points == points);
    assert(rows[0].changedPoints == 0);
    assert(rows[0].accuracy >= majority);
    assert(rows[1].changedPoints > 0 && rows[1].changedAccuracy >= 0.0 && rows[1].changedAccuracy <= 1.0);
    std::printf("test_board_probe_learns_k0 passed\n");
}

static void test_analyze_runs() {
    NaiveBackend be;
    MuZeroNetwork net(smallConfig(), 15);
    NetworkModel model(net, be);
    Game game(5, 24.5f, defaultMaxMoves(5));
    for (const char* m : {"C3", "D4", "B2"}) game.play(parseAction(m, 5));
    MctsConfig cfg;
    cfg.simulations = 40;
    const std::string text = analyzePosition(model, game, cfg, 5);
    assert(!text.empty() && text.find("visits") != std::string::npos);
    assert(text.find("principal variation") != std::string::npos);

    // The principal variation starts with the most visited root move.
    Search search(cfg, model.dims());
    auto session = model.newSession(1);
    Rng rng(1);
    runSearch(search, *session, game, model.history(), rng);
    const std::vector<std::pair<int, float>> pv = search.principalVariation(8);
    assert(!pv.empty() && pv.size() <= 8);
    int best = -1, bestVisits = -1;
    for (const RootEdge& e : search.rootEdges())
        if (e.visits > bestVisits) best = e.action, bestVisits = e.visits;
    assert(pv[0].first == best);
    for (const auto& step : pv) assert(step.second >= -1.0f && step.second <= 1.0f);
    std::printf("test_analyze_runs passed\n");
}

int main() {
    test_probe_games();
    test_latent_probe_k0_is_zero();
    test_rules_probe_counts();
    test_rules_probe_captured_points();
    test_board_probe_learns_k0();
    test_analyze_runs();
    return 0;
}
