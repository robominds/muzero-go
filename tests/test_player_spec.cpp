#include <cassert>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include "mzgo/backend.hpp"
#include "mzgo/checkpoint.hpp"
#include "mzgo/game.hpp"
#include "mzgo/network.hpp"
#include "mzgo/player_spec.hpp"

using namespace mzgo;
namespace fs = std::filesystem;

static PlayerSpecOptions options() {
    PlayerSpecOptions o;
    o.boardSize = 5;
    o.simulations = 4;
    o.temperatureMoves = 2;
    o.backend = blasAvailable() ? "blas" : "naive";
    return o;
}

static void expectInvalid(const std::string& spec, const PlayerSpecOptions& o) {
    try {
        makePlayerFactory(spec, o);
    } catch (const std::invalid_argument& e) {
        assert(std::string(e.what()).find(spec) != std::string::npos);
        return;
    }
    assert(false && "expected std::invalid_argument");
}

static void test_player_spec() {
    const PlayerSpecOptions o = options();
    assert(makePlayerFactory("random", o)()->name() == "random");
    assert(makePlayerFactory("rollout:16", o)()->name() == "rollout-mcts-16");

    // ckpt: a tiny 5x5 network; its players search it and pick legal moves.
    const std::string path = "test_player_spec.mzn";
    NetConfig cfg;
    cfg.boardSize = 5; cfg.history = 2; cfg.reprBlocks = 1; cfg.dynBlocks = 1; cfg.filters = 4; cfg.valueHidden = 4;
    saveNetwork(MuZeroNetwork(cfg, 7), path);
    const PlayerFactory ckpt = makePlayerFactory("ckpt:" + path, o);
    fs::remove(path);   // loaded once, when the factory was made
    auto p1 = ckpt(), p2 = ckpt();
    assert(p1->name() == "ckpt:" + path);
    Game g(5, 0.5f, defaultMaxMoves(5));
    Rng rng(3);
    const int move = p1->selectMove(g, rng);
    assert(g.isLegal(move));
    assert(g.isLegal(p2->selectMove(g, rng)));

    expectInvalid("foo", o);
    expectInvalid("rollout:", o);
    expectInvalid("rollout:abc", o);
    expectInvalid("rollout:0", o);
    expectInvalid("ckpt:", o);
    // A checkpoint for another board size is refused.
    saveNetwork(MuZeroNetwork(cfg, 7), path);
    PlayerSpecOptions nine = o;
    nine.boardSize = 9;
    expectInvalid("ckpt:" + path, nine);
    fs::remove(path);
    // A missing checkpoint is the loader's runtime_error.
    try {
        makePlayerFactory("ckpt:no_such_file.mzn", o);
        assert(false && "expected std::runtime_error");
    } catch (const std::runtime_error& e) {
        assert(std::string(e.what()).find("no_such_file.mzn") != std::string::npos);
    }
    std::printf("test_player_spec passed\n");
}

int main() {
    test_player_spec();
    return 0;
}
