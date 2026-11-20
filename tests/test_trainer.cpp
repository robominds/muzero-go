#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "mzgo/backend.hpp"
#include "mzgo/checkpoint.hpp"
#include "mzgo/config.hpp"
#include "mzgo/game_record.hpp"
#include "mzgo/replay_buffer.hpp"
#include "mzgo/rng.hpp"
#include "mzgo/serialize.hpp"
#include "mzgo/trainer.hpp"

using namespace mzgo;
namespace fs = std::filesystem;

// Every run of these tests lives under one scratch directory next to the
// test binary, wiped at the start and the end.
static const std::string kRoot = "test_trainer_runs";

static std::string freshDir(const std::string& name) {
    const std::string dir = kRoot + "/" + name;
    fs::remove_all(dir);
    return dir;
}

// The tiny run from the brief: 5x5, one block each, 8 filters, 4 games of 8
// simulations, 4 steps of batch 8, 2 arena games, an 8-simulation rollout bot.
static RunConfig tinyConfig() {
    RunConfig c;
    c.boardSize = 5;
    c.reprBlocks = 1; c.dynBlocks = 1; c.filters = 8; c.valueHidden = 8;
    c.gamesPerGeneration = 4; c.simulations = 8; c.trainSteps = 4; c.batchSize = 8;
    c.arenaGames = 2; c.arenaSimulations = 8; c.rolloutBotSimulations = 8;
    c.generations = 2; c.workers = 2; c.trainThreads = 2;
    c.backend = blasAvailable() ? "blas" : "naive";
    return c;
}

static std::string readFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    assert(in);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static std::vector<std::string> lines(const std::string& text) {
    std::vector<std::string> out;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) out.push_back(line);
    return out;
}

static std::vector<std::string> splitCsv(const std::string& line) {
    std::vector<std::string> out;
    std::string field;
    std::istringstream in(line);
    while (std::getline(in, field, ',')) out.push_back(field);
    if (!line.empty() && line.back() == ',') out.push_back("");
    return out;
}

static bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

static void test_net_config_from() {
    RunConfig c = tinyConfig();
    c.history = 4; c.rewardHead = true; c.passPlane = true; c.reprBlocks = 2; c.dynBlocks = 3;
    const NetConfig n = netConfigFrom(c);
    assert(n.boardSize == 5 && n.history == 4 && n.reprBlocks == 2 && n.dynBlocks == 3 && n.filters == 8 &&
           n.valueHidden == 8 && n.rewardHead && n.passPlane && n.planes() == 11);
    std::printf("test_net_config_from passed\n");
}

static void test_two_generations() {
    const std::string dir = freshDir("two");
    {
        Trainer t = Trainer::startNew(tinyConfig(), dir);
        assert(t.generation() == 0);
        t.run();
        assert(t.generation() == 2);
    }
    assert(fs::exists(dir + "/config.cfg"));
    assert(parseConfigFile(dir + "/config.cfg").filters == 8);
    for (int g = 0; g <= 2; ++g) assert(fs::exists(checkpointPath(dir, g)));
    assert(checkpointPath(dir, 2) == dir + "/checkpoints/gen_0002.mzn");
    assert(fs::exists(dir + "/sgf/gen_0001_0.sgf") && fs::exists(dir + "/sgf/gen_0002_1.sgf"));
    assert(!fs::exists(dir + "/sgf/gen_0002_2.sgf"));

    const std::vector<std::string> log = lines(readFile(dir + "/log.csv"));
    assert(log.size() == 3);
    const std::string header =
        "generation,seconds_selfplay,seconds_train,games,mean_length,capped_games,mean_depth,max_depth,"
        "past_end_fraction,lr,loss_total,policy_k0,policy_k1,policy_k2,policy_k3,policy_k4,policy_k5,"
        "value_k0,value_k1,value_k2,value_k3,value_k4,value_k5,reward_k1,reward_k2,reward_k3,reward_k4,"
        "reward_k5,arena_prev,arena_rollout,arena_random,replay_games,replay_positions";
    assert(log[0] == header);
    const size_t columns = splitCsv(header).size();
    for (int g = 1; g <= 2; ++g) {
        const std::vector<std::string> row = splitCsv(log[g]);
        assert(row.size() == columns);
        assert(row[0] == std::to_string(g));
        assert(row[3] == "4");                                   // games
        assert(std::isfinite(std::stod(row[10])));               // loss_total
        for (size_t a = columns - 5; a < columns - 2; ++a) {     // the arena ran every generation
            const double score = std::stod(row[a]);
            assert(score >= 0 && score <= 1);
        }
        assert(row[columns - 2] == std::to_string(4 * g));       // replay_games
    }

    // state.bin loads, and the resumed trainer is at generation 2.
    Trainer resumed = Trainer::resume(dir, {});
    assert(resumed.generation() == 2 && resumed.globalStep() == 8);
    std::printf("test_two_generations passed\n");
}

static void test_resume_matches_uninterrupted() {
    const std::string straight = freshDir("straight"), split = freshDir("split");
    Trainer::startNew(tinyConfig(), straight).run();

    RunConfig one = tinyConfig();
    one.generations = 1;
    Trainer::startNew(one, split).run();
    Trainer second = Trainer::resume(split, {"generations=2"});
    assert(second.generation() == 1 && second.config().generations == 2);
    second.run();
    assert(second.generation() == 2);

    assert(readFile(checkpointPath(straight, 2)) == readFile(checkpointPath(split, 2)));
    assert(readFile(straight + "/state.bin") == readFile(split + "/state.bin"));
    // The resumed run appended its row to the same log.
    assert(lines(readFile(split + "/log.csv")).size() == 3);
    // The override was saved, so the next resume sees generations = 2.
    assert(parseConfigFile(split + "/config.cfg").generations == 2);
    std::printf("test_resume_matches_uninterrupted passed\n");
}

static void test_threads_independent() {
    std::string bytes[2];
    for (int i = 0; i < 2; ++i) {
        RunConfig c = tinyConfig();
        c.generations = 1;
        c.arenaEvery = 0;   // the arena does not touch the checkpoint
        c.workers = i == 0 ? 1 : 3;
        c.trainThreads = i == 0 ? 1 : 3;
        const std::string dir = freshDir("threads" + std::to_string(i));
        Trainer::startNew(c, dir).run();
        bytes[i] = readFile(checkpointPath(dir, 1));
        // Arena off: the arena columns are empty.
        const std::vector<std::string> row = splitCsv(lines(readFile(dir + "/log.csv"))[1]);
        assert(row[row.size() - 5].empty() && row[row.size() - 4].empty() && row[row.size() - 3].empty());
    }
    assert(bytes[0] == bytes[1]);
    std::printf("test_threads_independent passed\n");
}

static void test_resume_refuses_arch_change() {
    const std::string dir = freshDir("arch");
    Trainer::startNew(tinyConfig(), dir);
    try {
        Trainer::resume(dir, {"filters=16"});
        assert(false && "expected std::invalid_argument");
    } catch (const std::invalid_argument& e) {
        assert(contains(e.what(), "filters"));
    }
    // unroll_steps is an architecture key too: batch norm keeps statistics
    // per unroll step (and log.csv has columns per step).
    try {
        Trainer::resume(dir, {"unroll_steps=3"});
        assert(false && "expected std::invalid_argument");
    } catch (const std::invalid_argument& e) {
        assert(contains(e.what(), "unroll_steps") && contains(e.what(), "architecture key"));
    }
    try {
        Trainer::resume(dir, {"pass_plane=true"});
        assert(false && "expected std::invalid_argument");
    } catch (const std::invalid_argument& e) {
        assert(contains(e.what(), "pass_plane") && contains(e.what(), "architecture key"));
    }
    // Restating an architecture key's value is no change; tunable keys may change.
    Trainer ok = Trainer::resume(dir, {"filters=8", "simulations=4"});
    assert(ok.config().simulations == 4);
    std::printf("test_resume_refuses_arch_change passed\n");
}

static void test_corrupt_state() {
    const std::string dir = freshDir("corrupt");
    Trainer::startNew(tinyConfig(), dir);
    const std::string path = dir + "/state.bin";
    const std::string full = readFile(path);
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(full.data(), static_cast<std::streamsize>(full.size() / 2));
    }
    try {
        Trainer::resume(dir, {});
        assert(false && "expected std::runtime_error");
    } catch (const std::runtime_error& e) {
        assert(contains(e.what(), path));
    }
    std::printf("test_corrupt_state passed\n");
}

// A state.bin whose replay games are of another board size than the
// config's is refused, naming the file.
static void test_resume_refuses_replay_board_size() {
    const std::string dir = freshDir("replay_size");
    const RunConfig cfg = tinyConfig();
    Trainer::startNew(cfg, dir);
    const std::string path = dir + "/state.bin";
    const std::string full = readFile(path);
    // A new run's replay buffer is empty and last: magic, version, capacity, count.
    const size_t emptyReplay = 16;
    assert(full.compare(full.size() - emptyReplay, 4, "MZRB") == 0);
    GameRecord g;
    g.boardSize = 7;
    g.komi = cfg.komi;
    g.moves = {0, 1};
    g.stones.assign(3u * 49, 0);
    g.policies.assign(2u * 50, 0.0f);
    g.rootValues.assign(2, 0.0f);
    ReplayBuffer replay(cfg.replayGames);
    replay.add(g);
    std::ostringstream tail(std::ios::binary);
    { BinaryWriter w(tail); replay.save(w); }
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out << full.substr(0, full.size() - emptyReplay) << tail.str();
    }
    try {
        Trainer::resume(dir, {});
        assert(false && "expected std::runtime_error");
    } catch (const std::runtime_error& e) {
        assert(contains(e.what(), path) && contains(e.what(), "board size"));
    }
    std::printf("test_resume_refuses_replay_board_size passed\n");
}

static void test_refusals() {
    const std::string dir = freshDir("refuse");
    Trainer::startNew(tinyConfig(), dir);
    try {   // a new run into a directory that is not empty
        Trainer::startNew(tinyConfig(), dir);
        assert(false && "expected std::runtime_error");
    } catch (const std::runtime_error& e) {
        assert(contains(e.what(), dir));
    }
    fs::remove(dir + "/state.bin");
    try {   // a resume without state.bin
        Trainer::resume(dir, {});
        assert(false && "expected std::runtime_error");
    } catch (const std::runtime_error& e) {
        assert(contains(e.what(), "state.bin"));
    }
    std::printf("test_refusals passed\n");
}

static void test_nan_loss_stops() {
    const std::string dir = freshDir("nan");
    RunConfig c = tinyConfig();
    c.lrInit = 1e30f;
    Trainer t = Trainer::startNew(c, dir);
    const std::string state = readFile(dir + "/state.bin");
    try {
        t.runGeneration();
        assert(false && "expected std::runtime_error");
    } catch (const std::runtime_error& e) {
        assert(contains(e.what(), "non-finite loss at generation 1 step"));
    }
    assert(!fs::exists(checkpointPath(dir, 1)));
    assert(readFile(dir + "/state.bin") == state);
    assert(lines(readFile(dir + "/log.csv")).size() == 1);
    std::printf("test_nan_loss_stops passed\n");
}

// The last step's update is checked too: one step leaves a finite loss
// (computed before the update) but infinite weights, which must not reach
// the checkpoint or state.bin. The learning rate alone is not enough (every
// |velocity| can stay below 1, keeping lr * v finite even at float's
// maximum), so a large weight decay makes v ~ 1000 * w and the update overflows.
static void test_nan_weights_stop() {
    const std::string dir = freshDir("nanweights");
    RunConfig c = tinyConfig();
    c.trainSteps = 1;
    c.lrInit = std::numeric_limits<float>::max();
    c.weightDecay = 1000.0f;
    Trainer t = Trainer::startNew(c, dir);
    const std::string state = readFile(dir + "/state.bin");
    try {
        t.runGeneration();
        assert(false && "expected std::runtime_error");
    } catch (const std::runtime_error& e) {
        assert(contains(e.what(), "non-finite weights at generation 1 step 1"));
    }
    assert(!fs::exists(checkpointPath(dir, 1)));
    assert(readFile(dir + "/state.bin") == state);
    assert(lines(readFile(dir + "/log.csv")).size() == 1);
    std::printf("test_nan_weights_stop passed\n");
}

// bn_calibrate replaces generation 0's initial running statistics (mean 0,
// variance 1) by measured ones.
static void test_bn_calibrate() {
    const std::string dir = freshDir("calibrate");
    RunConfig c = tinyConfig();
    c.bnCalibrate = true;
    Trainer::startNew(c, dir);
    MuZeroNetwork calibrated = loadNetwork(checkpointPath(dir, 0));
    MuZeroNetwork fresh(netConfigFrom(c), deriveSeed(c.seed, 0xA11));
    const std::vector<BufferView> a = calibrated.buffers(), b = fresh.buffers();
    assert(a.size() == b.size());
    size_t changed = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        bool differs = false;
        for (size_t j = 0; j < a[i].size; ++j) {
            assert(std::isfinite(a[i].value[j]));
            if (a[i].value[j] != b[i].value[j]) differs = true;
        }
        if (differs) ++changed;
    }
    assert(changed == a.size());   // every batch norm (h, g and f) was calibrated
    std::printf("test_bn_calibrate passed\n");
}

int main() {
    fs::remove_all(kRoot);
    test_net_config_from();
    test_two_generations();
    test_resume_matches_uninterrupted();
    test_threads_independent();
    test_resume_refuses_arch_change();
    test_corrupt_state();
    test_resume_refuses_replay_board_size();
    test_refusals();
    test_nan_loss_stops();
    test_nan_weights_stop();
    test_bn_calibrate();
    fs::remove_all(kRoot);
    return 0;
}
