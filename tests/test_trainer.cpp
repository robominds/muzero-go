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
#include "mzgo/league.hpp"
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
    // A new run's replay buffer is empty (magic, version, capacity, count),
    // followed only by the empty league (its member count).
    const size_t emptyReplay = 16, emptyLeague = 4;
    assert(full.compare(full.size() - emptyLeague - emptyReplay, 4, "MZRB") == 0);
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
        out << full.substr(0, full.size() - emptyLeague - emptyReplay) << tail.str()
            << full.substr(full.size() - emptyLeague);
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


// ---- League play ----------------------------------------------------------

// The tiny run with a league of two random bots, where any score replaces a
// member: 4 games per generation, 2 of them league games (one per slot).
static RunConfig leagueConfig() {
    RunConfig c = tinyConfig();
    c.leagueSize = 2;
    c.leagueReplaceScore = 0;
    c.leagueMinGames = 0;
    return c;
}

static const std::string kTodaysHeader =
    "generation,seconds_selfplay,seconds_train,games,mean_length,capped_games,mean_depth,max_depth,"
    "past_end_fraction,lr,loss_total,policy_k0,policy_k1,policy_k2,policy_k3,policy_k4,policy_k5,"
    "value_k0,value_k1,value_k2,value_k3,value_k4,value_k5,reward_k1,reward_k2,reward_k3,reward_k4,"
    "reward_k5,arena_prev,arena_rollout,arena_random,replay_games,replay_positions";

static std::vector<std::string> splitOn(const std::string& s, char sep) {
    std::vector<std::string> out;
    size_t start = 0;
    for (size_t i = 0; i <= s.size(); ++i)
        if (i == s.size() || s[i] == sep) {
            out.push_back(s.substr(start, i - start));
            start = i + 1;
        }
    return out;
}

static void test_league_generation() {
    const std::string dir = freshDir("league");
    RunConfig c = leagueConfig();
    c.generations = 1;
    Trainer t = Trainer::startNew(c, dir);
    assert(t.league().size() == 2 && t.league()[0].isRandom() && t.league()[1].isRandom());
    std::ostringstream summary;
    t.setLog(&summary);
    t.run();
    // Exactly one slot now holds generation 0, the network that played; the other is still random.
    const std::vector<LeagueMember>& league = t.league();
    assert(league.size() == 2);
    const int replaced = league[0].isRandom() ? 1 : 0;
    assert(league[replaced].generation == 0 && league[1 - replaced].isRandom());

    const std::vector<std::string> log = lines(readFile(dir + "/log.csv"));
    assert(log.size() == 2);
    assert(log[0] == kTodaysHeader + ",league_members,league_scores,league_replaced");
    const std::vector<std::string> row = splitCsv(log[1]);
    assert(row.size() == splitCsv(log[0]).size());
    const size_t n = row.size();
    assert(row[n - 3] == league[0].name() + ";" + league[1].name());
    assert(row[n - 1] == std::to_string(replaced));
    const std::vector<std::string> scores = splitOn(row[n - 2], ';');
    assert(scores.size() == 2);
    for (const std::string& s : scores) assert(!s.empty() && std::stod(s) >= 0 && std::stod(s) <= 1);
    assert(row[3] == "4");                    // games: all of them, league games included
    assert(row[n - 5] == "4");                // replay_games: 2 self-play + 2 league

    // The first sgf_per_generation league games are saved too.
    assert(fs::exists(dir + "/sgf/gen_0001_0.sgf") && fs::exists(dir + "/sgf/gen_0001_1.sgf"));
    assert(fs::exists(dir + "/sgf/gen_0001_league_0.sgf") && fs::exists(dir + "/sgf/gen_0001_league_1.sgf"));
    assert(!fs::exists(dir + "/sgf/gen_0001_league_2.sgf"));

    // The summary line names the members and their scores, and the replacement.
    const std::string line = summary.str();
    assert(contains(line, ", league random " + scores[0] + " | random " + scores[1]));
    assert(contains(line, ", slot " + std::to_string(replaced) + " -> gen_0"));

    // state.bin keeps the league.
    Trainer resumed = Trainer::resume(dir, {});
    assert(resumed.league().size() == 2 && resumed.league()[replaced].generation == 0 &&
           resumed.league()[1 - replaced].isRandom());
    std::printf("test_league_generation passed\n");
}

// log.csv without its two timing columns (seconds_selfplay, seconds_train).
static std::string logWithoutTimes(const std::string& dir) {
    std::string out;
    for (const std::string& line : lines(readFile(dir + "/log.csv"))) {
        std::vector<std::string> f = splitCsv(line);
        f[1] = f[2] = "";
        for (size_t i = 0; i < f.size(); ++i) out += (i ? "," : "") + f[i];
        out += "\n";
    }
    return out;
}

static void test_league_resume_matches_uninterrupted() {
    const std::string straight = freshDir("league_straight"), split = freshDir("league_split");
    RunConfig three = leagueConfig();
    three.generations = 3;
    Trainer::startNew(three, straight).run();

    RunConfig one = leagueConfig();
    one.generations = 1;
    Trainer::startNew(one, split).run();
    Trainer second = Trainer::resume(split, {"generations=3"});
    second.run();
    assert(second.generation() == 3);

    for (int g = 1; g <= 3; ++g) assert(readFile(checkpointPath(straight, g)) == readFile(checkpointPath(split, g)));
    assert(readFile(straight + "/state.bin") == readFile(split + "/state.bin"));
    assert(logWithoutTimes(straight) == logWithoutTimes(split));
    for (int g = 1; g <= 3; ++g)
        for (int i = 0; i < 2; ++i) {
            const std::string name = "/sgf/gen_000" + std::to_string(g) + "_league_" + std::to_string(i) + ".sgf";
            assert(readFile(straight + name) == readFile(split + name));
        }
    // Generations 2 and 3 played a network member (gen 0 was in the league
    // after generation 1), and each generation replaced one slot.
    const std::vector<std::string> log = lines(readFile(straight + "/log.csv"));
    assert(log.size() == 4);
    for (int g = 1; g <= 3; ++g) {
        const std::vector<std::string> row = splitCsv(log[g]);
        assert(row.back() == "0" || row.back() == "1");
    }
    assert(contains(splitCsv(log[1])[splitCsv(log[1]).size() - 3], "gen_0"));
    std::printf("test_league_resume_matches_uninterrupted passed\n");
}

// League fraction giving no league games: no league games, the columns
// still written (empty scores, nothing replaced), the league unchanged.
static void test_league_no_games() {
    const std::string dir = freshDir("league_none");
    RunConfig c = leagueConfig();
    c.generations = 1;
    c.leagueFraction = 0.1f;   // round(4 * 0.1) = 0
    Trainer t = Trainer::startNew(c, dir);
    t.run();
    assert(t.league()[0].isRandom() && t.league()[1].isRandom());
    const std::vector<std::string> log = lines(readFile(dir + "/log.csv"));
    const std::vector<std::string> row = splitCsv(log[1]);
    assert(row.size() == splitCsv(log[0]).size());
    const size_t n = row.size();
    assert(row[n - 3] == "random;random" && row[n - 2] == ";" && row[n - 1] == "-1");
    assert(row[n - 5] == "4");
    assert(!fs::exists(dir + "/sgf/gen_0001_league_0.sgf"));
    std::printf("test_league_no_games passed\n");
}

static void test_league_size_fixed_on_resume() {
    const std::string dir = freshDir("league_fixed");
    Trainer::startNew(leagueConfig(), dir);
    for (const char* o : {"league_size=1", "league_size=0"}) {
        try {
            Trainer::resume(dir, {o});
            assert(false && "expected std::invalid_argument");
        } catch (const std::invalid_argument& e) {
            assert(contains(e.what(), "league_size") && contains(e.what(), "cannot change on resume"));
        }
    }
    // Restating it is no change; the other league keys may change.
    Trainer ok = Trainer::resume(dir, {"league_size=2", "league_fraction=0.25", "league_replace_score=0.7",
                                       "league_min_games=3"});
    assert(ok.config().leagueFraction == 0.25f && ok.config().leagueMinGames == 3 && ok.league().size() == 2);
    std::printf("test_league_size_fixed_on_resume passed\n");
}

static void test_league_missing_checkpoint() {
    const std::string dir = freshDir("league_missing");
    RunConfig c = leagueConfig();
    c.generations = 1;
    Trainer::startNew(c, dir).run();   // one slot now holds gen 0
    const std::string path = checkpointPath(dir, 0);
    fs::remove(path);
    try {
        Trainer::resume(dir, {});
        assert(false && "expected std::runtime_error");
    } catch (const std::runtime_error& e) {
        assert(contains(e.what(), path));
    }
    std::printf("test_league_missing_checkpoint passed\n");
}

// state.bin version 1 (before the league) still loads, with an empty
// league; a league whose size is not league_size is refused.
static void test_league_state_versions() {
    const std::string dir = freshDir("league_v1");
    Trainer::startNew(tinyConfig(), dir);
    const std::string path = dir + "/state.bin";
    const std::string v2 = readFile(path);
    assert(v2.compare(0, 4, "MZRS") == 0 && v2[4] == 2);
    assert(v2.compare(v2.size() - 4, 4, std::string(4, '\0')) == 0);   // no members
    {
        std::string v1 = v2.substr(0, v2.size() - 4);
        v1[4] = 1;
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out << v1;
    }
    Trainer t = Trainer::resume(dir, {});
    assert(t.generation() == 0 && t.league().empty());

    // A run whose config says league_size 2 but whose state.bin holds no league.
    const std::string bad = freshDir("league_mismatch");
    Trainer::startNew(tinyConfig(), bad);
    {
        std::ofstream out(bad + "/config.cfg", std::ios::app);
        out << "league_size = 2\n";
    }
    try {
        Trainer::resume(bad, {});
        assert(false && "expected std::runtime_error");
    } catch (const std::runtime_error& e) {
        assert(contains(e.what(), bad + "/state.bin") && contains(e.what(), "league"));
    }
    std::printf("test_league_state_versions passed\n");
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
    test_league_generation();
    test_league_resume_matches_uninterrupted();
    test_league_no_games();
    test_league_size_fixed_on_resume();
    test_league_missing_checkpoint();
    test_league_state_versions();
    fs::remove_all(kRoot);
    return 0;
}
