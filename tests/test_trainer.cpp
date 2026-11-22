#include <algorithm>
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

// log.csv's header with K = 5 and no league: the columns of every run before
// the league, which a league-off run must still write exactly.
static const std::string kHeaderK5 =
    "generation,seconds_selfplay,seconds_train,games,mean_length,capped_games,mean_depth,max_depth,"
    "past_end_fraction,lr,loss_total,policy_k0,policy_k1,policy_k2,policy_k3,policy_k4,policy_k5,"
    "value_k0,value_k1,value_k2,value_k3,value_k4,value_k5,reward_k1,reward_k2,reward_k3,reward_k4,"
    "reward_k5,arena_prev,arena_rollout,arena_random,replay_games,replay_positions";

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
    const std::string& header = kHeaderK5;
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
    Trainer ok = Trainer::resume(dir, {"filters=8", "simulations=4", "no_pass_before=5"});
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
// member: 8 games per generation, 4 of them league games (games 0 and 1
// against slot 0, games 2 and 3 against slot 1).
static RunConfig leagueConfig() {
    RunConfig c = tinyConfig();
    c.gamesPerGeneration = 8;
    c.leagueSize = 2;
    c.leagueReplaceScore = 0;
    c.leagueMinGames = 0;
    return c;
}

// The league columns follow the others: n - 6 .. n - 1 of a row of n fields.
enum LeagueColumn { kMembers = 6, kScores = 5, kReplaced = 4, kLeagueMeanLength = 3, kLeagueCapped = 2,
                    kLeaguePositions = 1, kReplayGames = 8, kReplayPositions = 7 };

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
    c.generations = 2;
    Trainer t = Trainer::startNew(c, dir);
    assert(t.league().size() == 2 && t.league()[0].isRandom() && t.league()[1].isRandom());
    std::ostringstream summary;
    t.setLog(&summary);
    t.runGeneration();
    // Any score beats a member here, but a random member needs two wins in a
    // row: after generation 1 both are still random, each with a first strike.
    const std::vector<LeagueMember>& league = t.league();
    assert(league.size() == 2);
    assert(league[0].isRandom() && league[1].isRandom());
    assert(league[0].beatenLastGeneration && league[1].beatenLastGeneration);

    const std::vector<std::string> log = lines(readFile(dir + "/log.csv"));
    assert(log.size() == 2);
    assert(log[0] == kHeaderK5 + ",league_members,league_scores,league_replaced,league_mean_length,"
                                 "league_capped_games,league_positions");
    const std::vector<std::string> row = splitCsv(log[1]);
    assert(row.size() == splitCsv(log[0]).size());
    const size_t n = row.size();
    assert(row[n - kMembers] == "random;random");   // the members the scores were against
    assert(row[n - kReplaced] == "-1");
    const std::vector<std::string> scores = splitOn(row[n - kScores], ';');
    assert(scores.size() == 2);
    for (const std::string& s : scores) assert(!s.empty() && std::stod(s) >= 0 && std::stod(s) <= 1);
    assert(row[3] == "8");                         // games: all of them, league games included
    assert(row[n - kReplayGames] == "8");          // 4 self-play + 4 league
    // mean_length is the 4 self-play games'; the league games have their own
    // columns, and together they make up the window's positions.
    const double selfPlayPositions = 4 * std::stod(row[4]);
    const double leaguePositions = std::stod(row[n - kLeaguePositions]);
    assert(std::fabs(4 * std::stod(row[n - kLeagueMeanLength]) - leaguePositions) < 0.01);
    assert(std::fabs(selfPlayPositions + leaguePositions - std::stod(row[n - kReplayPositions])) < 0.01);
    const int leagueCapped = std::stoi(row[n - kLeagueCapped]);
    assert(leagueCapped >= 0 && leagueCapped <= 4);

    // The first sgf_per_generation league games are saved too.
    assert(fs::exists(dir + "/sgf/gen_0001_0.sgf") && fs::exists(dir + "/sgf/gen_0001_1.sgf"));
    assert(fs::exists(dir + "/sgf/gen_0001_league_0.sgf") && fs::exists(dir + "/sgf/gen_0001_league_1.sgf"));
    assert(!fs::exists(dir + "/sgf/gen_0001_league_2.sgf"));

    // The summary line names the members and their scores, '*' marking a
    // random member's first strike; nothing was replaced.
    const std::string line = summary.str();
    assert(contains(line, ", league (4 games, mean length " + row[n - kLeagueMeanLength] + ") random " + scores[0] +
                              "* | random " + scores[1] + "*"));
    assert(contains(line, "selfplay ") && contains(line, "s (4 games, mean length " + row[4] + ")"));
    assert(!contains(line, "-> gen_"));

    // state.bin keeps the league, strikes included.
    {
        Trainer resumed = Trainer::resume(dir, {});
        assert(resumed.league().size() == 2 && resumed.league()[0].isRandom() &&
               resumed.league()[0].beatenLastGeneration && resumed.league()[1].beatenLastGeneration);
    }

    // Generation 2: both are beaten again, so both are eligible; one (the
    // higher score, ties to slot 0) is replaced by gen 1, the network that
    // played, and (league_reset_on_replace, default true) the other's strike
    // is cleared too.
    summary.str("");
    t.runGeneration();
    const std::vector<std::string> row2 = splitCsv(lines(readFile(dir + "/log.csv"))[2]);
    const int replaced = std::stoi(row2[n - kReplaced]);
    assert(replaced == 0 || replaced == 1);
    assert(row2[n - kMembers] == "random;random");
    assert(league[replaced].generation == 1 && !league[replaced].beatenLastGeneration);
    assert(league[1 - replaced].isRandom() && !league[1 - replaced].beatenLastGeneration);
    const std::vector<std::string> scores2 = splitOn(row2[n - kScores], ';');
    assert(std::stod(scores2[replaced]) >= std::stod(scores2[1 - replaced]));
    assert(contains(summary.str(), ", slot " + std::to_string(replaced) + " -> gen_1"));
    assert(contains(summary.str(), "random " + scores2[1 - replaced] + " | ") ||
           contains(summary.str(), "random " + scores2[1 - replaced] + ", slot"));
    assert(!contains(summary.str(), "random " + scores2[1 - replaced] + "*"));
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
    three.leagueResetOnReplace = false;   // keep the other slot's strike, so generation 3 replaces again
    three.generations = 3;
    Trainer::startNew(three, straight).run();

    RunConfig one = leagueConfig();
    one.leagueResetOnReplace = false;
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
    // The split falls between the random members' first strike (generation
    // 1, nothing replaced) and their second (generation 2, one replaced by
    // gen 1); generation 3 replaces a slot again. Each row shows the members
    // the generation played, so a new member appears in the next row.
    const std::vector<std::string> log = lines(readFile(straight + "/log.csv"));
    assert(log.size() == 4);
    std::vector<std::vector<std::string>> rows;
    for (int g = 1; g <= 3; ++g) rows.push_back(splitCsv(log[g]));
    const size_t n = rows[0].size();
    assert(rows[0][n - kMembers] == "random;random" && rows[0][n - kReplaced] == "-1");
    for (int g = 2; g <= 3; ++g) {
        const std::string& replaced = rows[g - 1][n - kReplaced];
        assert(replaced == "0" || replaced == "1");
    }
    assert(rows[1][n - kMembers] == "random;random");
    std::vector<std::string> expected = splitOn(rows[1][n - kMembers], ';');
    expected[std::stoi(rows[1][n - kReplaced])] = "gen_1";
    assert(rows[2][n - kMembers] == expected[0] + ";" + expected[1]);
    // league_network_wins defaults to 2: generation 3 beats gen_1 for the
    // first time (a first strike), so the slot replaced is the random one
    // that was beaten twice, not the network member.
    assert(expected[std::stoi(rows[2][n - kReplaced])] == "random");
    std::printf("test_league_resume_matches_uninterrupted passed\n");
}

// League fraction giving no league games: no league games, the columns
// still written (empty scores, nothing replaced), the league unchanged.
static void test_league_no_games() {
    const std::string dir = freshDir("league_none");
    RunConfig c = leagueConfig();
    c.generations = 1;
    c.leagueFraction = 0.05f;   // round(8 * 0.05) = 0
    Trainer t = Trainer::startNew(c, dir);
    t.run();
    assert(t.league()[0].isRandom() && t.league()[1].isRandom());
    const std::vector<std::string> log = lines(readFile(dir + "/log.csv"));
    const std::vector<std::string> row = splitCsv(log[1]);
    assert(row.size() == splitCsv(log[0]).size());
    const size_t n = row.size();
    assert(row[n - kMembers] == "random;random" && row[n - kScores] == ";" && row[n - kReplaced] == "-1");
    assert(row[n - kLeagueMeanLength].empty() && row[n - kLeagueCapped] == "0" && row[n - kLeaguePositions] == "0");
    assert(row[n - kReplayGames] == "8");
    assert(!fs::exists(dir + "/sgf/gen_0001_league_0.sgf"));
    std::printf("test_league_no_games passed\n");
}

// league_value = false: the league games (added after the self-play games)
// train no value; self-play games do. It also survives state.bin, and may
// change on resume.
static void test_league_value_off() {
    const std::string dir = freshDir("league_value_off");
    RunConfig c = leagueConfig();
    c.leagueValue = false;
    Trainer t = Trainer::startNew(c, dir);
    t.runGeneration();
    assert(t.replay().games() == 8);
    for (int i = 0; i < 8; ++i) assert(t.replay().game(i).trainValue == (i < 4));
    Trainer back = Trainer::resume(dir, {});
    assert(back.replay().games() == 8);
    for (int i = 0; i < 8; ++i) assert(back.replay().game(i).trainValue == (i < 4));
    // Switching it on for the next generation only affects the new records.
    Trainer on = Trainer::resume(dir, {"league_value=true", "generations=2"});
    on.runGeneration();
    assert(on.replay().games() == 16);
    for (int i = 0; i < 4; ++i) assert(on.replay().game(i).trainValue);
    for (int i = 4; i < 8; ++i) assert(!on.replay().game(i).trainValue);
    for (int i = 8; i < 16; ++i) assert(on.replay().game(i).trainValue);
    // With the default, every record trains the value.
    const std::string dir2 = freshDir("league_value_on");
    Trainer d = Trainer::startNew(leagueConfig(), dir2);
    d.runGeneration();
    for (int i = 0; i < 8; ++i) assert(d.replay().game(i).trainValue);
    std::printf("test_league_value_off passed\n");
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
    Trainer ok = Trainer::resume(dir, {"league_size=2", "league_fraction=1", "league_replace_score=0.7",
                                       "league_min_games=3"});
    assert(ok.config().leagueFraction == 1.0f && ok.config().leagueMinGames == 3 && ok.league().size() == 2);
    std::printf("test_league_size_fixed_on_resume passed\n");
}

static void test_league_missing_checkpoint() {
    const std::string dir = freshDir("league_missing");
    RunConfig c = leagueConfig();
    c.generations = 2;
    Trainer t = Trainer::startNew(c, dir);
    t.run();   // two strikes: one slot now holds gen 1
    const int held = std::max(t.league()[0].generation, t.league()[1].generation);
    assert(held == 1);
    const std::string path = checkpointPath(dir, held);
    fs::remove(path);
    try {
        Trainer::resume(dir, {});
        assert(false && "expected std::runtime_error");
    } catch (const std::runtime_error& e) {
        assert(contains(e.what(), path));
    }
    std::printf("test_league_missing_checkpoint passed\n");
}

// state.bin version 1 (before the league), whose replay buffer is version 1
// too (no policy mask flags), still loads, with an empty league, and goes on
// exactly as the same run's version 2 state does. A league whose size is not
// league_size is refused.
static void test_league_state_versions() {
    const std::string dirV1 = freshDir("league_v1"), dirV2 = freshDir("league_v2");
    RunConfig c = tinyConfig();
    c.generations = 1;
    Trainer::startNew(c, dirV1).run();
    fs::copy(dirV1, dirV2, fs::copy_options::recursive);
    const std::string path = dirV1 + "/state.bin";
    const std::string v2 = readFile(path);
    assert(v2.compare(0, 4, "MZRS") == 0 && v2[4] == 2);
    assert(v2.compare(v2.size() - 4, 4, std::string(4, '\0')) == 0);   // no members
    // The replay buffer: the last "MZRB", checked by loading it to the league's start.
    const size_t replayAt = v2.rfind("MZRB");
    assert(replayAt != std::string::npos);
    std::istringstream replayIn(v2.substr(replayAt, v2.size() - 4 - replayAt), std::ios::binary);
    BinaryReader r(replayIn);
    const ReplayBuffer replay = ReplayBuffer::load(r);
    assert(replayIn.peek() == std::char_traits<char>::eof() && replay.games() == 4);
    // Version 1 of both: no league after the replay buffer, no mask flag per game.
    std::ostringstream v1(std::ios::binary);
    {
        BinaryWriter w(v1);
        std::string head = v2.substr(0, replayAt);
        head[4] = 1;
        w.bytes(head.data(), head.size());
        w.bytes("MZRB", 4);
        w.u32(1);
        w.bytes(v2.data() + replayAt + 8, 8);   // capacity, game count
        for (int i = 0; i < replay.games(); ++i) {
            const GameRecord& g = replay.game(i);
            assert(g.policyMask.empty());
            w.u32(static_cast<uint32_t>(g.boardSize));
            w.f32(g.komi);
            w.f32(g.resultForBlack);
            w.u32(static_cast<uint32_t>(g.length()));
            for (int m : g.moves) w.i32(m);
            w.bytes(g.stones.data(), g.stones.size());
            w.floats(g.policies.data(), g.policies.size());
            w.floats(g.rootValues.data(), g.rootValues.size());
        }
    }
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out << v1.str();
    }
    Trainer t = Trainer::resume(dirV1, {"generations=2"});
    assert(t.generation() == 1 && t.league().empty());
    t.run();
    Trainer::resume(dirV2, {"generations=2"}).run();
    assert(readFile(checkpointPath(dirV1, 2)) == readFile(checkpointPath(dirV2, 2)));
    assert(readFile(dirV1 + "/state.bin") == readFile(dirV2 + "/state.bin"));   // written as version 2 again

    // A run whose config says league_size 2 but whose state.bin holds no league.
    const std::string bad = freshDir("league_mismatch");
    Trainer::startNew(tinyConfig(), bad);
    {
        std::ofstream out(bad + "/config.cfg", std::ios::app);
        out << "league_size = 2\nleague_min_games = 0\n";
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
    test_league_value_off();
    test_league_size_fixed_on_resume();
    test_league_missing_checkpoint();
    test_league_state_versions();
    fs::remove_all(kRoot);
    return 0;
}
