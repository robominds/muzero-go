#include "mzgo/trainer.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include "mzgo/checkpoint.hpp"
#include "mzgo/game.hpp"
#include "mzgo/game_record.hpp"
#include "mzgo/league.hpp"
#include "mzgo/match.hpp"
#include "mzgo/players.hpp"
#include "mzgo/rng.hpp"
#include "mzgo/selfplay.hpp"
#include "mzgo/serialize.hpp"
#include "mzgo/sgf.hpp"

namespace mzgo {

namespace fs = std::filesystem;

namespace {

using Clock = std::chrono::steady_clock;

// state.bin: magic "MZRS", u32 version, i32 generation, u64 global step, the
// network (checkpoint.hpp's format), u32 velocity count, per velocity tensor
// u64 size + floats (params() order), the replay buffer (ReplayBuffer::save),
// then the league: u32 member count and per slot an i32 generation (-1 = the
// random bot) and a u8 beaten-last-generation flag. Nothing may follow.
// Version 1 has no league and loads with an empty one.
const char kStateMagic[4] = {'M', 'Z', 'R', 'S'};
constexpr uint32_t kStateVersion = 2;

// Games of random moves whose positions calibrate batch norm (bn_calibrate).
// A random 5x5 game runs to the move cap or close to it, so even the
// smallest board gives several hundred positions to draw one batch from.
constexpr int kCalibrationGames = 16;

double secondsSince(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string num(double v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.6g", v);
    return buf;
}

std::string statePath(const std::string& runDir) { return runDir + "/state.bin"; }
std::string configPath(const std::string& runDir) { return runDir + "/config.cfg"; }
std::string logPath(const std::string& runDir) { return runDir + "/log.csv"; }

std::string sgfPath(const std::string& runDir, int generation, int index) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "/sgf/gen_%04d_%d.sgf", generation, index);
    return runDir + buf;
}

std::string leagueSgfPath(const std::string& runDir, int generation, int index) {
    char buf[40];
    std::snprintf(buf, sizeof buf, "/sgf/gen_%04d_league_%d.sgf", generation, index);
    return runDir + buf;
}

// With a league (league_size > 0), six league columns follow the others.
std::string logHeader(int K, bool league) {
    std::string h =
        "generation,seconds_selfplay,seconds_train,games,mean_length,capped_games,mean_depth,max_depth,"
        "past_end_fraction,lr,loss_total";
    for (int k = 0; k <= K; ++k) h += ",policy_k" + std::to_string(k);
    for (int k = 0; k <= K; ++k) h += ",value_k" + std::to_string(k);
    for (int k = 1; k <= K; ++k) h += ",reward_k" + std::to_string(k);
    h += ",arena_prev,arena_rollout,arena_random,replay_games,replay_positions";
    if (league)
        h += ",league_members,league_scores,league_replaced,league_mean_length,league_capped_games,league_positions";
    return h;
}

void appendLine(const std::string& path, const std::string& line) {
    std::ofstream out(path, std::ios::app);
    out << line << '\n';
    out.flush();
    if (!out) throw std::runtime_error("cannot append to " + path);
}

// One key's value as config.cfg would print it ("" if the key is unknown).
std::string valueOf(const RunConfig& cfg, const std::string& key) {
    std::istringstream in(configToString(cfg));
    const std::string prefix = key + " = ";
    std::string line;
    while (std::getline(in, line))
        if (line.compare(0, prefix.size(), prefix) == 0) return line.substr(prefix.size());
    return "";
}

// The SGF of a self-play or league game; the boards are replayed for the score.
std::string sgfOf(const GameRecord& rec, int maxMoves, const std::string& comment) {
    Game game(rec.boardSize, rec.komi, maxMoves);
    for (int m : rec.moves) game.play(m);
    return gameToSgf(rec.boardSize, rec.komi, rec.moves, game.scoreForBlack(), comment);
}

template <typename T, typename F>
std::string joined(const std::vector<T>& items, const char* sep, F format) {
    std::string out;
    for (size_t i = 0; i < items.size(); ++i) out += (i ? sep : "") + format(items[i]);
    return out;
}

struct ArenaScores {
    bool ran = false;
    double prev = 0, rollout = 0, random = 0;   // the current network's score against each
};

} // namespace

NetConfig netConfigFrom(const RunConfig& c) {
    NetConfig n;
    n.boardSize = c.boardSize;
    n.history = c.history;
    n.reprBlocks = c.reprBlocks;
    n.dynBlocks = c.dynBlocks;
    n.filters = c.filters;
    n.valueHidden = c.valueHidden;
    n.rewardHead = c.rewardHead;
    n.passPlane = c.passPlane;
    n.unrollSteps = c.unrollSteps;
    return n;
}

std::string checkpointPath(const std::string& runDir, int generation) {
    char buf[40];
    std::snprintf(buf, sizeof buf, "/checkpoints/gen_%04d.mzn", generation);
    return runDir + buf;
}

Trainer::Trainer(const RunConfig& cfg, std::string runDir, MuZeroNetwork net)
    : cfg_(cfg),
      runDir_(std::move(runDir)),
      backend_(makeBackend(cfg.backend)),
      pool_(std::make_unique<ThreadPool>(resolveTrainThreads(cfg.trainThreads))),
      net_(std::make_unique<MuZeroNetwork>(std::move(net))),
      workspace_(std::make_unique<TrainWorkspace>(net_->config(), cfg.batchSize, cfg.unrollSteps, *backend_,
                                                  pool_.get())),
      optimizer_(net_->params(), cfg.momentum, cfg.weightDecay),
      replay_(cfg.replayGames) {}

Trainer Trainer::startNew(const RunConfig& cfg, const std::string& runDir) {
    validate(cfg);
    if (fs::exists(runDir) && !fs::is_empty(runDir))
        throw std::runtime_error("run directory " + runDir +
                                 " is not empty: resume that run or choose another directory");
    Trainer t(cfg, runDir, MuZeroNetwork(netConfigFrom(cfg), deriveSeed(cfg.seed, 0xA11)));
    if (cfg.bnCalibrate) t.calibrateBatchNorm();
    t.league_.assign(static_cast<size_t>(cfg.leagueSize), LeagueMember{});   // every slot the random bot

    fs::create_directories(runDir + "/checkpoints");
    fs::create_directories(runDir + "/sgf");
    writeFileAtomically(configPath(runDir), configToString(cfg));
    saveNetwork(*t.net_, checkpointPath(runDir, 0));
    t.writeState();
    writeFileAtomically(logPath(runDir), logHeader(cfg.unrollSteps, cfg.leagueSize > 0) + "\n");
    return t;
}

Trainer Trainer::resume(const std::string& runDir, const std::vector<std::string>& overrides) {
    const std::string path = statePath(runDir);
    if (!fs::exists(path)) throw std::runtime_error(runDir + " has no state.bin to resume from");

    RunConfig cfg = parseConfigFile(configPath(runDir));
    const RunConfig saved = cfg;
    for (const std::string& o : overrides) applyOverride(cfg, o);
    for (const std::string& key : architectureKeys()) {
        if (valueOf(cfg, key) != valueOf(saved, key))
            throw std::invalid_argument(key + ": an architecture key cannot change on resume (the run has " +
                                        valueOf(saved, key) + ", the override says " + valueOf(cfg, key) + ")");
    }
    // Not an architecture key (it fixes no tensor shape), but state.bin and
    // log.csv's columns depend on it.
    if (cfg.leagueSize != saved.leagueSize)
        throw std::invalid_argument("league_size: cannot change on resume (the run has " +
                                    std::to_string(saved.leagueSize) + ", the override says " +
                                    std::to_string(cfg.leagueSize) + ")");
    validate(cfg);

    // Everything is read before the Trainer is built, so any error below is
    // about the file.
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    BinaryReader r(in);
    int generation = 0;
    uint64_t step = 0;
    std::unique_ptr<MuZeroNetwork> net;
    std::vector<std::vector<float>> velocity;
    std::unique_ptr<ReplayBuffer> replay;
    std::vector<LeagueMember> league;
    try {
        char magic[4];
        r.bytes(magic, 4);
        if (!std::equal(magic, magic + 4, kStateMagic)) throw std::runtime_error("not an MZRS run state");
        const uint32_t version = r.u32();
        if (version < 1 || version > kStateVersion)
            throw std::runtime_error("unsupported version " + std::to_string(version));
        generation = r.i32();
        step = r.u64();
        if (generation < 0) throw std::runtime_error("negative generation");
        net = std::make_unique<MuZeroNetwork>(readNetwork(r, "network"));
        if (net->config() != netConfigFrom(cfg))
            throw std::runtime_error("the network's shape does not match " + configPath(runDir));
        const std::vector<ParamView> params = net->params();
        const uint32_t count = r.u32();
        if (count != params.size())
            throw std::runtime_error("has " + std::to_string(count) + " velocity tensors but the network has " +
                                     std::to_string(params.size()) + " parameters");
        for (const ParamView& p : params) {
            const uint64_t size = r.u64();
            if (size != p.size)
                throw std::runtime_error("velocity of '" + p.name + "' has " + std::to_string(size) +
                                         " values but expected " + std::to_string(p.size));
            velocity.emplace_back(p.size);
            r.floats(velocity.back().data(), p.size);
        }
        replay = std::make_unique<ReplayBuffer>(ReplayBuffer::load(r));
        if (replay->games() > 0 && replay->game(0).boardSize != cfg.boardSize)
            throw std::runtime_error("the replay buffer holds games of board size " +
                                     std::to_string(replay->game(0).boardSize) + " but board_size is " +
                                     std::to_string(cfg.boardSize));
        if (version >= 2) {
            const uint32_t members = r.u32();
            if (members > 8) throw std::runtime_error("has a league of " + std::to_string(members) + " members");
            for (uint32_t i = 0; i < members; ++i) {
                const int m = r.i32();
                if (m < -1 || m >= generation)
                    throw std::runtime_error("league slot " + std::to_string(i) + " holds generation " +
                                             std::to_string(m) + ", not one before " + std::to_string(generation));
                const uint8_t beaten = r.u8();
                if (beaten > 1)
                    throw std::runtime_error("league slot " + std::to_string(i) + " has a bad beaten flag");
                league.push_back(LeagueMember{m, beaten == 1});
            }
        }
        if (static_cast<int>(league.size()) != cfg.leagueSize)
            throw std::runtime_error("holds a league of " + std::to_string(league.size()) +
                                     " members but league_size is " + std::to_string(cfg.leagueSize));
        if (in.peek() != std::char_traits<char>::eof()) throw std::runtime_error("unexpected data at the end");
    } catch (const std::runtime_error& e) {
        throw std::runtime_error(path + ": " + e.what());
    }

    // A network member plays from its checkpoint: a missing one would fail
    // only once its games start, so refuse it now.
    for (size_t i = 0; i < league.size(); ++i) {
        if (league[i].isRandom()) continue;
        const std::string member = checkpointPath(runDir, league[i].generation);
        if (!fs::exists(member))
            throw std::runtime_error("league slot " + std::to_string(i) + " is " + league[i].name() + " but " +
                                     member + " is missing");
    }

    Trainer t(cfg, runDir, std::move(*net));
    t.generation_ = generation;
    t.league_ = std::move(league);
    t.globalStep_ = static_cast<long>(step);
    t.optimizer_.velocity() = std::move(velocity);
    // Re-adding the games in order rebuilds the same window, and applies a
    // replay_games override (the oldest games beyond it are dropped).
    for (int i = 0; i < replay->games(); ++i) t.replay_.add(replay->game(i));
    writeFileAtomically(configPath(runDir), configToString(cfg));
    return t;
}

// bn_calibrate (opt-in, default off): a freshly initialized network's batch
// norm running statistics are mean 0, variance 1, far from what its layers
// actually see, so search with generation 0 runs on badly scaled
// activations. Calibration sets them from real inputs: positions of games of
// uniformly random legal moves, one batch of batch_size samples.
//
// A Calibrate pass SETS, at every unroll step k, each layer's statistics of
// call index k to those of its input in that pass (network.hpp keeps one
// set per step), so one K-step pass calibrates every set: h and f on the
// real positions' s0, g, the reward head and f on s1..sK.
void Trainer::calibrateBatchNorm() {
    const NetConfig netCfg = net_->config();
    const int n = cfg_.boardSize, maxMoves = cfg_.resolvedMaxMoves(), A = netCfg.actions();
    const PlayerFactory random = [] { return std::make_unique<RandomPlayer>(); };
    const MatchResult games = playMatch(random, random, kCalibrationGames, n, cfg_.komi, maxMoves,
                                        deriveSeed(cfg_.seed, 0xCA1), resolveWorkers(cfg_.workers));
    ReplayBuffer buffer(kCalibrationGames);
    for (const GameRecordLite& lite : games.games) {
        // The targets are never used (calibration computes no loss); the
        // move played stands in for the search policy.
        Game game(n, cfg_.komi, maxMoves);
        std::vector<float> policies(lite.moves.size() * static_cast<size_t>(A), 0.0f);
        const std::vector<float> rootValues(lite.moves.size(), 0.0f);
        for (size_t t = 0; t < lite.moves.size(); ++t) {
            policies[t * A + lite.moves[t]] = 1.0f;
            game.play(lite.moves[t]);
        }
        buffer.add(recordFromGame(game, policies, rootValues));
    }
    TrainBatch batch;
    Rng rng(deriveSeed(cfg_.seed, 0xCA1, 1));
    buffer.sampleBatch(cfg_.batchSize, cfg_.unrollSteps, cfg_.history, cfg_.passPlane, false, rng, batch);
    net_->calibrateBatchNorm(*workspace_, batch);
}

void Trainer::writeState() const {
    std::ostringstream buffer(std::ios::binary);
    BinaryWriter w(buffer);
    w.bytes(kStateMagic, 4);
    w.u32(kStateVersion);
    w.i32(generation_);
    w.u64(static_cast<uint64_t>(globalStep_));
    writeNetwork(w, *net_);
    const auto& velocity = optimizer_.velocity();
    w.u32(static_cast<uint32_t>(velocity.size()));
    for (const std::vector<float>& v : velocity) {
        w.u64(v.size());
        w.floats(v.data(), v.size());
    }
    replay_.save(w);
    w.u32(static_cast<uint32_t>(league_.size()));
    for (const LeagueMember& m : league_) {
        w.i32(m.generation);
        w.u8(m.beatenLastGeneration ? 1 : 0);
    }
    writeFileAtomically(statePath(runDir_), buffer.str());
}

void Trainer::runGeneration() {
    const int g = generation_ + 1;
    const int n = cfg_.boardSize, K = cfg_.unrollSteps, maxMoves = cfg_.resolvedMaxMoves();
    const int workers = resolveWorkers(cfg_.workers);

    // 1. Self-play with the current network (only read while it plays).
    SelfPlayConfig sp;
    sp.mcts.simulations = cfg_.simulations;
    sp.mcts.addNoise = true;
    sp.mcts.dirichletAlpha = cfg_.dirichletScale / static_cast<float>(n * n);
    sp.mcts.noiseFraction = cfg_.noiseFraction;
    sp.boardSize = n;
    sp.maxMoves = maxMoves;
    sp.temperatureMoves = cfg_.temperatureMoves;
    sp.komi = cfg_.komi;
    sp.group = cfg_.selfplayGroup;
    // With a league, the last L games are league games: games 2k and 2k + 1
    // play the member in slot k mod league_size, the learner Black in the
    // first, White in the second; each network member is loaded once.
    const bool hasLeague = cfg_.leagueSize > 0;
    assert(static_cast<int>(league_.size()) == cfg_.leagueSize);
    const int leagueGames = leagueGameCount(cfg_.gamesPerGeneration, cfg_.leagueFraction, cfg_.leagueSize);
    const int selfPlayGames = cfg_.gamesPerGeneration - leagueGames;
    const std::vector<int> opponentOf = leagueOpponents(leagueGames, cfg_.leagueSize);
    // log.csv's self-play columns (mean length, capped games, search
    // statistics) stay the self-play games' alone, so a league run reads like
    // any other; the league games get columns of their own. seconds covers both.
    SelfPlayStats stats{}, leagueStats{};
    std::vector<GameRecord> records, leagueRecords;
    {
        const NetworkModel model(*net_, *backend_);
        if (selfPlayGames > 0)
            records = playSelfPlay(model, sp, selfPlayGames, deriveSeed(cfg_.seed, g, 1), workers, &stats);
        if (leagueGames > 0) {
            std::map<int, std::unique_ptr<MuZeroNetwork>> nets;
            std::map<int, std::unique_ptr<NetworkModel>> models;
            std::vector<LeagueOpponent> opponents;
            for (const LeagueMember& m : league_) {
                if (m.isRandom()) {
                    opponents.push_back(LeagueOpponent{});
                    continue;
                }
                if (!models.count(m.generation)) {
                    const std::string path = checkpointPath(runDir_, m.generation);
                    auto net = std::make_unique<MuZeroNetwork>(loadNetwork(path));
                    if (net->config() != net_->config())
                        throw std::runtime_error(path + ": league member's network shape does not match the run's");
                    models[m.generation] = std::make_unique<NetworkModel>(*net, *backend_);
                    nets[m.generation] = std::move(net);
                }
                opponents.push_back(LeagueOpponent{models[m.generation].get()});
            }
            leagueRecords = playLeagueGames(model, opponents, opponentOf, sp, deriveSeed(cfg_.seed, g, 4), workers,
                                            &leagueStats);
        }
        stats.seconds += leagueStats.seconds;
    }
    std::vector<std::string> sgfs, leagueSgfs;
    for (int i = 0; i < std::min(cfg_.sgfPerGeneration, static_cast<int>(records.size())); ++i)
        sgfs.push_back(sgfOf(records[i], maxMoves,
                             "mzgo self-play, generation " + std::to_string(g) + ", game " + std::to_string(i)));
    for (int i = 0; i < std::min(cfg_.sgfPerGeneration, leagueGames); ++i)
        leagueSgfs.push_back(sgfOf(leagueRecords[i], maxMoves,
                                   "mzgo league, generation " + std::to_string(g) + ", game " + std::to_string(i) +
                                       ": gen_" + std::to_string(g - 1) + " (training) as " +
                                       (i % 2 == 0 ? "Black" : "White") + " vs " +
                                       league_[opponentOf[i]].name()));
    const std::vector<LeagueResult> results = leagueResults(leagueRecords, opponentOf, cfg_.leagueSize);
    long leaguePositions = 0;
    for (const GameRecord& rec : leagueRecords) leaguePositions += rec.length();

    // 2. Into the replay window: the self-play games, then the league games.
    for (GameRecord& rec : records) replay_.add(std::move(rec));
    for (GameRecord& rec : leagueRecords) {
        if (!cfg_.leagueValue) rec.trainValue = false;
        replay_.add(std::move(rec));
    }

    // 3. Training. Step s's batch comes from its own seed, so neither the
    // number of threads nor a resume changes which positions it sees.
    const auto trainStart = Clock::now();
    const std::vector<ParamView> params = net_->params();
    const float firstLr = exponentialLr(cfg_.lrInit, cfg_.lrDecayRate, cfg_.lrDecaySteps, globalStep_);
    UnrollLoss sum;
    sum.policy.assign(K + 1, 0.0);
    sum.value.assign(K + 1, 0.0);
    sum.reward.assign(K + 1, 0.0);
    TrainBatch batch;
    for (int s = 1; s <= cfg_.trainSteps; ++s) {
        Rng rng(deriveSeed(cfg_.seed, static_cast<uint64_t>(g), 2, static_cast<uint64_t>(s)));
        replay_.sampleBatch(cfg_.batchSize, K, cfg_.history, cfg_.passPlane, cfg_.symmetryAugment, rng, batch);
        const float lr = exponentialLr(cfg_.lrInit, cfg_.lrDecayRate, cfg_.lrDecaySteps, globalStep_);
        const UnrollLoss loss = net_->trainStep(*workspace_, batch);
        if (!std::isfinite(loss.total))
            throw std::runtime_error("non-finite loss at generation " + std::to_string(g) + " step " +
                                     std::to_string(s));
        optimizer_.step(params, lr, pool_.get());
        ++globalStep_;
        sum.total += loss.total;
        for (int k = 0; k <= K; ++k) {
            sum.policy[k] += loss.policy[k];
            sum.value[k] += loss.value[k];
            sum.reward[k] += loss.reward[k];
        }
    }
    const double secondsTrain = secondsSince(trainStart);
    // Each loss above was computed before its step's update, so the last
    // update is still unchecked: a blow-up there must not reach the
    // checkpoint or state.bin, from which there would be no way back.
    if (cfg_.trainSteps > 0) {
        bool finite = true;
        for (const ParamView& p : params)
            finite = finite && std::all_of(p.value, p.value + p.size, [](float v) { return std::isfinite(v); });
        for (const BufferView& b : net_->buffers())
            finite = finite && std::all_of(b.value, b.value + b.size, [](float v) { return std::isfinite(v); });
        if (!finite)
            throw std::runtime_error("non-finite weights at generation " + std::to_string(g) + " step " +
                                     std::to_string(cfg_.trainSteps));
    }

    // The league: the network that played (generation g - 1's) replaces the
    // most clearly beaten eligible member (a random one needs a second win in
    // a row), at most one per generation; every slot's strike is updated. A
    // generation without league games beats nobody, so it clears the strikes.
    const std::vector<LeagueMember> leagueBefore = league_;
    const int replacedSlot =
        hasLeague ? advanceLeague(league_, results, cfg_.leagueReplaceScore, cfg_.leagueMinGames, g - 1) : -1;

    // 4. Checkpoint, then state.bin: from here on the generation is done
    // and a resume continues with g + 1. A crash after this point (in the
    // arena, say) loses only the generation's log row and summary line.
    saveNetwork(*net_, checkpointPath(runDir_, g));
    generation_ = g;
    writeState();
    for (size_t i = 0; i < sgfs.size(); ++i) writeFileAtomically(sgfPath(runDir_, g, static_cast<int>(i)), sgfs[i]);
    for (size_t i = 0; i < leagueSgfs.size(); ++i)
        writeFileAtomically(leagueSgfPath(runDir_, g, static_cast<int>(i)), leagueSgfs[i]);

    // 5. Arena: no noise, the first arena_temperature_moves moves sampled.
    ArenaScores arena;
    if (cfg_.arenaEvery > 0 && cfg_.arenaGames > 0 && g % cfg_.arenaEvery == 0) {
        MctsConfig mcts;
        mcts.simulations = cfg_.arenaSimulations;
        mcts.addNoise = false;
        const int temperatureMoves = cfg_.arenaTemperatureMoves;
        const int prevGen = std::max(0, g - cfg_.arenaGap);
        const MuZeroNetwork prevNet = loadNetwork(checkpointPath(runDir_, prevGen));
        const NetworkModel current(*net_, *backend_), prev(prevNet, *backend_);
        const std::string name = "gen_" + std::to_string(g), prevName = "gen_" + std::to_string(prevGen);
        const PlayerFactory a = [&] { return std::make_unique<MuZeroPlayer>(current, mcts, temperatureMoves, name); };
        const PlayerFactory vsPrev = [&] {
            return std::make_unique<MuZeroPlayer>(prev, mcts, temperatureMoves, prevName);
        };
        const int rolloutSims = cfg_.rolloutBotSimulations;
        const PlayerFactory vsRollout = [rolloutSims] { return std::make_unique<RolloutMctsPlayer>(rolloutSims); };
        const PlayerFactory vsRandom = [] { return std::make_unique<RandomPlayer>(); };
        const uint64_t seed = deriveSeed(cfg_.seed, static_cast<uint64_t>(g), 3);
        auto score = [&](const PlayerFactory& b) {
            return playMatch(a, b, cfg_.arenaGames, n, cfg_.komi, maxMoves, seed, workers).scoreA();
        };
        arena.ran = true;
        arena.prev = score(vsPrev);
        arena.rollout = score(vsRollout);
        arena.random = score(vsRandom);
    }

    // 6. Log row and summary line.
    const int steps = cfg_.trainSteps;
    auto mean = [steps](double total) { return steps > 0 ? num(total / steps) : std::string(); };
    std::string row = std::to_string(g) + "," + num(stats.seconds) + "," + num(secondsTrain) + "," +
                      std::to_string(cfg_.gamesPerGeneration) + "," + num(stats.meanLength) + "," +
                      std::to_string(stats.cappedGames) + "," + num(stats.meanDepth) + "," +
                      std::to_string(stats.maxDepth) + "," + num(stats.pastEndFraction) + "," + num(firstLr) + "," +
                      mean(sum.total);
    for (int k = 0; k <= K; ++k) row += "," + mean(sum.policy[k]);
    for (int k = 0; k <= K; ++k) row += "," + mean(sum.value[k]);
    for (int k = 1; k <= K; ++k) row += "," + mean(sum.reward[k]);
    for (double s : {arena.prev, arena.rollout, arena.random}) row += "," + (arena.ran ? num(s) : std::string());
    row += "," + std::to_string(replay_.games()) + "," + std::to_string(replay_.positions());
    if (hasLeague) {
        // The members as they were during the generation, each next to its
        // score; the new member shows in the next row.
        row += "," + joined(leagueBefore, ";", [](const LeagueMember& m) { return m.name(); });
        row += "," + joined(results, ";", [](const LeagueResult& r) { return r.games > 0 ? num(r.score()) : ""; });
        row += "," + std::to_string(replacedSlot);
        row += "," + (leagueGames > 0 ? num(leagueStats.meanLength) : std::string());
        row += "," + std::to_string(leagueStats.cappedGames) + "," + std::to_string(leaguePositions);
    }
    appendLine(logPath(runDir_), row);

    if (log_) {
        *log_ << "gen " << g << ": selfplay " << num(stats.seconds) << "s (" << selfPlayGames
              << " games, mean length " << num(stats.meanLength) << "), train " << num(secondsTrain) << "s ("
              << steps << " steps, loss " << (steps > 0 ? mean(sum.total) : "-") << ", lr " << num(firstLr) << ")";
        if (arena.ran)
            *log_ << ", arena vs gen " << std::max(0, g - cfg_.arenaGap) << " " << num(arena.prev) << ", rollout "
                  << num(arena.rollout) << ", random " << num(arena.random);
        if (hasLeague) {
            *log_ << ", league (" << leagueGames << " games";
            if (leagueGames > 0) *log_ << ", mean length " << num(leagueStats.meanLength);
            *log_ << ") ";
            for (size_t i = 0; i < leagueBefore.size(); ++i)
                // '*': a random member's strike (beaten, not replaced); one
                // more win in a row replaces it.
                *log_ << (i ? " | " : "") << leagueBefore[i].name() << " "
                      << (results[i].games > 0 ? num(results[i].score()) : "-")
                      << (league_[i].isRandom() && league_[i].beatenLastGeneration ? "*" : "");
            if (replacedSlot >= 0)
                *log_ << ", slot " << replacedSlot << " -> " << league_[static_cast<size_t>(replacedSlot)].name();
        }
        *log_ << std::endl;
    }
}

void Trainer::run() {
    while (generation_ < cfg_.generations) runGeneration();
}

} // namespace mzgo
