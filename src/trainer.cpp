#include "mzgo/trainer.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include "mzgo/checkpoint.hpp"
#include "mzgo/game.hpp"
#include "mzgo/game_record.hpp"
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
// u64 size + floats (params() order), then the replay buffer
// (ReplayBuffer::save). Nothing may follow.
const char kStateMagic[4] = {'M', 'Z', 'R', 'S'};
constexpr uint32_t kStateVersion = 1;

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

std::string logHeader(int K) {
    std::string h =
        "generation,seconds_selfplay,seconds_train,games,mean_length,capped_games,mean_depth,max_depth,"
        "past_end_fraction,lr,loss_total";
    for (int k = 0; k <= K; ++k) h += ",policy_k" + std::to_string(k);
    for (int k = 0; k <= K; ++k) h += ",value_k" + std::to_string(k);
    for (int k = 1; k <= K; ++k) h += ",reward_k" + std::to_string(k);
    return h + ",arena_prev,arena_rollout,arena_random,replay_games,replay_positions";
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

// The first K steps of an unroll of b.K >= K steps, sample by sample.
TrainBatch truncateUnroll(const TrainBatch& b, int K, const NetConfig& cfg) {
    const int A = cfg.actions();
    TrainBatch t;
    t.resize(b.B, K, cfg.planes(), cfg.area(), A);
    t.observations = b.observations;
    for (int s = 0; s < b.B; ++s) {
        std::copy_n(b.actions.begin() + s * b.K, K, t.actions.begin() + s * K);
        std::copy_n(b.policyTargets.begin() + s * (b.K + 1) * A, (K + 1) * A, t.policyTargets.begin() + s * (K + 1) * A);
        std::copy_n(b.policyMask.begin() + s * (b.K + 1), K + 1, t.policyMask.begin() + s * (K + 1));
        std::copy_n(b.valueTargets.begin() + s * (b.K + 1), K + 1, t.valueTargets.begin() + s * (K + 1));
        std::copy_n(b.rewardTargets.begin() + s * (b.K + 1), K + 1, t.rewardTargets.begin() + s * (K + 1));
    }
    return t;
}

// The SGF of a self-play game; the boards are replayed for the score.
std::string sgfOf(const GameRecord& rec, int maxMoves, int generation, int index) {
    Game game(rec.boardSize, rec.komi, maxMoves);
    for (int m : rec.moves) game.play(m);
    return gameToSgf(rec.boardSize, rec.komi, rec.moves, game.scoreForBlack(),
                     "mzgo self-play, generation " + std::to_string(generation) + ", game " +
                         std::to_string(index));
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

    fs::create_directories(runDir + "/checkpoints");
    fs::create_directories(runDir + "/sgf");
    writeFileAtomically(configPath(runDir), configToString(cfg));
    saveNetwork(*t.net_, checkpointPath(runDir, 0));
    t.writeState();
    writeFileAtomically(logPath(runDir), logHeader(cfg.unrollSteps) + "\n");
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
    // Not an architecture key, but log.csv's header (written once, by
    // startNew) has one column per unroll step, so every row must keep K.
    if (cfg.unrollSteps != saved.unrollSteps)
        throw std::invalid_argument("unroll_steps: cannot change on resume (log.csv has columns for " +
                                    std::to_string(saved.unrollSteps) + " unroll steps)");
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
    try {
        char magic[4];
        r.bytes(magic, 4);
        if (!std::equal(magic, magic + 4, kStateMagic)) throw std::runtime_error("not an MZRS run state");
        const uint32_t version = r.u32();
        if (version != kStateVersion) throw std::runtime_error("unsupported version " + std::to_string(version));
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
        if (in.peek() != std::char_traits<char>::eof()) throw std::runtime_error("unexpected data at the end");
    } catch (const std::runtime_error& e) {
        throw std::runtime_error(path + ": " + e.what());
    }

    Trainer t(cfg, runDir, std::move(*net));
    t.generation_ = generation;
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
// A Calibrate pass SETS each layer's statistics to those of its input in
// that pass, and a layer reused in the unroll (g K times, f K + 1 times)
// keeps only its last call's statistics. One K-step pass would leave g with
// step K's statistics and f with s^K's. So two passes: first K = 1, which
// sets g (and the reward head) from its first call, on s0; then K = 0,
// which sets h and f from the real position's s0 and leaves g as it is.
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
    buffer.sampleBatch(cfg_.batchSize, cfg_.unrollSteps, cfg_.history, false, rng, batch);
    for (int K : {1, 0}) {
        TrainWorkspace ws(netCfg, cfg_.batchSize, K, *backend_, pool_.get());
        net_->calibrateBatchNorm(ws, truncateUnroll(batch, K, netCfg));
    }
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
    SelfPlayStats stats{};
    std::vector<GameRecord> records;
    {
        const NetworkModel model(*net_, *backend_);
        records = playSelfPlay(model, sp, cfg_.gamesPerGeneration, deriveSeed(cfg_.seed, g, 1), workers, &stats);
    }
    std::vector<std::string> sgfs;
    for (int i = 0; i < std::min(cfg_.sgfPerGeneration, static_cast<int>(records.size())); ++i)
        sgfs.push_back(sgfOf(records[i], maxMoves, g, i));

    // 2. Into the replay window.
    for (GameRecord& rec : records) replay_.add(std::move(rec));

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
        replay_.sampleBatch(cfg_.batchSize, K, cfg_.history, cfg_.symmetryAugment, rng, batch);
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

    // 4. Checkpoint, then state.bin: from here on the generation is done
    // and a resume continues with g + 1. A crash after this point (in the
    // arena, say) loses only the generation's log row and summary line.
    saveNetwork(*net_, checkpointPath(runDir_, g));
    generation_ = g;
    writeState();
    for (size_t i = 0; i < sgfs.size(); ++i) writeFileAtomically(sgfPath(runDir_, g, static_cast<int>(i)), sgfs[i]);

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
    appendLine(logPath(runDir_), row);

    if (log_) {
        *log_ << "gen " << g << ": selfplay " << num(stats.seconds) << "s (" << cfg_.gamesPerGeneration
              << " games, mean length " << num(stats.meanLength) << "), train " << num(secondsTrain) << "s ("
              << steps << " steps, loss " << (steps > 0 ? mean(sum.total) : "-") << ", lr " << num(firstLr) << ")";
        if (arena.ran)
            *log_ << ", arena vs gen " << std::max(0, g - cfg_.arenaGap) << " " << num(arena.prev) << ", rollout "
                  << num(arena.rollout) << ", random " << num(arena.random);
        *log_ << std::endl;
    }
}

void Trainer::run() {
    while (generation_ < cfg_.generations) runGeneration();
}

} // namespace mzgo
