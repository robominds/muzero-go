#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <string>
#include "mzgo/config.hpp"

using namespace mzgo;

static bool sameConfig(const RunConfig& a, const RunConfig& b) {
    return a.seed == b.seed && a.generations == b.generations && a.workers == b.workers &&
           a.trainThreads == b.trainThreads && a.backend == b.backend && a.boardSize == b.boardSize &&
           a.komi == b.komi && a.maxMoves == b.maxMoves && a.history == b.history &&
           a.reprBlocks == b.reprBlocks && a.dynBlocks == b.dynBlocks && a.filters == b.filters &&
           a.valueHidden == b.valueHidden && a.rewardHead == b.rewardHead &&
           a.gamesPerGeneration == b.gamesPerGeneration && a.simulations == b.simulations &&
           a.selfplayGroup == b.selfplayGroup && a.dirichletScale == b.dirichletScale &&
           a.noiseFraction == b.noiseFraction && a.temperatureMoves == b.temperatureMoves &&
           a.replayGames == b.replayGames && a.trainSteps == b.trainSteps && a.batchSize == b.batchSize &&
           a.unrollSteps == b.unrollSteps && a.lrInit == b.lrInit && a.lrDecayRate == b.lrDecayRate &&
           a.lrDecaySteps == b.lrDecaySteps && a.momentum == b.momentum && a.weightDecay == b.weightDecay &&
           a.symmetryAugment == b.symmetryAugment && a.bnCalibrate == b.bnCalibrate &&
           a.arenaEvery == b.arenaEvery && a.arenaGames == b.arenaGames && a.arenaGap == b.arenaGap &&
           a.arenaSimulations == b.arenaSimulations && a.arenaTemperatureMoves == b.arenaTemperatureMoves &&
           a.rolloutBotSimulations == b.rolloutBotSimulations && a.sgfPerGeneration == b.sgfPerGeneration;
}

static void test_defaults_round_trip() {
    RunConfig d;
    assert(sameConfig(parseConfigString(configToString(d)), d));
    RunConfig c;  // non-default values, including floats that are not short decimals
    c.seed = 18446744073709551615ULL; c.komi = 7.5f; c.lrInit = 0.0123457f; c.weightDecay = 1e-5f;
    c.dirichletScale = 0.3f; c.backend = "naive"; c.rewardHead = true; c.symmetryAugment = true;
    c.maxMoves = 77; c.boardSize = 9;
    assert(sameConfig(parseConfigString(configToString(c)), c));
    assert(d.resolvedMaxMoves() == 50 && c.resolvedMaxMoves() == 77);
    std::printf("test_defaults_round_trip passed\n");
}

static void test_presets_parse() {
    const std::string dir = std::string(MZGO_SOURCE_DIR) + "/configs/";
    RunConfig c5 = parseConfigFile(dir + "5x5.cfg");
    validate(c5);
    // The presets turn the reward head on (docs/results.md); the struct
    // default stays off so a config without the key keeps its old meaning.
    assert(!RunConfig().rewardHead);
    RunConfig expected5;
    expected5.rewardHead = true;
    assert(sameConfig(c5, expected5));
    RunConfig c9 = parseConfigFile(dir + "9x9.cfg");
    validate(c9);
    assert(c9.rewardHead);
    assert(c9.boardSize == 9 && c9.filters == 64 && c9.dynBlocks == 6 && c9.reprBlocks == 6);
    assert(c9.komi == 7.5f && c9.batchSize == 256 && c9.lrDecaySteps == 60000 && c9.arenaTemperatureMoves == 4);
    std::printf("test_presets_parse passed\n");
}

static void test_override() {
    RunConfig c;
    applyOverride(c, "simulations=50");
    assert(c.simulations == 50);
    applyOverride(c, " reward_head = true ");
    assert(c.rewardHead);
    assert(architectureKeys().size() == 7);
    std::printf("test_override passed\n");
}

static void expectThrowsNaming(const std::string& key, const std::string& kv, bool viaValidate) {
    RunConfig c;
    try {
        applyOverride(c, kv);
        if (viaValidate) validate(c);
    } catch (const std::invalid_argument& e) {
        assert(std::string(e.what()).find(key) != std::string::npos);
        return;
    }
    assert(false && "expected std::invalid_argument");
}

static void test_errors() {
    expectThrowsNaming("foo", "foo=1", false);
    expectThrowsNaming("board_size", "board_size=4", true);
    expectThrowsNaming("filters", "filters=x", false);
    expectThrowsNaming("filters", "filters=0", true);
    expectThrowsNaming("noise_fraction", "noise_fraction=1.5", true);
    expectThrowsNaming("lr_init", "lr_init=0", true);
    expectThrowsNaming("unroll_steps", "unroll_steps=0", true);
    // Ranges the trainer relies on.
    expectThrowsNaming("lr_decay_steps", "lr_decay_steps=0", true);
    expectThrowsNaming("train_steps", "train_steps=-1", true);
    expectThrowsNaming("workers", "workers=-1", true);
    expectThrowsNaming("train_threads", "train_threads=-1", true);
    expectThrowsNaming("arena_games", "arena_games=-2", true);
    expectThrowsNaming("arena_games", "arena_games=3", true);
    expectThrowsNaming("arena_every", "arena_every=-1", true);
    expectThrowsNaming("arena_gap", "arena_gap=0", true);
    expectThrowsNaming("arena_temperature_moves", "arena_temperature_moves=-1", true);
    expectThrowsNaming("rollout_bot_simulations", "rollout_bot_simulations=0", true);
    expectThrowsNaming("temperature_moves", "temperature_moves=-1", true);
    expectThrowsNaming("selfplay_group", "selfplay_group=0", true);
    expectThrowsNaming("max_moves", "max_moves=-1", true);
    expectThrowsNaming("sgf_per_generation", "sgf_per_generation=-1", true);
    // Their boundary values are accepted.
    RunConfig edge;
    for (const char* kv : {"train_steps=0", "workers=0", "train_threads=0", "arena_games=0", "arena_every=0",
                           "temperature_moves=0", "max_moves=0", "lr_decay_steps=1", "selfplay_group=1",
                           "arena_gap=1", "sgf_per_generation=0"})
        applyOverride(edge, kv);
    validate(edge);
    std::printf("test_errors passed\n");
}

int main() {
    test_defaults_round_trip();
    test_presets_parse();
    test_override();
    test_errors();
    return 0;
}
