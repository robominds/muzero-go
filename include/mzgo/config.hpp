// Run configuration.
//
// One RunConfig holds every knob of a training run: the game, the network
// shape, self-play search, training and the arena. A run is described by a
// plain text file ("key = value" lines, '#' starts a comment) and any number
// of command-line overrides ("--set key=value"). Keeping the whole run in one
// printable struct makes an experiment reproducible: configToString() writes
// every key in a canonical form that parses back to an identical config.
//
// Keys are the snake_case spelling of the field names (boardSize -> board_size).
// The "architecture keys" are the ones that fix the shape of the network's
// weights; a checkpoint can only be loaded into a run that agrees on them.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace mzgo {

struct RunConfig {
    // run
    uint64_t seed = 1; int generations = 60; int workers = 0; int trainThreads = 0; std::string backend = "blas";
    // game
    int boardSize = 5; float komi = 24.5f; int maxMoves = 0;              // 0 -> 2*n*n
    // network (architecture keys)
    int history = 8, reprBlocks = 3, dynBlocks = 3, filters = 32, valueHidden = 32; bool rewardHead = false;
    // search / self-play
    int gamesPerGeneration = 200, simulations = 200, selfplayGroup = 8;
    float dirichletScale = 10, noiseFraction = 0.25f; int temperatureMoves = 8;
    // training
    int replayGames = 2000, trainSteps = 200, batchSize = 128, unrollSteps = 5;
    float lrInit = 0.02f, lrDecayRate = 0.1f; int lrDecaySteps = 12000;
    float momentum = 0.9f, weightDecay = 1e-4f; bool symmetryAugment = false; bool bnCalibrate = false;
    // arena
    int arenaEvery = 1, arenaGames = 20, arenaGap = 5, arenaSimulations = 100, arenaTemperatureMoves = 2, rolloutBotSimulations = 200;
    // output
    int sgfPerGeneration = 2;

    // maxMoves, or 2*n*n when maxMoves is 0.
    int resolvedMaxMoves() const;
};

// All of these throw std::invalid_argument; the message names the offending key.
RunConfig parseConfigFile(const std::string& path);          // '#' comments, "key = value"
RunConfig parseConfigString(const std::string& text);        // same, from memory
void applyOverride(RunConfig& cfg, const std::string& keyEqualsValue);  // --set key=value
std::string configToString(const RunConfig& cfg);            // every key, canonical, re-parseable
std::vector<std::string> architectureKeys();                 // keys that fix the network's weight shapes
void validate(const RunConfig& cfg);                         // range checks

}  // namespace mzgo
