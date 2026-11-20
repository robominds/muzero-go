// Matches between two players.
//
// A match is a batch of independent games between player A and player B, with
// A taking Black in even-numbered games and White in odd ones so neither side
// gets the first-move advantage for free. Game i draws all its randomness from
// Rng(deriveSeed(seed, i)) and results are stored by index, so a match gives
// identical output for any number of worker threads.
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>
#include "mzgo/players.hpp"

namespace mzgo {

// Each game gets fresh players, so factories must be safe to call from several threads.
using PlayerFactory = std::function<std::unique_ptr<Player>()>;

struct GameRecordLite {
    std::vector<int> moves;
    float resultForBlack = 0;
    bool aWasBlack = false;
};

struct MatchResult {
    int winsA = 0, winsB = 0, draws = 0;
    double scoreA() const;   // (winsA + draws / 2) / games; 0 if no games
    std::vector<GameRecordLite> games;
};

// maxMoves 0 means defaultMaxMoves(boardSize).
MatchResult playMatch(const PlayerFactory& a, const PlayerFactory& b, int games, int boardSize, float komi,
                      int maxMoves, uint64_t seed, int workers);

} // namespace mzgo
