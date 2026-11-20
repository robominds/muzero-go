// Players named on a command line.
//
// A match between any two players (a trained network, the rollout bot, the
// random bot) needs a way to say which one: a short "spec" string.
//   ckpt:PATH     a MuZero network from a checkpoint, searching with
//                 `simulations` simulations, sampling its first
//                 `temperatureMoves` moves in proportion to visits, no noise
//   rollout:SIMS  RolloutMctsPlayer(SIMS), pure search on the real rules
//   random        RandomPlayer
// A match makes fresh players for every game (match.hpp), so a spec becomes
// a PlayerFactory. For ckpt the network is loaded once, when the factory is
// made, and every player made by the factory searches that same network,
// each with its own session.
#pragma once
#include <string>
#include "mzgo/match.hpp"

namespace mzgo {

struct PlayerSpecOptions {
    int boardSize = 9;           // a ckpt network must be for this board size
    int simulations = 200;       // ckpt: search simulations per move
    int temperatureMoves = 4;    // ckpt: moves sampled ∝ visits before argmax
    std::string backend = "blas";
};

// Throws std::invalid_argument naming the spec for an unknown or malformed
// spec or a checkpoint for another board size, and std::runtime_error (from
// loadNetwork) for a checkpoint that cannot be read.
PlayerFactory makePlayerFactory(const std::string& spec, const PlayerSpecOptions& options);

} // namespace mzgo
