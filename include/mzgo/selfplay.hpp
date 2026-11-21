// Self-play: the model plays both sides against itself, choosing every move
// by search, and each finished game becomes a GameRecord for training.
//
// Every move runs a full MuZero search (mcts.hpp) from the real position:
// one initial inference on the observation, then `simulations` recurrent
// inferences, one per imagined leaf. Run one game at a time and every model
// call has batch size 1, which wastes a matrix-multiply backend. So a worker
// thread plays a *group* of games in lockstep instead: each step it runs one
// simulation's selection in every live game of the group, evaluates all those
// leaves in a single batched recurrentInference, and hands each answer back
// to its own game (the roots of games starting a move share one batched
// initialInference the same way). Batching only spans independent games;
// each game's tree is exactly the sequential search, so nothing like the
// "virtual loss" trick that batches leaves within one tree is needed.
//
// Exploration in self-play comes from two sources: Dirichlet noise mixed
// into the root priors (cfg.mcts.addNoise), and sampling the move in
// proportion to visit counts for the first temperatureMoves moves (argmax
// afterwards). Each game i draws all its randomness, in a fixed order, from
// its own Rng(deriveSeed(seed, i)), so neither the thread count nor which
// games share a group changes its random stream.
//
// League games (playLeagueGames) pit the network being trained, the learner,
// against another player: an earlier network, which searches exactly like the
// learner, or the random-move bot. Only the learner's own moves train the
// policy (GameRecord::policyMask); the opponent's positions still train
// value, reward and dynamics. A group then holds games against one opponent,
// and each step splits its live games by whose turn it is: the learner's
// searches and the opponent network's are batched separately, each through
// its own session, and the bot moves at once, drawing from the game's Rng.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "mzgo/game_record.hpp"
#include "mzgo/mcts.hpp"
#include "mzgo/model.hpp"
#include "mzgo/players.hpp"

namespace mzgo {

struct SelfPlayConfig {
    MctsConfig mcts;                   // addNoise true; alpha = dirichletScale / (n*n)
    int boardSize, maxMoves, temperatureMoves; float komi;
    int group = 8;                     // games advanced in lockstep by one worker
};

struct SelfPlayStats {
    double seconds;          // wall time of the whole call
    double meanLength;       // moves per game
    // The search statistics cover the learner's searches only (in self-play, every search).
    double meanDepth;        // SearchResult::meanDepth averaged over all searches
    int maxDepth;            // deepest leaf of any search
    double pastEndFraction;  // SearchResult::pastEndFraction averaged over all searches
    int cappedGames;         // games ended by the move cap rather than two passes
};

// Game i uses deriveSeed(seed, i). Games are cut into groups of `group` consecutive indices;
// groups are handed to `workers` threads. Each step a group runs one simulation in every live
// game, batching their leaves into one recurrentInference (roots: one initialInference).
// Output order = game index; bits independent of `workers` (not of `group`).
std::vector<GameRecord> playSelfPlay(const Model&, const SelfPlayConfig&, int games, uint64_t seed,
                                     int workers, SelfPlayStats* stats = nullptr);

// A league opponent: a network that plays MuZero search with the self-play
// settings (noise and temperature as for the learner), or, when model is
// null, the random-move bot (RandomPlayer).
struct LeagueOpponent { const Model* model = nullptr; };

// Game i plays the learner against opponents[opponentOf[i]]; the learner is
// Black when i is even. Every record's policyMask is 1 at the learner's
// positions and 0 at the opponent's. A bot move records a uniform-zero policy
// row and root value 0. Same seeding and worker-independence guarantees as
// playSelfPlay (game i draws from Rng(deriveSeed(seed, i))). Games sharing an
// opponent are batched in lockstep groups of cfg.group; the learner's and a
// network opponent's searches are batched separately, each through its own
// session. All models must agree on dims(), history() and passPlane() (asserted).
std::vector<GameRecord> playLeagueGames(const Model& learner, const std::vector<LeagueOpponent>& opponents,
                                        const std::vector<int>& opponentOf, const SelfPlayConfig& cfg,
                                        uint64_t seed, int workers, SelfPlayStats* stats = nullptr);

// Search player for arena and GTP: no noise, sample ∝ visits for the first `temperatureMoves`
// moves, then argmax. One session per player instance.
class MuZeroPlayer : public Player {
public:
    MuZeroPlayer(const Model&, const MctsConfig&, int temperatureMoves, std::string name = "muzero");
    std::string name() const override { return name_; }
    int selectMove(const Game& game, Rng& rng) override;
    SearchResult lastSearch() const;   // for GTP's stderr line and analyze

private:
    const Model& model_;
    int temperatureMoves_;
    std::string name_;
    std::unique_ptr<ModelSession> session_;
    Search search_;
    SearchResult last_{};
};

} // namespace mzgo
