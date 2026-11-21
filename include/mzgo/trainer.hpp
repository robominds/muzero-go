// The training loop: where self-play, the replay window, the unrolled loss
// and the arena come together.
//
// Training runs in synchronous generations (the paper runs actors and a
// learner asynchronously; generations are the scaled-down equivalent).
// Generation g (counting from 1):
//   1. self-play: the current network plays games_per_generation games
//      against itself, each move chosen by MuZero search (selfplay.hpp).
//      With a league (league_size > 0, league.hpp), the last
//      L = round(games_per_generation * league_fraction) of them are league
//      games instead: game j against the member in slot j mod league_size
//      (seed deriveSeed(seed, g, 4); the self-play games keep theirs);
//   2. the games join the replay window of the last replay_games games
//      (the self-play games, then the league games);
//   3. train_steps SGD steps, each on a batch of K-step unrolls sampled
//      from the window (network.hpp's trainStep, then the optimizer);
//      with a league, then at most one beaten member is replaced by the
//      network that played (chooseReplacement);
//   4. the new network is saved as checkpoints/gen_GGGG.mzn, then the whole
//      run's state as state.bin;
//   5. every arena_every generations the network plays an earlier
//      generation, the rollout bot and the random bot;
//   6. one row of log.csv and one summary line.
//
// A run lives in one directory:
//   config.cfg        the resolved config (every key)
//   log.csv           one row per generation: self-play statistics, the lr
//                     of the generation's first step, the losses averaged
//                     over its steps (empty with train_steps = 0), the
//                     arena scores of the new network in [0, 1] (empty when
//                     the arena did not run), the replay window's size; the
//                     statistics cover league games too. With a league,
//                     three more: league_members (the slots after this
//                     generation's replacement, "random" or "gen_N",
//                     ';'-joined), league_scores (the learner's score
//                     against each slot's member before it, empty for one
//                     with no games) and league_replaced (the slot, or -1)
//   checkpoints/      gen_0000.mzn (the starting network), gen_0001.mzn, ...
//   sgf/              the first sgf_per_generation self-play games of each
//                     generation, gen_GGGG_I.sgf (I from 0), and as many
//                     league games, gen_GGGG_league_I.sgf
//   state.bin         what a resume needs (below)
//
// Resume: every random choice of generation g comes from a seed derived
// from (seed, g, phase, ...), never from a generator carried across
// generations. So state.bin only needs the generation, the global step,
// the network (weights and batch norm statistics), the optimizer's velocity,
// the replay window and the league (a network member plays from its
// checkpoint), and a run stopped after any generation and resumed writes
// exactly the bytes an uninterrupted run would have written.
#pragma once
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>
#include "mzgo/backend.hpp"
#include "mzgo/config.hpp"
#include "mzgo/league.hpp"
#include "mzgo/network.hpp"
#include "mzgo/optimizer.hpp"
#include "mzgo/parallel.hpp"
#include "mzgo/replay_buffer.hpp"

namespace mzgo {

// The architecture keys of a RunConfig as the network's shape.
NetConfig netConfigFrom(const RunConfig& c);

// runDir/checkpoints/gen_GGGG.mzn (four digits, zero-padded).
std::string checkpointPath(const std::string& runDir, int generation);

class Trainer {
public:
    // New run: refuses (std::runtime_error) a runDir that exists and is not
    // empty. Writes runDir/config.cfg (configToString), the generation 0
    // network (seed deriveSeed(seed, 0xA11)) as gen_0000.mzn, state.bin and
    // log.csv's header. A league starts with every slot the random bot.
    // With cfg.bnCalibrate, generation 0's batch norm
    // statistics (every unroll step's set) are first set from one K-step
    // batch of random-move positions.
    // Throws std::invalid_argument for an invalid config (validate()).
    static Trainer startNew(const RunConfig& cfg, const std::string& runDir);
    // Resume: reads config.cfg and state.bin and rewrites config.cfg with the
    // overrides applied. Overrides may change only non-architecture keys
    // (restating an architecture key's value is allowed; unroll_steps is one,
    // since batch norm keeps statistics per unroll step and log.csv has
    // columns per step), else std::invalid_argument naming the key; nor may
    // league_size. A missing or corrupt state.bin, or one whose league has
    // other than league_size members, is a std::runtime_error naming the
    // file, and so is a league member whose checkpoint is missing.
    static Trainer resume(const std::string& runDir, const std::vector<std::string>& overrides);

    Trainer(Trainer&&) = default;
    Trainer& operator=(Trainer&&) = default;

    int generation() const { return generation_; }       // generations completed
    long globalStep() const { return globalStep_; }      // SGD steps completed
    const RunConfig& config() const { return cfg_; }
    const std::vector<LeagueMember>& league() const { return league_; }   // league_size slots
    const std::string& runDir() const { return runDir_; }
    // Where the one-line summary of each generation goes (none by default).
    void setLog(std::ostream* out) { log_ = out; }

    // One generation: self-play (and league games), replay add,
    // train_steps steps, league replacement, checkpoint, state.bin, SGF
    // files, arena, log row. A non-finite loss throws
    // std::runtime_error("non-finite loss at generation g step s"), and a
    // non-finite weight or batch norm statistic after the last step throws
    // std::runtime_error("non-finite weights at generation g step s"), both
    // before anything of the generation is written. After either, the
    // Trainer is partly advanced in memory and should be discarded.
    void runGeneration();
    void run();                                          // until generation() == cfg.generations

private:
    Trainer(const RunConfig& cfg, std::string runDir, MuZeroNetwork net);
    void calibrateBatchNorm();
    void writeState() const;

    RunConfig cfg_;
    std::string runDir_;
    int generation_ = 0;
    long globalStep_ = 0;
    std::ostream* log_ = nullptr;
    std::unique_ptr<Backend> backend_;
    std::unique_ptr<ThreadPool> pool_;                   // the training step's threads
    std::unique_ptr<MuZeroNetwork> net_;                 // on the heap: models and workspaces refer to it
    std::unique_ptr<TrainWorkspace> workspace_;
    SgdMomentum optimizer_;
    ReplayBuffer replay_;
    std::vector<LeagueMember> league_;
};

} // namespace mzgo
