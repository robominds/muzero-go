// The replay window: the most recent self-play games, sampled for training.
//
// Training batches draw positions uniformly over every stored position (so a
// long game contributes proportionally more than a short one). A prefix sum of
// game lengths turns "pick a global position index" into "which game, which
// ply" with one binary search.
#pragma once
#include <vector>
#include "mzgo/game_record.hpp"
#include "mzgo/rng.hpp"
#include "mzgo/serialize.hpp"
#include "mzgo/train_batch.hpp"

namespace mzgo {

class ReplayBuffer {
public:
    explicit ReplayBuffer(int capacityGames);
    void add(GameRecord g);           // drops the oldest game beyond capacity
    int games() const { return static_cast<int>(games_.size()); }
    long positions() const { return prefix_.empty() ? 0 : prefix_.back(); }
    const GameRecord& game(int i) const { return games_.at(static_cast<size_t>(i)); }

    // B samples, each a position chosen uniformly over all stored positions (rng),
    // symmetry uniform in [0,8) when augment, else 0. Per sample the rng is
    // consumed in a fixed order: position, symmetry, then padding actions.
    void sampleBatch(int B, int K, int history, bool passPlane, bool augment, Rng& rng, TrainBatch& out) const;

    // Format: "MZRB", u32 version 2, u32 capacity, u32 game count, then games.
    // Each game: board size, komi, result, T, moves, stones, policies, root
    // values, then (version 2) a u8 flag (0 = empty policy mask, 1 = present)
    // and, when present, T mask bytes, then a u8 train-value flag (0 or 1). Version
    // 1 (neither) still loads, with empty masks and train-value true.
    void save(BinaryWriter&) const;
    // Throws std::runtime_error on bad data, including games whose board
    // size differs from the first game's.
    static ReplayBuffer load(BinaryReader&);

private:
    void rebuildPrefix();
    int capacity_;
    std::vector<GameRecord> games_;
    std::vector<long> prefix_;        // prefix_[i] = positions in games 0..i
};

} // namespace mzgo
