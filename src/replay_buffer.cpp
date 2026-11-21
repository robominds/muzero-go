#include "mzgo/replay_buffer.hpp"
#include <algorithm>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <string>
#include "mzgo/encode.hpp"
#include "mzgo/symmetry.hpp"

namespace mzgo {

ReplayBuffer::ReplayBuffer(int capacityGames) : capacity_(capacityGames) {
    assert(capacityGames > 0);
}

void ReplayBuffer::rebuildPrefix() {
    prefix_.resize(games_.size());
    long sum = 0;
    for (size_t i = 0; i < games_.size(); ++i) {
        sum += games_[i].length();
        prefix_[i] = sum;
    }
}

void ReplayBuffer::add(GameRecord g) {
    games_.push_back(std::move(g));
    if (static_cast<int>(games_.size()) > capacity_) {
        games_.erase(games_.begin(), games_.begin() + (games_.size() - capacity_));
    }
    rebuildPrefix();
}

void ReplayBuffer::sampleBatch(int B, int K, int history, bool passPlane, bool augment, Rng& rng,
                               TrainBatch& out) const {
    const long total = positions();
    assert(total > 0 && total <= std::numeric_limits<int>::max());
    const GameRecord& first = games_.front();
    out.resize(B, K, numPlanes(history, passPlane), first.area(), first.numActions());
    for (int b = 0; b < B; ++b) {
        const long pick = rng.below(static_cast<int>(total));
        // First game whose cumulative count exceeds pick.
        const size_t gi = static_cast<size_t>(
            std::upper_bound(prefix_.begin(), prefix_.end(), pick) - prefix_.begin());
        const long before = gi == 0 ? 0 : prefix_[gi - 1];
        const int t = static_cast<int>(pick - before);
        const int sym = augment ? rng.below(kNumSymmetries) : 0;
        fillSample(games_[gi], t, K, history, passPlane, sym, rng, out, b);
    }
}

static const char kMagic[4] = {'M', 'Z', 'R', 'B'};
static const uint32_t kVersion = 2;
static const uint32_t kMaxMoves = 1u << 20;   // sanity bound against corrupt lengths

void ReplayBuffer::save(BinaryWriter& w) const {
    w.bytes(kMagic, 4);
    w.u32(kVersion);
    w.u32(static_cast<uint32_t>(capacity_));
    w.u32(static_cast<uint32_t>(games_.size()));
    for (const GameRecord& g : games_) {
        w.u32(static_cast<uint32_t>(g.boardSize));
        w.f32(g.komi);
        w.f32(g.resultForBlack);
        w.u32(static_cast<uint32_t>(g.length()));
        for (int m : g.moves) w.i32(m);
        w.bytes(g.stones.data(), g.stones.size());
        w.floats(g.policies.data(), g.policies.size());
        w.floats(g.rootValues.data(), g.rootValues.size());
        w.u8(g.policyMask.empty() ? 0 : 1);
        if (!g.policyMask.empty()) w.bytes(g.policyMask.data(), g.policyMask.size());
    }
}

ReplayBuffer ReplayBuffer::load(BinaryReader& r) {
    char magic[4];
    r.bytes(magic, 4);
    if (!std::equal(magic, magic + 4, kMagic)) throw std::runtime_error("replay buffer: bad magic");
    const uint32_t version = r.u32();
    if (version != 1 && version != kVersion) throw std::runtime_error("replay buffer: unsupported version");
    const uint32_t capacity = r.u32();
    const uint32_t count = r.u32();
    if (capacity == 0 || capacity > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        count > capacity) {
        throw std::runtime_error("replay buffer: bad capacity or game count");
    }
    ReplayBuffer buf(static_cast<int>(capacity));
    uint32_t firstSize = 0;
    for (uint32_t i = 0; i < count; ++i) {
        GameRecord g;
        const uint32_t n = r.u32();
        if (n < static_cast<uint32_t>(kMinBoardSize) || n > static_cast<uint32_t>(kMaxBoardSize)) {
            throw std::runtime_error("replay buffer: bad board size");
        }
        // sampleBatch sizes a batch from the first game: every game must agree.
        if (i == 0) firstSize = n;
        if (n != firstSize) {
            throw std::runtime_error("replay buffer: game " + std::to_string(i) + " has board size " +
                                     std::to_string(n) + " but the first game has " + std::to_string(firstSize));
        }
        g.boardSize = static_cast<int>(n);
        g.komi = r.f32();
        g.resultForBlack = r.f32();
        const uint32_t T = r.u32();
        if (T > kMaxMoves) throw std::runtime_error("replay buffer: bad game length");
        const int area = g.area();
        g.moves.resize(T);
        for (uint32_t p = 0; p < T; ++p) {
            const int m = r.i32();
            if (m < 0 || m > area) throw std::runtime_error("replay buffer: move out of range");
            g.moves[p] = m;
        }
        g.stones.resize(static_cast<size_t>(T + 1) * area);
        r.bytes(g.stones.data(), g.stones.size());
        for (uint8_t s : g.stones) {
            if (s > static_cast<uint8_t>(Color::White)) {
                throw std::runtime_error("replay buffer: bad stone value");
            }
        }
        g.policies.resize(static_cast<size_t>(T) * g.numActions());
        r.floats(g.policies.data(), g.policies.size());
        g.rootValues.resize(T);
        r.floats(g.rootValues.data(), g.rootValues.size());
        if (version >= 2) {
            const uint8_t flag = r.u8();
            if (flag > 1) throw std::runtime_error("replay buffer: bad policy mask flag");
            if (flag == 1) {
                g.policyMask.resize(T);
                r.bytes(g.policyMask.data(), g.policyMask.size());
            }
        }
        buf.add(std::move(g));
    }
    return buf;
}

} // namespace mzgo
