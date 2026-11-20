// Tests for the ReplayBuffer: uniform position sampling, capacity, save/load.
#include <cassert>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "mzgo/encode.hpp"
#include "mzgo/game_record.hpp"
#include "mzgo/replay_buffer.hpp"
#include "mzgo/rng.hpp"
#include "mzgo/serialize.hpp"

using namespace mzgo;

static const int N = 5, A = N * N + 1;

// A record of T moves whose every policy is one-hot on point `id` (so a sampled
// position's game is recoverable from its policy target when sym = 0).
static GameRecord rec(int T, int id) {
    GameRecord r;
    r.boardSize = N; r.komi = 0.5f;
    for (int p = 0; p < T; ++p) r.moves.push_back((id + p) % (N * N));
    r.stones.assign(static_cast<size_t>(T + 1) * N * N, 0);
    for (size_t i = 0; i < r.stones.size(); ++i) r.stones[i] = static_cast<uint8_t>((i + id) % 3);
    r.policies.assign(static_cast<size_t>(T) * A, 0.0f);
    for (int p = 0; p < T; ++p) r.policies[static_cast<size_t>(p) * A + id] = 1.0f;
    r.rootValues.assign(T, 0.25f * id);
    r.resultForBlack = id % 2 ? 1.0f : -1.0f;
    return r;
}

static bool same(const GameRecord& a, const GameRecord& b) {
    return a.boardSize == b.boardSize && a.komi == b.komi && a.moves == b.moves &&
           a.stones == b.stones && a.policies == b.policies && a.rootValues == b.rootValues &&
           a.resultForBlack == b.resultForBlack;
}

static int argmax0(const TrainBatch& b, int s) {
    const float* p = b.policyTargets.data() + static_cast<size_t>(s) * (b.K + 1) * A;
    int arg = 0;
    for (int a = 0; a < A; ++a) if (p[a] > p[arg]) arg = a;
    return arg;
}

static void test_uniform_positions() {
    ReplayBuffer buf(10);
    buf.add(rec(10, 1));
    buf.add(rec(30, 2));
    assert(buf.games() == 2 && buf.positions() == 40);
    Rng rng(7);
    const int B = 40000;
    TrainBatch out;
    buf.sampleBatch(B, 1, 1, false, false, rng, out);
    assert(out.B == B && out.K == 1);
    int fromLong = 0;
    for (int b = 0; b < B; ++b) fromLong += argmax0(out, b) == 2;
    const double share = static_cast<double>(fromLong) / B;
    assert(std::fabs(share - 0.75) < 0.02);

    // Same seed -> identical batch (determinism), augmentation included.
    TrainBatch x, y;
    Rng r1(99), r2(99);
    buf.sampleBatch(64, 3, 2, false, true, r1, x);
    buf.sampleBatch(64, 3, 2, false, true, r2, y);
    assert(x.observations == y.observations && x.actions == y.actions &&
           x.policyTargets == y.policyTargets && x.valueTargets == y.valueTargets);
    std::printf("test_uniform_positions ok (long share %.4f)\n", share);
}

static void test_capacity() {
    ReplayBuffer buf(2);
    buf.add(rec(4, 1));
    buf.add(rec(5, 2));
    buf.add(rec(6, 3));
    assert(buf.games() == 2 && buf.positions() == 11);
    assert(buf.game(0).moves[0] == 2 && buf.game(1).moves[0] == 3);
    // After dropping, sampling only sees the remaining games (prefix sums recomputed).
    Rng rng(5);
    TrainBatch out;
    buf.sampleBatch(2000, 1, 1, false, false, rng, out);
    for (int b = 0; b < 2000; ++b) assert(argmax0(out, b) != 1);
    std::printf("test_capacity ok\n");
}

static void test_save_load() {
    ReplayBuffer buf(3);
    buf.add(rec(7, 1));
    buf.add(rec(9, 4));
    std::stringstream ss;
    { BinaryWriter w(ss); buf.save(w); }
    const std::string data = ss.str();
    {
        std::stringstream in(data);
        BinaryReader r(in);
        ReplayBuffer back = ReplayBuffer::load(r);
        assert(back.games() == 2 && back.positions() == buf.positions());
        for (int i = 0; i < 2; ++i) assert(same(back.game(i), buf.game(i)));
        back.add(rec(2, 5)); back.add(rec(2, 6));   // capacity survives the round trip
        assert(back.games() == 3);
    }
    // Truncation anywhere throws.
    for (size_t cut : {size_t(0), size_t(3), size_t(10), data.size() / 2, data.size() - 1}) {
        std::stringstream in(data.substr(0, cut));
        BinaryReader r(in);
        bool threw = false;
        try { ReplayBuffer::load(r); } catch (const std::runtime_error&) { threw = true; }
        assert(threw);
    }
    // Bad magic and bad version throw.
    for (size_t pos : {size_t(0), size_t(4)}) {
        std::string bad = data;
        bad[pos] = static_cast<char>(bad[pos] ^ 0x55);
        std::stringstream in(bad);
        BinaryReader r(in);
        bool threw = false;
        try { ReplayBuffer::load(r); } catch (const std::runtime_error&) { threw = true; }
        assert(threw);
    }
    std::printf("test_save_load ok\n");
}

// A buffer holds one board size: a file whose games disagree is refused.
static void test_load_refuses_mixed_board_sizes() {
    GameRecord big = rec(3, 1);
    big.boardSize = 7;
    big.stones.assign(4u * 49, 0);
    big.policies.assign(3u * 50, 0.0f);
    ReplayBuffer buf(3);
    buf.add(rec(4, 2));
    buf.add(big);
    std::stringstream ss;
    { BinaryWriter w(ss); buf.save(w); }
    BinaryReader r(ss);
    bool threw = false;
    try {
        ReplayBuffer::load(r);
    } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("board size") != std::string::npos;
    }
    assert(threw);
    std::printf("test_load_refuses_mixed_board_sizes ok\n");
}

int main() {
    test_uniform_positions();
    test_capacity();
    test_save_load();
    test_load_refuses_mixed_board_sizes();
    return 0;
}
