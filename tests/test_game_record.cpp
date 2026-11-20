// Tests for GameRecord, encodeRecordPosition and fillSample (MuZero targets).
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include "mzgo/encode.hpp"
#include "mzgo/game.hpp"
#include "mzgo/game_record.hpp"
#include "mzgo/rng.hpp"
#include "mzgo/symmetry.hpp"
#include "mzgo/train_batch.hpp"

using namespace mzgo;

static const int N = 5, A = N * N + 1;

// One-hot policy on the move played, value 0.1 * ply as a recognizable root value.
static GameRecord playedRecord(const std::vector<int>& moves, bool passPass) {
    Game g(N, 0.5f, defaultMaxMoves(N));
    for (int m : moves) g.play(m);
    if (passPass) { g.play(N * N); g.play(N * N); }
    const int T = g.moveCount();
    std::vector<float> pol(static_cast<size_t>(T) * A, 0.0f), rv(T);
    for (int p = 0; p < T; ++p) {
        pol[static_cast<size_t>(p) * A + g.moves()[p]] = 1.0f;
        rv[p] = 0.1f * p;
    }
    return recordFromGame(g, pol, rv);
}

// Moves on distinct points; with endsWithPasses the last two are passes (a
// game the players ended), else the game stopped at the move cap.
static GameRecord handRecord(int T, float resultForBlack, bool endsWithPasses) {
    GameRecord r;
    r.boardSize = N; r.komi = 0.5f;
    for (int p = 0; p < T; ++p) r.moves.push_back(endsWithPasses && p >= T - 2 ? N * N : p);
    r.stones.assign(static_cast<size_t>(T + 1) * N * N, 0);
    r.policies.assign(static_cast<size_t>(T) * A, 0.0f);
    for (int p = 0; p < T; ++p) r.policies[static_cast<size_t>(p) * A + r.moves[p]] = 1.0f;
    r.rootValues.assign(T, 0.0f);
    r.resultForBlack = resultForBlack;
    return r;
}

static void test_record_from_game() {
    const std::vector<int> mv = {0, 24, 12, 18, 6, 20};
    Game g(N, 0.5f, defaultMaxMoves(N));
    for (int m : mv) g.play(m);
    g.play(N * N); g.play(N * N);
    GameRecord r = playedRecord(mv, true);
    assert(r.length() == 8 && r.boardSize == N);
    assert(r.stones.size() == 9u * N * N);
    std::vector<uint8_t> s(N * N);
    g.boardAt(3).exportStones(s.data());
    for (int i = 0; i < N * N; ++i) assert(r.stones[3 * N * N + i] == s[i]);
    assert(r.resultForBlack == g.resultForBlack());
    assert(r.toMoveAt(0) == Color::Black && r.toMoveAt(1) == Color::White);
    for (int p = 0; p < r.length(); ++p) {
        const float want = (p % 2 == 0) ? r.resultForBlack : -r.resultForBlack;
        assert(r.outcomeFor(p) == want);
    }
    // encodeRecordPosition == encodeGame on the replayed game, for every position/symmetry.
    const int H = 3, planes = numPlanes(H);
    for (int p = 0; p < r.length(); ++p) {
        Game replay(N, 0.5f, defaultMaxMoves(N));
        for (int i = 0; i < p; ++i) replay.play(r.moves[i]);
        for (int sym : {0, 3, 5}) {
            std::vector<float> a(planes * N * N), b(planes * N * N);
            encodeGame(replay, H, sym, a.data());
            encodeRecordPosition(r, p, H, sym, b.data());
            assert(a == b);
        }
    }
    std::printf("test_record_from_game ok\n");
}

static void test_targets_signs() {
    GameRecord r = handRecord(8, 1.0f, false);   // Black won (at the cap; t + K < T)
    Rng rng(1);
    TrainBatch b;
    b.resize(1, 4, numPlanes(2), N * N, A);
    fillSample(r, 0, 4, 2, 0, rng, b, 0);
    for (int k = 0; k <= 4; ++k) {
        assert(b.valueTargets[k] == (k % 2 == 0 ? 1.0f : -1.0f));
        assert(b.policyMask[k] == 1.0f && b.valueMask[k] == 1.0f && b.rewardMask[k] == 1.0f);
        assert(b.policyTargets[static_cast<size_t>(k) * A + k] == 1.0f);
    }
    for (int k = 0; k < 4; ++k) assert(b.actions[k] == k);
    for (int k = 1; k <= 4; ++k) assert(b.rewardTargets[k] == 0.0f);
    std::printf("test_targets_signs ok\n");
}

static void test_short_game_padding() {
    GameRecord r = handRecord(3, 1.0f, true);   // T = 3: B1, pass, pass; Black won; move 2 made by Black
    assert(r.endedByPasses());
    Rng rng(2);
    const int K = 5, t = 1;
    TrainBatch b;
    b.resize(1, K, numPlanes(1), N * N, A);
    fillSample(r, t, K, 1, 0, rng, b, 0);
    for (int k = 0; k <= K; ++k) {
        const int p = t + k;
        if (p < 3) {
            assert(b.policyMask[k] == 1.0f);
            assert(b.valueTargets[k] == r.outcomeFor(p));
        } else {
            assert(b.policyMask[k] == 0.0f);
            assert(b.valueTargets[k] == 0.0f);
            for (int a = 0; a < A; ++a) assert(b.policyTargets[static_cast<size_t>(k) * A + a] == 0.0f);
        }
        if (k >= 1) assert(b.rewardTargets[k] == (p == 3 ? r.outcomeFor(2) : 0.0f));
        // A game that ended: the absorbing state past T is a real target.
        assert(b.valueMask[k] == 1.0f && b.rewardMask[k] == 1.0f);
    }
    assert(b.rewardTargets[2] == 1.0f);   // Black made move 2 and won
    assert(b.actions[0] == N * N && b.actions[1] == N * N);
    for (int k = 0; k < K; ++k) assert(b.actions[k] >= 0 && b.actions[k] < A);
    std::printf("test_short_game_padding ok\n");
}

static void test_symmetry_consistency() {
    GameRecord r = handRecord(6, -1.0f, false);
    Rng rng(3);
    const int t = 2, K = 2, sym = 3;
    TrainBatch b;
    b.resize(1, K, numPlanes(2), N * N, A);
    fillSample(r, t, K, 2, sym, rng, b, 0);
    for (int k = 0; k < K; ++k) assert(b.actions[k] == transformAction(sym, r.moves[t + k], N));
    int arg = 0;
    for (int a = 0; a < A; ++a) if (b.policyTargets[a] > b.policyTargets[arg]) arg = a;
    assert(arg == transformAction(sym, r.moves[t], N));
    std::vector<float> obs(numPlanes(2) * N * N);
    encodeRecordPosition(r, t, 2, sym, obs.data());
    assert(obs == b.observations);
    std::printf("test_symmetry_consistency ok\n");
}

// A game stopped by the move cap was truncated, not ended: its last move
// pays nothing (reward 0 on every edge), the position at the cap is valued
// by its score (outcomeFor(T)), and past T nothing is known (every mask 0).
static void test_capped_game_truncated() {
    const int T = 6;
    GameRecord r = handRecord(T, 1.0f, false);   // Black ahead at the cap
    assert(!r.endedByPasses());
    Rng rng(4);
    const int K = 5, t = 3;
    TrainBatch b;
    b.resize(1, K, numPlanes(1), N * N, A);
    fillSample(r, t, K, 1, 0, rng, b, 0);
    for (int k = 0; k <= K; ++k) {
        const int p = t + k;
        if (k >= 1) assert(b.rewardTargets[k] == 0.0f);
        if (p < T) {
            assert(b.policyMask[k] == 1.0f && b.valueMask[k] == 1.0f && b.rewardMask[k] == 1.0f);
            assert(b.valueTargets[k] == r.outcomeFor(p));
        } else if (p == T) {
            assert(b.policyMask[k] == 0.0f && b.valueMask[k] == 1.0f && b.rewardMask[k] == 1.0f);
            assert(b.valueTargets[k] == r.outcomeFor(T));
        } else {
            assert(b.policyMask[k] == 0.0f && b.valueMask[k] == 0.0f && b.rewardMask[k] == 0.0f);
        }
    }
    assert(b.valueTargets[3] == 1.0f);            // Black to move at the cap, and ahead
    std::printf("test_capped_game_truncated ok\n");
}

static void test_ended_by_passes() {
    assert(handRecord(3, 1.0f, true).endedByPasses());
    assert(!handRecord(3, 1.0f, false).endedByPasses());
    assert(!handRecord(1, 1.0f, false).endedByPasses());
    GameRecord one = handRecord(2, 1.0f, false);
    one.moves = {N * N, 3};                       // a pass, then a move: not ended
    assert(!one.endedByPasses());
    one.moves = {N * N, N * N};
    assert(one.endedByPasses());
    std::printf("test_ended_by_passes ok\n");
}

int main() {
    test_record_from_game();
    test_targets_signs();
    test_short_game_padding();
    test_symmetry_consistency();
    test_capped_game_truncated();
    test_ended_by_passes();
    return 0;
}
