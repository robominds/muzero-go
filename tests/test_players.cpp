#include <cassert>
#include <chrono>
#include <cstdio>
#include "mzgo/match.hpp"
#include "mzgo/players.hpp"

using namespace mzgo;

static int pt(int row, int col, int n = 5) { return row * n + col; }

static void test_fills_own_eye() {
    // Black eye at (0,0) with healthy neighbours; a White eye at (4,4).
    Board b = boardFromRows({".X...",
                             "XX...",
                             ".....",
                             "...OO",
                             "...O."}, Color::Black);
    assert(fillsOwnEye(b, pt(0, 0)));
    assert(!fillsOwnEye(b, pt(2, 2)));      // no neighbours of own colour
    assert(!fillsOwnEye(b, pt(0, 1)));      // occupied
    assert(!fillsOwnEye(b, pt(4, 4)));      // neighbours are White, Black to move
    b.setToMove(Color::White);
    assert(fillsOwnEye(b, pt(4, 4)));
    // Black stone (0,1) is in atari: its only liberty is (0,0).
    Board f = boardFromRows({".XO..",
                             "XO...",
                             ".....",
                             ".....",
                             "....."}, Color::Black);
    assert(f.libertiesAt(pt(0, 1)) == 1);
    assert(!fillsOwnEye(f, pt(0, 0)));
    std::printf("test_fills_own_eye passed\n");
}

static void test_random_legal() {
    Rng rng(7);
    RandomPlayer rp;
    int moves = 0;
    while (moves < 1000) {
        Game g(5, 0.5f, defaultMaxMoves(5));
        while (!g.isTerminal() && moves < 1000) {
            const int a = rp.selectMove(g, rng);
            assert(g.isLegal(a));
            if (a == g.board().passAction()) {
                for (int p = 0; p < g.board().area(); ++p)
                    assert(!(g.isLegal(p) && !fillsOwnEye(g.board(), p)));
            }
            g.play(a);
            ++moves;
        }
    }
    std::printf("test_random_legal passed\n");
}

static void test_rollout_beats_random() {
    auto t0 = std::chrono::steady_clock::now();
    PlayerFactory a = [] { return std::unique_ptr<Player>(new RolloutMctsPlayer(200)); };
    PlayerFactory b = [] { return std::unique_ptr<Player>(new RandomPlayer()); };
    MatchResult r = playMatch(a, b, 20, 5, 0.5f, 0, 11, 4);
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("  rollout score %.2f (%d-%d-%d) in %.1fs\n", r.scoreA(), r.winsA, r.winsB, r.draws, secs);
    assert(r.scoreA() >= 0.8);
    std::printf("test_rollout_beats_random passed\n");
}

int main() {
    test_fills_own_eye();
    test_random_legal();
    test_rollout_beats_random();
    return 0;
}
