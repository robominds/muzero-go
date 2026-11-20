// Copied from alphazero-go (Mark Castelluccio), adapted for mzgo.
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <vector>
#include "mzgo/encode.hpp"
#include "mzgo/game.hpp"
#include "mzgo/symmetry.hpp"

using namespace mzgo;

static int pt(int row, int col, int n = 5) { return row * n + col; }

static std::vector<float> encodeB(const std::vector<const Board*>& history, int H, int sym) {
    const int n = history[0]->size();
    std::vector<float> out(static_cast<size_t>(numPlanes(H)) * n * n, -1.0f);  // -1: catch unwritten
    encodeBoards(history, H, sym, out.data());
    return out;
}

static std::vector<float> encodeG(const Game& g, int H, int sym, bool passPlane = false) {
    const int n = g.size();
    std::vector<float> out(static_cast<size_t>(numPlanes(H, passPlane)) * n * n, -1.0f);
    encodeGame(g, H, sym, out.data(), passPlane);
    return out;
}

static float at(const std::vector<float>& planes, int plane, int point, int n = 5) {
    return planes[static_cast<size_t>(plane) * n * n + point];
}

// The plane holds exactly `ones` set to 1 and every other point 0.
static bool planeIs(const std::vector<float>& planes, int plane, const std::vector<int>& ones, int n = 5) {
    for (int p = 0; p < n * n; ++p) {
        bool expectOne = false;
        for (int q : ones) expectOne = expectOne || (q == p);
        if (at(planes, plane, p, n) != (expectOne ? 1.0f : 0.0f)) return false;
    }
    return true;
}

static std::vector<int> all(int n = 5) {
    std::vector<int> v;
    for (int p = 0; p < n * n; ++p) v.push_back(p);
    return v;
}

static void test_plane_counts() {
    assert(numPlanes(1) == 4);
    assert(numPlanes(8) == 18);
    Board b(9);
    std::vector<float> out = encodeB({&b}, 8, 0);
    assert(out.size() == 18u * 81u);
}

static void test_own_and_opponent_swap_with_side_to_move() {
    Board b(5);
    b.setStone(pt(0, 0), Color::Black);
    b.setStone(pt(4, 4), Color::White);

    b.setToMove(Color::Black);
    std::vector<float> black = encodeB({&b}, 1, 0);
    assert(planeIs(black, 0, {pt(0, 0)}));          // own = Black
    assert(planeIs(black, 1, {pt(4, 4)}));          // opponent = White
    assert(planeIs(black, 2, all()));               // Black to move
    assert(planeIs(black, 3, {}));

    b.setToMove(Color::White);
    std::vector<float> white = encodeB({&b}, 1, 0);
    assert(planeIs(white, 0, {pt(4, 4)}));          // own = White
    assert(planeIs(white, 1, {pt(0, 0)}));
    assert(planeIs(white, 2, {}));
    assert(planeIs(white, 3, all()));               // White to move
}

static void test_missing_history_is_zero() {
    Board b(5);
    b.setStone(pt(2, 2), Color::Black);
    std::vector<float> out = encodeB({&b}, 4, 0);
    assert(planeIs(out, 0, {pt(2, 2)}));
    for (int plane = 1; plane < 8; ++plane) assert(planeIs(out, plane, {}));
    assert(planeIs(out, 8, all()));
    assert(planeIs(out, 9, {}));

    // A fresh game: no stones anywhere, Black to move.
    Game g(5, 7.5f, defaultMaxMoves(5));
    std::vector<float> start = encodeG(g, 8, 0);
    for (int plane = 0; plane < 16; ++plane) assert(planeIs(start, plane, {}));
    assert(planeIs(start, 16, all()));
    assert(planeIs(start, 17, {}));
}

static void test_encode_game_history() {
    // Black (0,0), White (4,4), Black (2,2): White to move after 3 moves.
    Game g(5, 7.5f, defaultMaxMoves(5));
    g.play(pt(0, 0));
    g.play(pt(4, 4));
    g.play(pt(2, 2));
    std::vector<float> out = encodeG(g, 4, 0);
    // Own = White throughout, because White is to move now.
    assert(planeIs(out, 0, {pt(4, 4)}));            // current position
    assert(planeIs(out, 1, {pt(0, 0), pt(2, 2)}));
    assert(planeIs(out, 2, {pt(4, 4)}));            // one ply ago
    assert(planeIs(out, 3, {pt(0, 0)}));
    assert(planeIs(out, 4, {}));                    // two plies ago
    assert(planeIs(out, 5, {pt(0, 0)}));
    assert(planeIs(out, 6, {}));                    // the empty start position
    assert(planeIs(out, 7, {}));
    assert(planeIs(out, 8, {}));                    // not Black to move
    assert(planeIs(out, 9, all()));                 // White to move

    // H = 2 keeps only the two most recent positions.
    std::vector<float> shortH = encodeG(g, 2, 0);
    assert(shortH.size() == 6u * 25u);
    assert(planeIs(shortH, 2, {pt(4, 4)}));
    assert(planeIs(shortH, 3, {pt(0, 0)}));
    assert(planeIs(shortH, 5, all()));

    // encodeGame is encodeBoards over the game's boards, newest first.
    std::vector<float> manual = encodeB({&g.boardAt(3), &g.boardAt(2), &g.boardAt(1), &g.boardAt(0)}, 4, 0);
    assert(manual == out);
}

static void test_symmetry_consistency() {
    Game g(5, 7.5f, defaultMaxMoves(5));
    for (int m : {pt(0, 1), pt(3, 3), pt(1, 0), pt(4, 2)}) g.play(m);
    const int H = 3, n = 5, P = numPlanes(H);
    const std::vector<float> base = encodeG(g, H, 0);
    for (int s = 0; s < kNumSymmetries; ++s) {
        const std::vector<float> t = encodeG(g, H, s);
        for (int plane = 0; plane < P; ++plane) {
            for (int p = 0; p < n * n; ++p) {
                assert(at(t, plane, transformPoint(s, p, n)) == at(base, plane, p));
            }
        }
    }
    // A non-identity transform really moves the stones.
    assert(encodeG(g, H, 1) != base);
}

static void test_pass_plane() {
    const int H = 3, n = 5, P = numPlanes(H), pass = n * n;
    assert(numPlanes(H, false) == 2 * H + 2 && numPlanes(H, true) == 2 * H + 3);
    assert(numPlanes(1, true) == 5 && numPlanes(8, true) == 19);

    Game g(n, 7.5f, defaultMaxMoves(n));
    // Move 0: no move led here, so 0.
    std::vector<float> e = encodeG(g, H, 0, true);
    assert(e.size() == static_cast<size_t>(P + 1) * n * n);
    assert(planeIs(e, P, {}));
    g.play(pt(1, 1));
    assert(planeIs(encodeG(g, H, 0, true), P, {}));    // after a stone move
    g.play(pass);
    assert(planeIs(encodeG(g, H, 0, true), P, all()));  // after one pass
    g.play(pass);                                       // two passes end the game
    assert(planeIs(encodeG(g, H, 0, true), P, all()));  // after two passes
    // A stone move after a pass clears it again.
    Game h(n, 7.5f, defaultMaxMoves(n));
    h.play(pass);
    assert(planeIs(encodeG(h, H, 0, true), P, all()));
    h.play(pt(2, 2));
    assert(planeIs(encodeG(h, H, 0, true), P, {}));

    // Enabling it changes no other plane, and disabled output is exactly the old layout.
    for (const Game* game : {&g, &h}) {
        const std::vector<float> off = encodeG(*game, H, 0, false), on = encodeG(*game, H, 0, true);
        assert(off.size() == static_cast<size_t>(P) * n * n);
        assert(std::equal(off.begin(), off.end(), on.begin()));
    }

    // Invariant under every symmetry, on the pass plane and the rest.
    for (int s = 0; s < kNumSymmetries; ++s) {
        const std::vector<float> t = encodeG(g, H, s, true);
        assert(planeIs(t, P, all()));
        const std::vector<float> base = encodeG(g, H, s, false);
        assert(std::equal(base.begin(), base.end(), t.begin()));
        assert(planeIs(encodeG(h, H, s, true), P, {}));
    }

    // encodeBoards / encodePlanes take lastMovePass directly.
    Board b(n);
    std::vector<const Board*> hist = {&b};
    std::vector<float> out(static_cast<size_t>(numPlanes(H, true)) * n * n, -1.0f);
    encodeBoards(hist, H, 0, out.data(), true, true);
    assert(planeIs(out, P, all()));
    encodeBoards(hist, H, 0, out.data(), true, false);
    assert(planeIs(out, P, {}));
    std::printf("test_pass_plane passed\n");
}

int main() {
    test_pass_plane();
    test_plane_counts();
    test_own_and_opponent_swap_with_side_to_move();
    test_missing_history_is_zero();
    test_encode_game_history();
    test_symmetry_consistency();
    std::printf("test_encode: all tests passed\n");
    return 0;
}
