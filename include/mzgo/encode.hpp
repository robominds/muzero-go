// Copied from alphazero-go (Mark Castelluccio), adapted for mzgo.
// What the network sees.
//
// A position is fed to the network as a stack of n x n binary planes, always
// from the point of view of the player about to move: "my stones" and "their
// stones" for the current position and the H - 1 before it (history reveals
// ko and recent fights), then one plane that is all ones if Black is to move
// and one that is all ones if White is. That constant plane also lets a
// zero-padded 3x3 convolution tell the board edge apart from empty points.
//
// Optionally (passPlane, off by default: the paper's input) one more plane
// follows the two colour planes: all ones iff the move that led to the
// position was a pass, else all zeros (and at move 0). A pass otherwise shows
// only as two identical history boards, which the network cannot learn to
// read as "a second pass ends the game". The plane is symmetry invariant.
#pragma once
#include <cstdint>
#include <vector>
#include "mzgo/board.hpp"

namespace mzgo {

class Game;

// 2H + 2: (own, opponent) stones for each of the last H positions, then a
// "black to move" plane and a "white to move" plane (exactly one is all ones).
// With passPlane, 2H + 3: plane 2H + 2 is the "last move was a pass" plane.
int numPlanes(int history, bool passPlane = false);

// stones[k] points to n*n Color bytes for the position k plies ago (k = 0 is
// current). stones.size() may be < H; missing positions are zero planes.
// "Own" means toMove (the side to move NOW) for every history plane.
// Writes numPlanes(H, passPlane) * n * n floats to out ([plane][point]) with symmetry
// `sym` applied: value for point p lands at transformPoint(sym, p, n).
void encodePlanes(const std::vector<const uint8_t*>& stones, int n, Color toMove,
                  int H, int sym, float* out, bool passPlane = false, bool lastMovePass = false);

// history[0] is the current board, history[1] the previous one, ...
void encodeBoards(const std::vector<const Board*>& history, int H, int sym, float* out,
                  bool passPlane = false, bool lastMovePass = false);

// Current position of a game, history taken from the game's earlier boards.
// With passPlane the pass plane is set iff the game's last move was a pass.
void encodeGame(const Game& game, int H, int sym, float* out, bool passPlane = false);

} // namespace mzgo
