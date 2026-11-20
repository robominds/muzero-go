// Bots and the Player interface.
//
// A Player picks a move for the side to move. Everything that plays games
// (arena matches, the GTP engine, evaluation against yardsticks) talks to this
// one small interface, so a learned MuZero player and a hand-written bot are
// interchangeable. The bots here are yardsticks: RandomPlayer is the floor,
// and RolloutMctsPlayer is classic Monte-Carlo tree search that needs no
// neural network, only the rules: it judges a move by playing many random
// games to the end and counting who wins.
#pragma once
#include <string>
#include "mzgo/game.hpp"
#include "mzgo/rng.hpp"

namespace mzgo {

class Player {
public:
    virtual ~Player() = default;
    virtual std::string name() const = 0;
    virtual void newGame() {}
    // Returns a legal action for game.board().toMove(); the game is not terminal.
    virtual int selectMove(const Game& game, Rng& rng) = 0;
};

// True if playing `point` would fill an eye of the side to move: the point is
// empty, every on-board neighbour is a stone of the side to move, and none of
// those groups is in atari (each has more than one liberty). Filling such an
// eye throws away a liberty for nothing, so random bots avoid it; without
// this rule random games never end sensibly because stones fill their own eyes.
bool fillsOwnEye(const Board& b, int point);

// Uniform over legal non-pass moves that do not fill an own eye; passes only
// when there is no other move.
class RandomPlayer : public Player {
public:
    std::string name() const override { return "random"; }
    int selectMove(const Game& game, Rng& rng) override;
};

// UCT (upper confidence bounds applied to trees) with random playouts on the
// real rules. Each simulation walks down the tree choosing the child with the
// best mean win rate plus an exploration bonus c * sqrt(ln(parent visits) /
// visits), c = sqrt(2); adds one new child; plays a random game (RandomPlayer's
// policy, capped at the game's move limit) from there; and backs the result
// up. The move played is the most visited root child.
class RolloutMctsPlayer : public Player {
public:
    explicit RolloutMctsPlayer(int simulations);
    std::string name() const override;
    int selectMove(const Game& game, Rng& rng) override;
private:
    int simulations_;
};

} // namespace mzgo
