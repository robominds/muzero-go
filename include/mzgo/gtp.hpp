// The Go Text Protocol (GTP, version 2) is how Go programs talk to each other
// and to graphical boards such as Sabaki: a controller writes one command per
// line ("play b D4", "genmove w") and the engine answers each with
// "= result" or "? error", terminated by a blank line. Speaking it lets a
// trained network play an outside reference program such as GNU Go, so its
// strength can be measured on a scale that is not its own ladder.
//
// GtpEngine is the whole protocol as a library class: handle() takes one input
// line and returns the complete response, so tests drive it without pipes and
// apps/gtp.cpp is only a read-print loop. The engine depends only on Player,
// so a MuZero player and a random bot are interchangeable behind it.
//
// Rules are the project's own (Game): positional superko, suicide illegal,
// Tromp-Taylor area scoring for final_score. The engine keeps the move list
// and replays it from scratch for undo, komi and clear_board. It never
// resigns, sets no move cap (the controller decides), and plays no further
// once both sides have passed in a row: genmove then answers
// "? game is over" and play "? illegal move" until clear_board.
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "mzgo/game.hpp"
#include "mzgo/players.hpp"
#include "mzgo/rng.hpp"

namespace mzgo {

class GtpEngine {
public:
    using Log = std::function<void(const std::string&)>;
    // detail(action) may add text (visits, value) to genmove's log line.
    using Detail = std::function<std::string(int action)>;

    // log defaults to writing the line to std::cerr; never stdout, which carries only GTP.
    GtpEngine(std::unique_ptr<Player> player, int boardSize, float komi, uint64_t seed = 1,
              Log log = nullptr, Detail detail = nullptr);

    // One line of controller input (with or without its trailing newline) in,
    // the full response out: "=[id] result\n\n" or "?[id] message\n\n". A line
    // with no command (empty, blank, or only a # comment) returns "" and must
    // produce no output.
    std::string handle(const std::string& line);

    bool quitRequested() const { return quit_; }
    const Game& game() const { return game_; }

    // Every command handle() knows, in list_commands order.
    static const std::vector<std::string>& commands();

private:
    std::unique_ptr<Player> player_;
    int n_;
    Game game_;
    Rng rng_;
    Log log_;
    Detail detail_;
    // Game moves added by each accepted play/genmove: 1, or 2 when an
    // out-of-turn move first recorded a pass for the side to move.
    std::vector<int> added_;
    bool quit_ = false;

    std::string dispatch(const std::string& command, const std::vector<std::string>& args, bool& ok);
    void restart(float komi, std::vector<int> moves);   // new Game replaying moves
    bool playAs(Color color, int action);                // false (state unchanged) if illegal
};

} // namespace mzgo
