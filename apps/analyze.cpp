// analyze: one MuZero search, shown in full.
//
// Plays a move list from the empty board, runs the search on the position
// reached (no root noise) and prints what it found: visits, prior and Q for
// the most visited moves, the search's depth, and the principal variation --
// the line the search believes in, which below the first move exists only in
// the model's imagination (the dynamics g, never the real rules) -- with the
// value along it.
//
//   mzgo_analyze --checkpoint P [--moves "C3 D4 pass ..."] [--simulations 200]
//                [--top 10] [--backend blas|naive] [--komi K]
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include "mzgo/backend.hpp"
#include "mzgo/board.hpp"
#include "mzgo/checkpoint.hpp"
#include "mzgo/game.hpp"
#include "mzgo/mcts.hpp"
#include "mzgo/network.hpp"
#include "mzgo/probes.hpp"

using namespace mzgo;

namespace {

const char* const kUsage =
    "usage: mzgo_analyze --checkpoint P [--moves \"C3 D4 pass ...\"] [--simulations 200]\n"
    "                    [--top 10] [--backend blas|naive] [--komi K]\n"
    "\n"
    "--moves are GTP coordinates (or pass) played from the empty board, Black\n"
    "first; the search runs on the position they reach. --top is the number of\n"
    "root moves listed; --komi defaults to 24.5 on 5x5 and 7.5 otherwise;\n"
    "--backend defaults to blas when built with it.\n";

int parseInt(const std::string& name, const char* text, int lo, int hi) {
    char* end = nullptr;
    const long v = std::strtol(text, &end, 10);
    if (*text == '\0' || *end != '\0' || v < lo || v > hi)
        throw std::invalid_argument("--" + name + ": '" + text + "' is not an integer in " + std::to_string(lo) +
                                    ".." + std::to_string(hi));
    return static_cast<int>(v);
}

float parseFloat(const std::string& name, const char* text) {
    char* end = nullptr;
    const float v = std::strtof(text, &end);
    if (*text == '\0' || *end != '\0') throw std::invalid_argument("--" + name + ": '" + text + "' is not a number");
    return v;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string checkpoint, moves, backendName = blasAvailable() ? "blas" : "naive";
        MctsConfig cfg;
        int top = 10;
        float komi = 0;
        bool komiSet = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                std::printf("%s", kUsage);
                return 0;
            }
            if (arg.compare(0, 2, "--") != 0 || i + 1 >= argc) throw std::invalid_argument("bad argument '" + arg + "'");
            const std::string name = arg.substr(2);
            const char* value = argv[++i];
            if (name == "checkpoint") checkpoint = value;
            else if (name == "moves") moves = value;
            else if (name == "simulations") cfg.simulations = parseInt(name, value, 1, 1000000);
            else if (name == "top") top = parseInt(name, value, 1, 1000);
            else if (name == "backend") backendName = value;
            else if (name == "komi") komi = parseFloat(name, value), komiSet = true;
            else throw std::invalid_argument("unknown option '" + arg + "'");
        }
        if (checkpoint.empty()) throw std::invalid_argument("--checkpoint is required");

        const std::unique_ptr<Backend> backend = makeBackend(backendName);
        const MuZeroNetwork net = loadNetwork(checkpoint);
        const NetworkModel model(net, *backend);
        const int n = net.config().boardSize;
        if (!komiSet) komi = n == 5 ? 24.5f : 7.5f;

        Game game(n, komi, defaultMaxMoves(n));
        std::istringstream words(moves);
        for (std::string word; words >> word;) {
            const int action = parseAction(word, n);
            if (action < 0) throw std::invalid_argument("--moves: '" + word + "' is not a move on " + std::to_string(n) +
                                                        "x" + std::to_string(n));
            if (!game.isLegal(action))
                throw std::invalid_argument("--moves: " + word + " is illegal after " +
                                            std::to_string(game.moveCount()) + " moves" +
                                            (game.isTerminal() ? " (the game is over)" : ""));
            game.play(action);
        }
        if (game.isTerminal()) throw std::invalid_argument("--moves: the game is over, nothing to search");

        std::printf("%s (%dx%d), backend %s\n\n", checkpoint.c_str(), n, n, backend->name());
        std::printf("%s", analyzePosition(model, game, cfg, top).c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mzgo_analyze: %s\n", e.what());
        return 1;
    }
}
