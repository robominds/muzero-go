// gtp: a checkpoint as a Go Text Protocol engine, for evaluation only. A GTP
// controller (tools/gtp_match.py, gogui-twogtp) or a board program such as
// Sabaki starts it and talks to it on stdin/stdout, so the network can play an
// outside reference such as GNU Go. Every move is an MCTS search at
// temperature 0 without root noise. stdout carries only GTP; the startup line,
// one line per genmove (move, its visits, the root value) and errors go to
// stderr.
//
//   mzgo_gtp --checkpoint runs/nine/checkpoints/gen_0040.mzn
//   mzgo_gtp --checkpoint ckpt.mzn --simulations 800 --backend blas --komi 7.5
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include "mzgo/backend.hpp"
#include "mzgo/checkpoint.hpp"
#include "mzgo/gtp.hpp"
#include "mzgo/network.hpp"
#include "mzgo/selfplay.hpp"

using namespace mzgo;

namespace {

const char* const kUsage =
    "usage: mzgo_gtp --checkpoint CKPT [--simulations 200] [--backend blas|naive] [--seed 1]\n"
    "                [--komi 7.5]\n"
    "\n"
    "Speaks GTP version 2 on stdin/stdout: protocol_version, name, version,\n"
    "known_command, list_commands, quit, boardsize (the checkpoint's size only),\n"
    "clear_board, komi, play, genmove, undo, showboard and final_score.\n"
    "genmove searches --simulations simulations at temperature 0 with no root noise\n"
    "and never resigns. Positional superko, no suicide; after two passes in a row\n"
    "the game is over. --backend defaults to blas when built with it. --komi holds\n"
    "until the controller sends komi.\n";

std::string baseName(const std::string& path) {
    const size_t slash = path.find_last_of('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

double parseNumber(const std::string& name, const char* text, double lo, double hi) {
    char* end = nullptr;
    const double v = std::strtod(text, &end);
    if (*text == '\0' || *end != '\0' || !(v >= lo && v <= hi))
        throw std::invalid_argument("--" + name + ": '" + text + "' is not a number in " + std::to_string(lo) + ".." +
                                    std::to_string(hi));
    return v;
}

// Wraps the MuZeroPlayer so the engine (which sees only Player) can still log
// the chosen move's visits and the root value, read from lastSearch().
class ReportingPlayer : public Player {
public:
    ReportingPlayer(const Model& model, const MctsConfig& cfg, std::string name) : inner_(model, cfg, 0, name) {}
    std::string name() const override { return inner_.name(); }
    void newGame() override { inner_.newGame(); }
    int selectMove(const Game& game, Rng& rng) override { return inner_.selectMove(game, rng); }
    std::string detail(int action) const {
        const SearchResult r = inner_.lastSearch();
        int total = 0;
        for (int v : r.visits) total += v;
        char buf[128];
        std::snprintf(buf, sizeof buf, ", %d of %d root visits, value %+.2f (side to move)",
                      r.visits[static_cast<size_t>(action)], total, r.rootValue);
        return buf;
    }
private:
    MuZeroPlayer inner_;
};

} // namespace

int main(int argc, char** argv) {
    try {
        std::string checkpoint, backendName = blasAvailable() ? "blas" : "naive";
        int simulations = 200;
        long seed = 1;
        double komi = 7.5;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                std::fputs(kUsage, stdout);
                return 0;
            }
            if (arg.compare(0, 2, "--") != 0 || i + 1 >= argc) throw std::invalid_argument("bad argument '" + arg + "'");
            const std::string name = arg.substr(2);
            const char* value = argv[++i];
            if (name == "checkpoint") checkpoint = value;
            else if (name == "simulations") simulations = static_cast<int>(parseNumber(name, value, 1, 100000));
            else if (name == "backend") backendName = value;
            else if (name == "seed") seed = static_cast<long>(parseNumber(name, value, 0, 2147483647.0));
            else if (name == "komi") komi = parseNumber(name, value, -1000, 1000);
            else throw std::invalid_argument("unknown option '" + arg + "'");
        }
        if (checkpoint.empty()) throw std::invalid_argument("--checkpoint CKPT is required");

        const std::unique_ptr<Backend> backend = makeBackend(backendName);
        const MuZeroNetwork net = loadNetwork(checkpoint);
        const int n = net.config().boardSize;
        const NetworkModel model(net, *backend);
        MctsConfig cfg;
        cfg.simulations = simulations;
        cfg.addNoise = false;

        auto player = std::make_unique<ReportingPlayer>(
            model, cfg, baseName(checkpoint) + ", " + std::to_string(simulations) + " simulations");
        const ReportingPlayer* reporter = player.get();
        GtpEngine engine(std::move(player), n, static_cast<float>(komi), static_cast<uint64_t>(seed), nullptr,
                         [reporter](int action) { return reporter->detail(action); });
        std::fprintf(stderr, "gtp: %s (%dx%d), %d simulations per move, komi %g, backend %s; reading GTP on stdin\n",
                     checkpoint.c_str(), n, n, simulations, komi, backend->name());

        std::string line;
        while (!engine.quitRequested() && std::getline(std::cin, line)) {
            const std::string response = engine.handle(line);
            if (response.empty()) continue;
            std::fputs(response.c_str(), stdout);
            std::fflush(stdout);   // the controller waits for each answer
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n\n%s", e.what(), kUsage);
        return 1;
    }
}
