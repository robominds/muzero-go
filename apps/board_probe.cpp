// board_probe: what board does the model think it is looking at?
//
// MuZero's hidden state is never trained to describe the board, only to
// predict policy, value (and reward). This fits a linear readout (softmax
// regression, the same weights at every point, no gradient into the
// network) from the hidden channels at a point to its content -- empty,
// own stone, opponent stone, "own" being the side to move -- on fresh h
// states of real positions, then applies it to the states g imagines k
// steps ahead along the moves actually played. Accuracy per k, and on the
// points whose content changed between the start and the real position
// reached (new stones, captures): the ones a model of the game must get right.
//
//   mzgo_board_probe --checkpoint P [--games 100] [--k 5] [--epochs 20]
//                    [--backend blas|naive] [--seed 1] [--workers 0] [--komi K]
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "mzgo/backend.hpp"
#include "mzgo/checkpoint.hpp"
#include "mzgo/network.hpp"
#include "mzgo/probes.hpp"
#include "mzgo/rng.hpp"

using namespace mzgo;

namespace {

const char* const kUsage =
    "usage: mzgo_board_probe --checkpoint P [--games 100] [--k 5] [--epochs 20]\n"
    "                        [--backend blas|naive] [--seed 1] [--workers 0] [--komi K]\n"
    "\n"
    "--games rollout-bot games to probe along (generated on --workers threads,\n"
    "0 = all cores); --k the deepest imagined step; --epochs of gradient descent\n"
    "for the readout (fitted on the same games); --komi defaults to 24.5 on\n"
    "5x5 and 7.5 otherwise; --backend defaults to blas when built with it.\n";

long parseLong(const std::string& name, const char* text, long lo, long hi) {
    char* end = nullptr;
    const long v = std::strtol(text, &end, 10);
    if (*text == '\0' || *end != '\0' || v < lo || v > hi)
        throw std::invalid_argument("--" + name + ": '" + text + "' is not an integer in " + std::to_string(lo) +
                                    ".." + std::to_string(hi));
    return v;
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
        std::string checkpoint, backendName = blasAvailable() ? "blas" : "naive";
        int games = 100, maxK = 5, workers = 0, epochs = 20;
        uint64_t seed = 1;
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
            else if (name == "games") games = static_cast<int>(parseLong(name, value, 1, 1000000));
            else if (name == "k") maxK = static_cast<int>(parseLong(name, value, 0, 64));
            else if (name == "epochs") epochs = static_cast<int>(parseLong(name, value, 1, 100000));
            else if (name == "backend") backendName = value;
            else if (name == "seed") seed = static_cast<uint64_t>(parseLong(name, value, 0, 2147483647L));
            else if (name == "workers") workers = static_cast<int>(parseLong(name, value, 0, 1024));
            else if (name == "komi") komi = parseFloat(name, value), komiSet = true;
            else throw std::invalid_argument("unknown option '" + arg + "'");
        }
        if (checkpoint.empty()) throw std::invalid_argument("--checkpoint is required");

        const std::unique_ptr<Backend> backend = makeBackend(backendName);
        const MuZeroNetwork net = loadNetwork(checkpoint);
        const NetworkModel model(net, *backend);
        const int n = net.config().boardSize;
        if (!komiSet) komi = n == 5 ? 24.5f : 7.5f;

        const std::vector<GameRecord> records = probeGames(n, komi, games, seed, workers);
        long positions = 0;
        for (const GameRecord& r : records) positions += r.length() + 1;   // the final board too
        const BoardProbe probe = fitBoardProbe(model, records, epochs, deriveSeed(seed, 1));
        const std::vector<BoardRow> rows = boardProbe(model, probe, records, maxK);

        std::printf("Linear board readout from hidden states: %s (%dx%d), backend %s\n", checkpoint.c_str(), n, n,
                    backend->name());
        std::printf("%d rollout-bot games, %ld positions, komi %.1f, seed %llu, %d epochs\n", games, positions, komi,
                    static_cast<unsigned long long>(seed), epochs);
        std::printf("Readout fitted on fresh h, applied to the state imagined k steps ahead;\n");
        std::printf("\"changed\": points whose content differs between the start and position t + k.\n\n");
        std::printf("   k     points   accuracy    changed   changed acc.\n");
        for (const BoardRow& r : rows) {
            if (r.points == 0) continue;
            if (r.changedPoints == 0)
                std::printf("  %2d   %8ld   %8.4f   %8ld   %12s\n", r.k, r.points, r.accuracy, r.changedPoints, "-");
            else
                std::printf("  %2d   %8ld   %8.4f   %8ld   %12.4f\n", r.k, r.points, r.accuracy, r.changedPoints,
                            r.changedAccuracy);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mzgo_board_probe: %s\n", e.what());
        return 1;
    }
}
