// latent_probe: does the learned dynamics track the real game?
//
// Along real games (rollout-bot games, or with --replay the self-play games
// a run trained on), roll the dynamics g
// forward k steps from h(position t) along the moves actually played, and
// compare what f says on that imagined state with f on a fresh h of the real
// position t + k: mean |value difference| and policy total-variation
// distance per k. k = 0 compares h with itself (0 by construction). A
// faithful model stays close as k grows; how fast it drifts is the point.
// Both values are also compared with the game's result (the value target),
// to tell which one is wrong when they disagree.
// AlphaZero has no such diagnostic: its search steps the real rules.
//
//   mzgo_latent_probe --checkpoint P [--games 100] [--k 5] [--backend blas|naive]
//                     [--seed 1] [--workers 0] [--komi K] [--replay STATE_BIN]
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
#include "mzgo/replay_buffer.hpp"
#include "mzgo/trainer.hpp"

using namespace mzgo;

namespace {

const char* const kUsage =
    "usage: mzgo_latent_probe --checkpoint P [--games 100] [--k 5] [--backend blas|naive]\n"
    "                         [--seed 1] [--workers 0] [--komi K] [--replay STATE_BIN]\n"
    "\n"
    "--games rollout-bot games to probe along (generated on --workers threads,\n"
    "0 = all cores); --k the deepest imagined step; --komi defaults to 24.5 on\n"
    "5x5 and 7.5 otherwise; --backend defaults to blas when built with it.\n"
    "--replay: probe along the last --games self-play games (league games,\n"
    "whose values are not trained, are skipped) of a run's state.bin instead;\n"
    "copy the file first if the run is live. --komi and --seed are then unused.\n";

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
        int games = 100, maxK = 5, workers = 0;
        uint64_t seed = 1;
        float komi = 0;
        bool komiSet = false;
        std::string replayPath;
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
            else if (name == "backend") backendName = value;
            else if (name == "seed") seed = static_cast<uint64_t>(parseLong(name, value, 0, 2147483647L));
            else if (name == "workers") workers = static_cast<int>(parseLong(name, value, 0, 1024));
            else if (name == "komi") komi = parseFloat(name, value), komiSet = true;
            else if (name == "replay") replayPath = value;
            else throw std::invalid_argument("unknown option '" + arg + "'");
        }
        if (checkpoint.empty()) throw std::invalid_argument("--checkpoint is required");

        const std::unique_ptr<Backend> backend = makeBackend(backendName);
        const MuZeroNetwork net = loadNetwork(checkpoint);
        const NetworkModel model(net, *backend);
        const int n = net.config().boardSize;
        if (!komiSet) komi = n == 5 ? 24.5f : 7.5f;

        std::vector<GameRecord> records;
        if (replayPath.empty()) {
            records = probeGames(n, komi, games, seed, workers);
        } else {
            const ReplayBuffer replay = readReplayFromState(replayPath);
            for (int i = replay.games() - 1; i >= 0 && static_cast<int>(records.size()) < games; --i)
                if (replay.game(i).trainValue) records.push_back(replay.game(i));
            if (!records.empty() && records[0].boardSize != n)
                throw std::invalid_argument(replayPath + " holds games of another board size");
            komi = records.empty() ? komi : records[0].komi;
        }
        long positions = 0;
        for (const GameRecord& r : records) positions += r.length();
        const std::vector<LatentRow> rows = latentProbe(model, records, maxK);

        std::printf("Learned model vs. real game: %s (%dx%d), backend %s\n", checkpoint.c_str(), n, n, backend->name());
        if (replayPath.empty())
            std::printf("%d rollout-bot games, %ld positions, komi %.1f, seed %llu\n", games, positions, komi,
                        static_cast<unsigned long long>(seed));
        else
            std::printf("%zu self-play games from %s, %ld positions, komi %.1f\n", records.size(), replayPath.c_str(),
                        positions, komi);
        std::printf("f on the state imagined k steps ahead (g along the real moves) vs.\n");
        std::printf("f on a fresh h of the real position reached.\n");
        std::printf("v_i: value on the imagined state; v_f: on the fresh one; z: the game's result,\n");
        std::printf("all from the view of the side to move at t+k.\n\n");
        std::printf("   k   samples   |v_i-v_f|   policy TV   mean v_i   mean v_f   (v_i-z)^2   (v_f-z)^2   |v_i+v_f|\n");
        for (const LatentRow& r : rows) {
            if (r.samples == 0) continue;
            std::printf("  %2d   %7ld   %9.4f   %9.4f   %8.4f   %8.4f   %9.4f   %9.4f   %9.4f\n", r.k, r.samples,
                        r.valueError, r.policyTv, r.imaginedValue, r.freshValue, r.imaginedSqError, r.freshSqError,
                        r.flipError);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mzgo_latent_probe: %s\n", e.what());
        return 1;
    }
}
