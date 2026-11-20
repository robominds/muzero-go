// rules_probe: has the learned model picked up the rules of Go?
//
// MuZero is never told the rules below the root of its search: there every
// action is allowed, and f's policy is the only thing steering away from
// nonsense. Along real games (rollout-bot games), this measures how much of
// f's policy mass falls on moves that are illegal in the real position --
// at k = 0 (a fresh h of the real position) and on the state g imagines k
// steps ahead along the moves actually played:
//   illegal   every move Game::isLegal rejects (occupied, suicide, superko)
//   occupied  points holding a stone
//   superko   empty, locally legal points that positional superko forbids
//   captured  points that held a stone at the start and were freed by a
//             capture within the k steps (playable again: a model that only
//             remembers "a stone was put there" gets these wrong)
//
//   mzgo_rules_probe --checkpoint P [--games 100] [--k 5] [--backend blas|naive]
//                    [--seed 1] [--workers 0] [--komi K]
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

using namespace mzgo;

namespace {

const char* const kUsage =
    "usage: mzgo_rules_probe --checkpoint P [--games 100] [--k 5] [--backend blas|naive]\n"
    "                        [--seed 1] [--workers 0] [--komi K]\n"
    "\n"
    "--games rollout-bot games to probe along (generated on --workers threads,\n"
    "0 = all cores); --k the deepest imagined step; --komi defaults to 24.5 on\n"
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
        int games = 100, maxK = 5, workers = 0;
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
        for (const GameRecord& r : records) positions += r.length();
        const std::vector<RulesRow> rows = rulesProbe(model, records, maxK);

        std::printf("Policy mass on illegal moves: %s (%dx%d), backend %s\n", checkpoint.c_str(), n, n,
                    backend->name());
        std::printf("%d rollout-bot games, %ld positions, komi %.1f, seed %llu\n", games, positions, komi,
                    static_cast<unsigned long long>(seed));
        std::printf("softmax(f) on the state imagined k steps ahead, judged by the real\n");
        std::printf("position reached (k = 0: fresh h of the real position).\n\n");
        std::printf("   k   samples    illegal   occupied    superko   captured\n");
        for (const RulesRow& r : rows) {
            if (r.samples == 0) continue;
            std::printf("  %2d   %7ld   %8.4f   %8.4f   %8.4f   %8.4f\n", r.k, r.samples, r.illegalMass,
                        r.occupiedMass, r.koSuperkoMass, r.capturedPointMass);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mzgo_rules_probe: %s\n", e.what());
        return 1;
    }
}
