// match: plays N games between two players and prints the score. Writes no files.
//
//   match --a SPEC --b SPEC --games N --board-size n --komi k
//         [--simulations 200] [--temperature-moves 4] [--workers 0] [--seed S] [--backend blas]
//
// SPEC is ckpt:PATH (a trained network), rollout:SIMS (the rollout MCTS bot)
// or random (player_spec.hpp). A takes Black in the even-numbered games.
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <map>
#include <stdexcept>
#include <string>
#include "mzgo/backend.hpp"
#include "mzgo/match.hpp"
#include "mzgo/parallel.hpp"
#include "mzgo/player_spec.hpp"

using namespace mzgo;

namespace {

const char* const kUsage =
    "usage: match --a SPEC --b SPEC --games N --board-size n --komi k\n"
    "             [--simulations 200] [--temperature-moves 4] [--workers 0] [--seed S] [--backend blas]\n"
    "\n"
    "SPEC: ckpt:PATH (a checkpoint, searching --simulations per move with the\n"
    "first --temperature-moves moves sampled in proportion to visits),\n"
    "rollout:SIMS (rollout MCTS with SIMS simulations) or random.\n"
    "Colors alternate: A is Black in games 0, 2, 4, ... --workers 0 = all cores.\n"
    "Prints 'A wins / B wins / draws, scoreA'; writes no files.\n";

int usageError(const std::string& message) {
    std::fprintf(stderr, "error: %s\n\n%s", message.c_str(), kUsage);
    return 1;
}

long parseLong(const std::string& name, const std::string& text, long lo, long hi) {
    char* end = nullptr;
    errno = 0;
    const long v = std::strtol(text.c_str(), &end, 10);
    if (text.empty() || *end != '\0' || errno == ERANGE || v < lo || v > hi)
        throw std::invalid_argument("--" + name + ": '" + text + "' is not an integer in " + std::to_string(lo) +
                                    ".." + std::to_string(hi));
    return v;
}

uint64_t parseU64(const std::string& name, const std::string& text) {
    char* end = nullptr;
    errno = 0;
    const unsigned long long v = std::strtoull(text.c_str(), &end, 10);
    if (text.empty() || text[0] == '-' || *end != '\0' || errno == ERANGE)
        throw std::invalid_argument("--" + name + ": '" + text + "' is not an unsigned integer");
    return v;
}

float parseFloat(const std::string& name, const std::string& text) {
    char* end = nullptr;
    errno = 0;
    const float v = std::strtof(text.c_str(), &end);
    if (text.empty() || *end != '\0' || errno == ERANGE)
        throw std::invalid_argument("--" + name + ": '" + text + "' is not a number");
    return v;
}

} // namespace

int main(int argc, char** argv) {
    const char* const known[] = {"a", "b", "games", "board-size", "komi", "simulations", "temperature-moves",
                                 "workers", "seed", "backend"};
    std::map<std::string, std::string> opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::fputs(kUsage, stdout);
            return 0;
        }
        bool isKnown = false;
        for (const char* k : known) isKnown = isKnown || arg == std::string("--") + k;
        if (!isKnown) return usageError("unknown argument '" + arg + "'");
        if (i + 1 >= argc) return usageError(arg + " requires a value");
        const std::string name = arg.substr(2);
        if (opt.count(name)) return usageError(arg + " given more than once");
        opt[name] = argv[++i];
    }
    for (const char* required : {"a", "b", "games", "board-size", "komi"})
        if (!opt.count(required)) return usageError(std::string("--") + required + " is required");

    try {
        auto get = [&](const std::string& name, const std::string& def) {
            return opt.count(name) ? opt[name] : def;
        };
        const int games = static_cast<int>(parseLong("games", opt["games"], 1, 1000000));
        const int n = static_cast<int>(parseLong("board-size", opt["board-size"], 5, 19));
        const float komi = parseFloat("komi", opt["komi"]);
        PlayerSpecOptions spec;
        spec.boardSize = n;
        spec.simulations = static_cast<int>(parseLong("simulations", get("simulations", "200"), 1, 1000000));
        spec.temperatureMoves =
            static_cast<int>(parseLong("temperature-moves", get("temperature-moves", "4"), 0, 1000000));
        spec.backend = get("backend", blasAvailable() ? "blas" : "naive");
        const int workers = static_cast<int>(parseLong("workers", get("workers", "0"), 0, 4096));
        const uint64_t seed = parseU64("seed", get("seed", "1"));

        const PlayerFactory a = makePlayerFactory(opt["a"], spec);
        const PlayerFactory b = makePlayerFactory(opt["b"], spec);
        const MatchResult r = playMatch(a, b, games, n, komi, 0, seed, resolveWorkers(workers));
        std::printf("A %s: %d wins / B %s: %d wins / %d draws, scoreA %.4f\n", opt["a"].c_str(), r.winsA,
                    opt["b"].c_str(), r.winsB, r.draws, r.scoreA());
        return 0;
    } catch (const std::invalid_argument& e) {
        return usageError(e.what());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
