// train: runs MuZero's generation loop (trainer.hpp) for a new or a resumed run.
//
//   train --config configs/5x5.cfg --run runs/name [--set key=value]...
//   train --resume runs/name [--set key=value]...
#include <cstdio>
#include <exception>
#include <iostream>
#include <string>
#include <vector>
#include "mzgo/config.hpp"
#include "mzgo/trainer.hpp"

using namespace mzgo;

namespace {

const char* const kUsage =
    "usage:\n"
    "  train --config FILE --run DIR [--set key=value]...\n"
    "  train --resume DIR [--set key=value]...\n"
    "\n"
    "A new run needs an empty or missing DIR; it saves FILE plus the overrides\n"
    "as DIR/config.cfg. --resume continues DIR from its state.bin; overrides may\n"
    "change any key but the architecture keys (board_size, history, repr_blocks,\n"
    "dyn_blocks, filters, value_hidden, reward_head, pass_plane, unroll_steps) and\n"
    "league_size. Training stops at the config's `generations`, e.g.\n"
    "--set generations=80 to go on.\n"
    "\n"
    "League play (off by default): --set league_size=3 (slots, 0..8) plays\n"
    "league_fraction (0.5) of each generation's games against the slots'\n"
    "members, all the random bot at first; a member the network scores\n"
    "league_replace_score (0.6) against over league_min_games (20) games is\n"
    "replaced by it, one per generation; the random bot only after two such\n"
    "generations in a row. --set league_value=false keeps the league games from\n"
    "training the value (policy and reward/dynamics only); it may change on resume.\n";

int usageError(const std::string& message) {
    std::fprintf(stderr, "error: %s\n\n%s", message.c_str(), kUsage);
    return 1;
}

} // namespace

int main(int argc, char** argv) {
    std::string configFile, runDir, resumeDir;
    std::vector<std::string> sets;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::fputs(kUsage, stdout);
            return 0;
        }
        if (arg != "--config" && arg != "--run" && arg != "--resume" && arg != "--set")
            return usageError("unknown argument '" + arg + "'");
        if (i + 1 >= argc) return usageError(arg + " requires a value");
        const std::string value = argv[++i];
        if (arg == "--set") {
            sets.push_back(value);
        } else {
            std::string* target = arg == "--config" ? &configFile : arg == "--run" ? &runDir : &resumeDir;
            if (!target->empty()) return usageError(arg + " given more than once");
            *target = value;
        }
    }
    const bool resume = !resumeDir.empty();
    if (resume && (!configFile.empty() || !runDir.empty()))
        return usageError("--resume DIR cannot be combined with --config or --run (the run's config.cfg is used)");
    if (!resume && (configFile.empty() || runDir.empty()))
        return usageError("a new run needs --config FILE and --run DIR (or --resume DIR to continue one)");

    try {
        Trainer trainer = [&] {
            if (resume) return Trainer::resume(resumeDir, sets);
            RunConfig cfg = parseConfigFile(configFile);
            for (const std::string& s : sets) applyOverride(cfg, s);
            return Trainer::startNew(cfg, runDir);
        }();
        std::cout << (resume ? "resuming " : "starting ") << trainer.runDir() << " at generation "
                  << trainer.generation() << " of " << trainer.config().generations << std::endl;
        trainer.setLog(&std::cout);
        trainer.run();
        std::cout << "done: generation " << trainer.generation() << " in " << trainer.runDir() << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
