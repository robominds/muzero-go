#include "mzgo/config.hpp"

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace mzgo {

namespace {

enum class Kind { U64, Int, Float, Bool, Str };

// One row per config key: where the value lives inside a RunConfig. A single
// table drives parsing, printing and overrides so the three cannot drift apart.
struct Field {
    const char* key;
    Kind kind;
    void* ptr;
};

std::vector<Field> fields(RunConfig& c) {
    return {
        {"seed", Kind::U64, &c.seed},
        {"generations", Kind::Int, &c.generations},
        {"workers", Kind::Int, &c.workers},
        {"train_threads", Kind::Int, &c.trainThreads},
        {"backend", Kind::Str, &c.backend},
        {"board_size", Kind::Int, &c.boardSize},
        {"komi", Kind::Float, &c.komi},
        {"max_moves", Kind::Int, &c.maxMoves},
        {"history", Kind::Int, &c.history},
        {"repr_blocks", Kind::Int, &c.reprBlocks},
        {"dyn_blocks", Kind::Int, &c.dynBlocks},
        {"filters", Kind::Int, &c.filters},
        {"value_hidden", Kind::Int, &c.valueHidden},
        {"reward_head", Kind::Bool, &c.rewardHead},
        {"pass_plane", Kind::Bool, &c.passPlane},
        {"games_per_generation", Kind::Int, &c.gamesPerGeneration},
        {"simulations", Kind::Int, &c.simulations},
        {"selfplay_group", Kind::Int, &c.selfplayGroup},
        {"dirichlet_scale", Kind::Float, &c.dirichletScale},
        {"noise_fraction", Kind::Float, &c.noiseFraction},
        {"temperature_moves", Kind::Int, &c.temperatureMoves},
        {"replay_games", Kind::Int, &c.replayGames},
        {"train_steps", Kind::Int, &c.trainSteps},
        {"batch_size", Kind::Int, &c.batchSize},
        {"unroll_steps", Kind::Int, &c.unrollSteps},
        {"lr_init", Kind::Float, &c.lrInit},
        {"lr_decay_rate", Kind::Float, &c.lrDecayRate},
        {"lr_decay_steps", Kind::Int, &c.lrDecaySteps},
        {"momentum", Kind::Float, &c.momentum},
        {"weight_decay", Kind::Float, &c.weightDecay},
        {"symmetry_augment", Kind::Bool, &c.symmetryAugment},
        {"bn_calibrate", Kind::Bool, &c.bnCalibrate},
        {"arena_every", Kind::Int, &c.arenaEvery},
        {"arena_games", Kind::Int, &c.arenaGames},
        {"arena_gap", Kind::Int, &c.arenaGap},
        {"arena_simulations", Kind::Int, &c.arenaSimulations},
        {"arena_temperature_moves", Kind::Int, &c.arenaTemperatureMoves},
        {"rollout_bot_simulations", Kind::Int, &c.rolloutBotSimulations},
        {"sgf_per_generation", Kind::Int, &c.sgfPerGeneration},
    };
}

[[noreturn]] void bad(const std::string& key, const std::string& why) {
    throw std::invalid_argument(key + ": " + why);
}

std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

void setField(const Field& f, const std::string& value) {
    const std::string key = f.key;
    if (value.empty()) bad(key, "empty value");
    const std::string malformed = "malformed value '" + value + "'";
    const char* s = value.c_str();
    char* end = nullptr;
    errno = 0;
    switch (f.kind) {
        case Kind::U64: {
            if (value[0] == '-') bad(key, malformed);
            unsigned long long v = std::strtoull(s, &end, 10);
            if (*end != '\0' || errno == ERANGE) bad(key, malformed);
            *static_cast<uint64_t*>(f.ptr) = v;
            break;
        }
        case Kind::Int: {
            long v = std::strtol(s, &end, 10);
            if (*end != '\0' || errno == ERANGE || v < INT_MIN || v > INT_MAX) bad(key, malformed);
            *static_cast<int*>(f.ptr) = static_cast<int>(v);
            break;
        }
        case Kind::Float: {
            float v = std::strtof(s, &end);
            if (*end != '\0' || errno == ERANGE) bad(key, malformed);
            *static_cast<float*>(f.ptr) = v;
            break;
        }
        case Kind::Bool: {
            if (value == "true" || value == "1") *static_cast<bool*>(f.ptr) = true;
            else if (value == "false" || value == "0") *static_cast<bool*>(f.ptr) = false;
            else bad(key, malformed + " (expected true or false)");
            break;
        }
        case Kind::Str:
            *static_cast<std::string*>(f.ptr) = value;
            break;
    }
}

std::string formatField(const Field& f) {
    char buf[64];
    switch (f.kind) {
        case Kind::U64:
            std::snprintf(buf, sizeof buf, "%llu", static_cast<unsigned long long>(*static_cast<uint64_t*>(f.ptr)));
            return buf;
        case Kind::Int:
            std::snprintf(buf, sizeof buf, "%d", *static_cast<int*>(f.ptr));
            return buf;
        case Kind::Float:
            // 9 significant digits are enough to recover any float exactly.
            std::snprintf(buf, sizeof buf, "%.9g", static_cast<double>(*static_cast<float*>(f.ptr)));
            return buf;
        case Kind::Bool:
            return *static_cast<bool*>(f.ptr) ? "true" : "false";
        case Kind::Str:
            return *static_cast<std::string*>(f.ptr);
    }
    return "";
}

}  // namespace

int RunConfig::resolvedMaxMoves() const {
    return maxMoves > 0 ? maxMoves : 2 * boardSize * boardSize;
}

void applyOverride(RunConfig& cfg, const std::string& keyEqualsValue) {
    size_t eq = keyEqualsValue.find('=');
    if (eq == std::string::npos) bad(keyEqualsValue, "expected key=value");
    const std::string key = trim(keyEqualsValue.substr(0, eq));
    const std::string value = trim(keyEqualsValue.substr(eq + 1));
    for (const Field& f : fields(cfg)) {
        if (key == f.key) { setField(f, value); return; }
    }
    bad(key, "unknown key");
}

RunConfig parseConfigString(const std::string& text) {
    RunConfig cfg;
    std::istringstream in(text);
    std::string line;
    int lineNo = 0;
    while (std::getline(in, line)) {
        ++lineNo;
        size_t hash = line.find('#');
        if (hash != std::string::npos) line.resize(hash);
        line = trim(line);
        if (line.empty()) continue;
        if (line.find('=') == std::string::npos)
            bad("line " + std::to_string(lineNo), "expected 'key = value', got '" + line + "'");
        applyOverride(cfg, line);
    }
    return cfg;
}

RunConfig parseConfigFile(const std::string& path) {
    std::ifstream in(path);
    if (!in) bad(path, "cannot open config file");
    std::stringstream ss;
    ss << in.rdbuf();
    return parseConfigString(ss.str());
}

std::string configToString(const RunConfig& cfg) {
    RunConfig copy = cfg;  // fields() hands out mutable pointers; we only read
    std::string out;
    for (const Field& f : fields(copy)) out += std::string(f.key) + " = " + formatField(f) + "\n";
    return out;
}

std::vector<std::string> architectureKeys() {
    return {"board_size", "history", "repr_blocks", "dyn_blocks", "filters", "value_hidden", "reward_head",
            "pass_plane", "unroll_steps"};
}

void validate(const RunConfig& c) {
    if (c.boardSize < 5 || c.boardSize > 19) bad("board_size", "must be in 5..19");
    if (c.history < 1) bad("history", "must be >= 1");
    if (c.reprBlocks < 0) bad("repr_blocks", "must be >= 0");
    if (c.dynBlocks < 0) bad("dyn_blocks", "must be >= 0");
    if (c.filters < 1) bad("filters", "must be >= 1");
    if (c.valueHidden < 1) bad("value_hidden", "must be >= 1");
    if (c.unrollSteps < 1 || c.unrollSteps > 64) bad("unroll_steps", "must be in 1..64");   // checkpoint.cpp's limit
    if (c.batchSize < 1) bad("batch_size", "must be >= 1");
    if (c.simulations < 1) bad("simulations", "must be >= 1");
    if (c.arenaSimulations < 1) bad("arena_simulations", "must be >= 1");
    if (c.gamesPerGeneration < 1) bad("games_per_generation", "must be >= 1");
    if (c.replayGames < 1) bad("replay_games", "must be >= 1");
    if (c.generations < 1) bad("generations", "must be >= 1");
    if (!(c.lrInit > 0)) bad("lr_init", "must be > 0");
    if (!(c.noiseFraction >= 0 && c.noiseFraction <= 1)) bad("noise_fraction", "must be in 0..1");
    // Ranges the trainer relies on.
    if (c.lrDecaySteps < 1) bad("lr_decay_steps", "must be >= 1");
    if (c.trainSteps < 0) bad("train_steps", "must be >= 0");
    if (c.workers < 0) bad("workers", "must be >= 0 (0 = all cores)");
    if (c.trainThreads < 0) bad("train_threads", "must be >= 0 (0 = all cores)");
    if (c.arenaGames < 0 || c.arenaGames % 2 != 0)
        bad("arena_games", "must be even and >= 0 (each side plays Black in half the games)");
    if (c.arenaEvery < 0) bad("arena_every", "must be >= 0 (0 = no arena)");
    if (c.arenaGap < 1) bad("arena_gap", "must be >= 1");
    if (c.arenaTemperatureMoves < 0) bad("arena_temperature_moves", "must be >= 0");
    if (c.rolloutBotSimulations < 1) bad("rollout_bot_simulations", "must be >= 1");
    if (c.temperatureMoves < 0) bad("temperature_moves", "must be >= 0");
    if (c.selfplayGroup < 1) bad("selfplay_group", "must be >= 1");
    if (c.maxMoves < 0) bad("max_moves", "must be >= 0 (0 = 2 * n * n)");
    if (c.sgfPerGeneration < 0) bad("sgf_per_generation", "must be >= 0");
}

}  // namespace mzgo
