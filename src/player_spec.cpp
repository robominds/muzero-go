#include "mzgo/player_spec.hpp"
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <utility>
#include "mzgo/backend.hpp"
#include "mzgo/checkpoint.hpp"
#include "mzgo/network.hpp"
#include "mzgo/selfplay.hpp"

namespace mzgo {

namespace {

// What a ckpt: factory keeps alive for its players: the network, the backend
// and the Model over both. The players hold references into it.
struct LoadedNetwork {
    LoadedNetwork(MuZeroNetwork n, const std::string& backendName)
        : net(std::move(n)), backend(makeBackend(backendName)), model(net, *backend) {}
    MuZeroNetwork net;
    std::unique_ptr<Backend> backend;
    NetworkModel model;
};

bool startsWith(const std::string& s, const std::string& prefix) {
    return s.compare(0, prefix.size(), prefix) == 0;
}

[[noreturn]] void badSpec(const std::string& spec, const std::string& why) {
    throw std::invalid_argument("player spec '" + spec + "': " + why);
}

} // namespace

PlayerFactory makePlayerFactory(const std::string& spec, const PlayerSpecOptions& options) {
    if (spec == "random") return [] { return std::make_unique<RandomPlayer>(); };

    if (startsWith(spec, "rollout:")) {
        const std::string text = spec.substr(8);
        char* end = nullptr;
        errno = 0;
        const long sims = std::strtol(text.c_str(), &end, 10);
        if (text.empty() || *end != '\0' || errno == ERANGE || sims < 1 || sims > INT_MAX)
            badSpec(spec, "expected rollout:SIMS with SIMS a positive integer");
        return [sims] { return std::make_unique<RolloutMctsPlayer>(static_cast<int>(sims)); };
    }

    if (startsWith(spec, "ckpt:")) {
        const std::string path = spec.substr(5);
        if (path.empty()) badSpec(spec, "expected ckpt:PATH");
        auto loaded = std::make_shared<LoadedNetwork>(loadNetwork(path), options.backend);
        const int n = loaded->net.config().boardSize;
        if (n != options.boardSize)
            badSpec(spec, "the network is for " + std::to_string(n) + "x" + std::to_string(n) + ", the match is " +
                              std::to_string(options.boardSize) + "x" + std::to_string(options.boardSize));
        MctsConfig mcts;
        mcts.simulations = options.simulations;
        mcts.addNoise = false;
        const int temperatureMoves = options.temperatureMoves;
        return [loaded, mcts, temperatureMoves, spec] {
            return std::make_unique<MuZeroPlayer>(loaded->model, mcts, temperatureMoves, spec);
        };
    }

    badSpec(spec, "expected ckpt:PATH, rollout:SIMS or random");
}

} // namespace mzgo
