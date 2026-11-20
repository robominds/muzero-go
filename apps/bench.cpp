// bench: what the MuZero network costs, so preset sizes are chosen from
// numbers rather than guesses. Two measurements on a random network:
//   - one training step (h, K steps of g, K + 1 of f, loss, backprop through
//     the unroll) at --batch, on a thread pool of --threads (0 = all cores),
//     as the training phase runs it;
//   - recurrent inference (g then f) at batch --group on one thread, as a
//     self-play worker's search calls it.
//
//   bench [--board-size 9] [--blocks 6] [--filters 64] [--batch 256] [--unroll 5]
//         [--group 8] [--backend naive|blas] [--threads 0] [--seconds 3]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "mzgo/backend.hpp"
#include "mzgo/network.hpp"
#include "mzgo/parallel.hpp"
#include "mzgo/rng.hpp"

using namespace mzgo;

namespace {

const char* const kUsage =
    "usage: bench [--board-size 9] [--blocks 6] [--filters 64] [--batch 256] [--unroll 5]\n"
    "             [--group 8] [--backend naive|blas] [--threads 0] [--seconds 3]\n"
    "\n"
    "--blocks sets both the representation's and the dynamics' residual blocks;\n"
    "--filters also sets the value head's hidden width (as both presets do).\n"
    "--backend defaults to blas when built with it. --threads is the training\n"
    "step's thread pool (0 = all cores); inference runs on one thread.\n"
    "--seconds is the minimum duration of each measurement.\n";

using Clock = std::chrono::steady_clock;

double secondsSince(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// Calls fn until `seconds` have passed (after one warm-up call); returns
// seconds per call.
template <class Fn>
double secondsPerCall(Fn fn, double seconds) {
    fn();
    long long calls = 0;
    const auto start = Clock::now();
    double elapsed = 0;
    do {
        fn();
        ++calls;
        elapsed = secondsSince(start);
    } while (elapsed < seconds);
    return elapsed / static_cast<double>(calls);
}

int parseInt(const std::string& name, const char* text, int lo, int hi) {
    char* end = nullptr;
    const long v = std::strtol(text, &end, 10);
    if (*text == '\0' || *end != '\0' || v < lo || v > hi)
        throw std::invalid_argument("--" + name + ": '" + text + "' is not an integer in " + std::to_string(lo) +
                                    ".." + std::to_string(hi));
    return static_cast<int>(v);
}

// Random planes, actions and targets: the cost does not depend on the values.
TrainBatch randomBatch(const NetConfig& cfg, int B, int K, Rng& rng) {
    TrainBatch t;
    t.resize(B, K, cfg.planes(), cfg.area(), cfg.actions());
    for (float& v : t.observations) v = static_cast<float>(rng.below(2));
    for (int& a : t.actions) a = rng.below(cfg.actions());
    for (float& p : t.policyTargets) p = 1.0f / cfg.actions();
    for (float& m : t.policyMask) m = 1.0f;
    for (float& m : t.valueMask) m = 1.0f;
    for (float& m : t.rewardMask) m = 1.0f;
    for (float& z : t.valueTargets) z = rng.below(2) ? 1.0f : -1.0f;
    return t;
}

} // namespace

int main(int argc, char** argv) {
    try {
        NetConfig cfg;
        int batch = 256, unroll = 5, group = 8, threads = 0;
        double seconds = 3.0;
        std::string backendName = blasAvailable() ? "blas" : "naive";
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                std::printf("%s", kUsage);
                return 0;
            }
            if (arg.compare(0, 2, "--") != 0 || i + 1 >= argc) throw std::invalid_argument("bad argument '" + arg + "'");
            const std::string name = arg.substr(2);
            const char* value = argv[++i];
            if (name == "board-size") cfg.boardSize = parseInt(name, value, 5, 19);
            else if (name == "blocks") cfg.reprBlocks = cfg.dynBlocks = parseInt(name, value, 0, 64);
            else if (name == "filters") cfg.filters = cfg.valueHidden = parseInt(name, value, 1, 1024);
            else if (name == "batch") batch = parseInt(name, value, 2, 4096);
            else if (name == "unroll") unroll = parseInt(name, value, 1, 64);
            else if (name == "group") group = parseInt(name, value, 1, 4096);
            else if (name == "backend") backendName = value;
            else if (name == "threads") threads = parseInt(name, value, 0, 256);
            else if (name == "seconds") seconds = parseInt(name, value, 0, 3600);
            else throw std::invalid_argument("unknown option '" + arg + "'");
        }
        cfg.unrollSteps = unroll;   // the network keeps batch norm statistics for this K
        const std::unique_ptr<Backend> backend = makeBackend(backendName);
        const int trainThreads = resolveTrainThreads(threads);
        std::printf("network: %dx%d, %d+%d blocks, %d filters, value hidden %d; backend %s\n", cfg.boardSize,
                    cfg.boardSize, cfg.reprBlocks, cfg.dynBlocks, cfg.filters, cfg.valueHidden, backend->name());

        MuZeroNetwork net(cfg, 1);
        Rng rng(2);
        const TrainBatch train = randomBatch(cfg, batch, unroll, rng);
        {
            ThreadPool pool(trainThreads);
            TrainWorkspace ws(cfg, batch, unroll, *backend, &pool);
            const double s = secondsPerCall([&] { net.trainStep(ws, train); }, seconds);
            std::printf("training step: batch %d, unroll %d, %d threads: %.1f ms/step\n", batch, unroll, trainThreads,
                        1e3 * s);
        }
        {
            NetworkModel model(net, *backend);
            std::unique_ptr<ModelSession> session = model.newSession(group);
            const TrainBatch obs = randomBatch(cfg, group, 1, rng);
            ModelOutput root, child;
            session->initialInference(obs.observations.data(), group, root);
            std::vector<int> actions(group);
            for (int& a : actions) a = rng.below(cfg.actions());
            const std::vector<int> depths(group, 1);
            const double s = secondsPerCall(
                [&] { session->recurrentInference(root.hidden.data(), actions.data(), depths.data(), group, child); },
                seconds);
            std::printf("recurrent inference: batch %d, 1 thread: %.0f inferences/s (%.3f ms/call)\n", group,
                        group / s, 1e3 * s);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n\n%s", e.what(), kUsage);
        return 1;
    }
}
