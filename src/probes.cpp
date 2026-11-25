#include "mzgo/probes.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <sstream>
#include "mzgo/encode.hpp"
#include "mzgo/match.hpp"
#include "mzgo/parallel.hpp"
#include "mzgo/players.hpp"

namespace mzgo {

namespace {

constexpr int kBatch = 64;          // samples per network call
constexpr int kPvMaxLen = 20;       // plies of principal variation printed by analyzePosition
constexpr int kProbeBatch = 256;    // mini-batch of the board probe's gradient descent

// One imagined rollout: from position t of game `game`, steps k = 0..last.
struct Start { int game, t, last; };

// Every start t with t + k <= T - endMargin for k = 0..maxK: endMargin 1
// stops at the last position where a move is chosen, 0 includes the final board.
std::vector<Start> startsAlong(const std::vector<GameRecord>& games, int maxK, int endMargin) {
    std::vector<Start> starts;
    for (int g = 0; g < static_cast<int>(games.size()); ++g) {
        const int limit = games[g].length() - endMargin;
        for (int t = 0; t <= limit; ++t) starts.push_back({g, t, std::min(maxK, limit - t)});
    }
    return starts;
}

// visit(start, k, out, row): the model's output for s_k of `start` is row
// `row` of `out` (hidden, logits, values).
using Visit = std::function<void(const Start&, int k, const ModelOutput& out, int row)>;

// Runs h on the starts, kBatch at a time, then g along each game's real moves
// for as long as each start's `last` allows, calling visit for every step.
void forEachImagined(const Model& model, const std::vector<GameRecord>& games, const std::vector<Start>& starts,
                     const Visit& visit) {
    const ModelDims dims = model.dims();
    const int history = model.history();
    const bool passPlane = model.passPlane();
    const size_t obsSize = static_cast<size_t>(numPlanes(history, passPlane)) * dims.area();
    const size_t hs = static_cast<size_t>(dims.hiddenSize());
    std::unique_ptr<ModelSession> session = model.newSession(kBatch);
    ModelOutput out;
    std::vector<float> obs, hidden;
    std::vector<int> actions;
    std::vector<const Start*> active, next;   // active[j] is row j of out
    for (size_t first = 0; first < starts.size(); first += kBatch) {
        const int B = static_cast<int>(std::min<size_t>(kBatch, starts.size() - first));
        obs.resize(B * obsSize);
        for (int b = 0; b < B; ++b) {
            const Start& s = starts[first + b];
            encodeRecordPosition(games[s.game], s.t, history, 0, obs.data() + b * obsSize, passPlane);
        }
        session->initialInference(obs.data(), B, out);
        active.clear();
        for (int b = 0; b < B; ++b) active.push_back(&starts[first + b]);
        for (int k = 0;; ++k) {
            next.clear();
            hidden.clear();
            actions.clear();
            for (int j = 0; j < static_cast<int>(active.size()); ++j) {
                const Start& s = *active[j];
                visit(s, k, out, j);
                if (s.last > k) {
                    next.push_back(&s);
                    hidden.insert(hidden.end(), out.hidden.begin() + j * hs, out.hidden.begin() + (j + 1) * hs);
                    actions.push_back(games[s.game].moves[s.t + k]);
                }
            }
            if (next.empty()) break;
            const std::vector<int> depths(next.size(), k + 1);   // every row produces s_{k+1}
            session->recurrentInference(hidden.data(), actions.data(), depths.data(), static_cast<int>(next.size()),
                                        out);
            active.swap(next);
        }
    }
}

// f on a fresh h at every position 0..T-1 of every game. Computed in exactly
// the batches the imagined pass uses for k = 0, so k = 0 compares identical bits.
struct Fresh {
    std::vector<std::vector<float>> probs;    // [game][position * actions], softmax of the logits
    std::vector<std::vector<float>> values;   // [game][position]
};
Fresh freshPredictions(const Model& model, const std::vector<GameRecord>& games) {
    const int A = model.dims().actions();
    Fresh fresh;
    for (const GameRecord& g : games) {
        fresh.probs.emplace_back(static_cast<size_t>(g.length()) * A);
        fresh.values.emplace_back(g.length());
    }
    forEachImagined(model, games, startsAlong(games, 0, 1), [&](const Start& s, int, const ModelOutput& out, int row) {
        softmax(out.logits.data() + static_cast<size_t>(row) * A, A,
                fresh.probs[s.game].data() + static_cast<size_t>(s.t) * A);
        fresh.values[s.game][s.t] = out.values[row];
    });
    return fresh;
}

// The record's moves replayed on its first board (Black to move), with a move
// cap past the record's length so every recorded position is still in play.
Game replay(const GameRecord& r) {
    Board start(r.boardSize);
    for (int i = 0; i < r.area(); ++i) start.setStone(i, static_cast<Color>(r.stones[i]));
    start.setToMove(Color::Black);
    return Game(start, r.komi, r.length() + 1);
}

// Board-probe class of a stone byte at position p: 0 empty, 1 own, 2 opponent.
int stoneClass(const GameRecord& r, int p, int point) {
    const Color c = static_cast<Color>(r.stones[static_cast<size_t>(p) * r.area() + point]);
    return c == Color::Empty ? 0 : (c == r.toMoveAt(p) ? 1 : 2);
}

int probeArgmax(const BoardProbe& probe, int filters, int area, const float* hidden, int point) {
    int best = 0;
    double bestLogit = 0;
    for (int c = 0; c < 3; ++c) {
        double z = probe.b[c];
        for (int f = 0; f < filters; ++f) z += static_cast<double>(probe.w[c * filters + f]) * hidden[f * area + point];
        if (c == 0 || z > bestLogit) best = c, bestLogit = z;
    }
    return best;
}

} // namespace

std::vector<GameRecord> probeGames(int boardSize, float komi, int games, uint64_t seed, int workers) {
    const PlayerFactory bot = [] { return std::unique_ptr<Player>(new RolloutMctsPlayer(64)); };
    const MatchResult match = playMatch(bot, bot, games, boardSize, komi, 0, seed, resolveWorkers(workers));
    std::vector<GameRecord> records;
    for (const GameRecordLite& lite : match.games) {
        Game game(boardSize, komi, defaultMaxMoves(boardSize));
        const int A = boardSize * boardSize + 1;
        std::vector<float> policies(lite.moves.size() * A, 0.0f);
        for (size_t p = 0; p < lite.moves.size(); ++p) {
            game.play(lite.moves[p]);
            policies[p * A + lite.moves[p]] = 1.0f;
        }
        records.push_back(recordFromGame(game, policies, std::vector<float>(lite.moves.size(), 0.0f)));
    }
    return records;
}

std::vector<LatentRow> latentProbe(const Model& model, const std::vector<GameRecord>& games, int maxK) {
    const int A = model.dims().actions();
    const Fresh fresh = freshPredictions(model, games);
    std::vector<LatentRow> rows;
    for (int k = 0; k <= maxK; ++k) rows.push_back({k, 0, 0.0, 0.0});
    std::vector<float> probs(A);
    forEachImagined(model, games, startsAlong(games, maxK, 1), [&](const Start& s, int k, const ModelOutput& out, int row) {
        const int p = s.t + k;
        softmax(out.logits.data() + static_cast<size_t>(row) * A, A, probs.data());
        const float* real = fresh.probs[s.game].data() + static_cast<size_t>(p) * A;
        double l1 = 0;
        for (int a = 0; a < A; ++a) l1 += std::fabs(static_cast<double>(probs[a]) - real[a]);
        LatentRow& r = rows[k];
        const double vi = out.values[row], vf = fresh.values[s.game][p], z = games[s.game].outcomeFor(p);
        r.samples += 1;
        r.valueError += std::fabs(vi - vf);
        r.policyTv += 0.5 * l1;
        r.imaginedValue += vi;
        r.freshValue += vf;
        r.imaginedSqError += (vi - z) * (vi - z);
        r.freshSqError += (vf - z) * (vf - z);
        r.flipError += std::fabs(vi + vf);
    });
    for (LatentRow& r : rows) {
        if (r.samples == 0) continue;
        const double n = static_cast<double>(r.samples);
        r.valueError /= n;
        r.policyTv /= n;
        r.imaginedValue /= n;
        r.freshValue /= n;
        r.imaginedSqError /= n;
        r.freshSqError /= n;
        r.flipError /= n;
    }
    return rows;
}

std::vector<RulesRow> rulesProbe(const Model& model, const std::vector<GameRecord>& games, int maxK) {
    const int A = model.dims().actions();
    enum : uint8_t { kIllegal = 1, kOccupied = 2, kSuperko = 4 };
    // What the real rules say about every action at every position 0..T-1.
    std::vector<std::vector<uint8_t>> kinds(games.size());
    for (size_t g = 0; g < games.size(); ++g) {
        const GameRecord& r = games[g];
        Game game = replay(r);
        kinds[g].assign(static_cast<size_t>(r.length()) * A, 0);
        for (int p = 0; p < r.length(); ++p) {
            uint8_t* row = kinds[g].data() + static_cast<size_t>(p) * A;
            const Board& board = game.board();
            for (int a = 0; a < A; ++a) {
                const bool legal = game.isLegal(a);
                row[a] = legal ? 0 : kIllegal;
                if (a < r.area() && board.at(a) != Color::Empty) row[a] |= kOccupied;
                else if (!legal && board.isLegal(a)) row[a] |= kSuperko;
            }
            game.play(r.moves[p]);
        }
    }
    std::vector<RulesRow> rows;
    for (int k = 0; k <= maxK; ++k) rows.push_back({k, 0, 0.0, 0.0, 0.0, 0.0});
    std::vector<float> probs(A);
    forEachImagined(model, games, startsAlong(games, maxK, 1), [&](const Start& s, int k, const ModelOutput& out, int row) {
        const GameRecord& r = games[s.game];
        const int p = s.t + k, area = r.area();
        softmax(out.logits.data() + static_cast<size_t>(row) * A, A, probs.data());
        const uint8_t* kind = kinds[s.game].data() + static_cast<size_t>(p) * A;
        const uint8_t* before = r.stones.data() + static_cast<size_t>(s.t) * area;
        const uint8_t* now = r.stones.data() + static_cast<size_t>(p) * area;
        RulesRow& acc = rows[k];
        acc.samples += 1;
        for (int a = 0; a < A; ++a) {
            const double m = probs[a];
            if (kind[a] & kIllegal) acc.illegalMass += m;
            if (kind[a] & kOccupied) acc.occupiedMass += m;
            if (kind[a] & kSuperko) acc.koSuperkoMass += m;
            if (a < area && before[a] != static_cast<uint8_t>(Color::Empty) && now[a] == static_cast<uint8_t>(Color::Empty))
                acc.capturedPointMass += m;
        }
    });
    for (RulesRow& r : rows) {
        if (r.samples == 0) continue;
        const double n = static_cast<double>(r.samples);
        r.illegalMass /= n;
        r.occupiedMass /= n;
        r.koSuperkoMass /= n;
        r.capturedPointMass /= n;
    }
    return rows;
}

BoardProbe fitBoardProbe(const Model& model, const std::vector<GameRecord>& games, int epochs, uint64_t seed) {
    const ModelDims dims = model.dims();
    const int F = dims.filters, area = dims.area();
    Rng rng(seed);

    // Positions to fit on: every position 0..T, or a random subset of them.
    std::vector<Start> positions = startsAlong(games, 0, 0);
    if (static_cast<int>(positions.size()) > kBoardProbeFitPositions) {
        for (int i = 0; i < kBoardProbeFitPositions; ++i)
            std::swap(positions[i], positions[i + rng.below(static_cast<int>(positions.size()) - i)]);
        positions.resize(kBoardProbeFitPositions);
    }

    // One sample per (position, point): its F hidden channels and its class.
    const size_t n = positions.size() * area;
    std::vector<float> x(n * F);
    std::vector<uint8_t> y(n);
    size_t next = 0;
    forEachImagined(model, games, positions, [&](const Start& s, int, const ModelOutput& out, int row) {
        const float* h = out.hidden.data() + static_cast<size_t>(row) * dims.hiddenSize();
        for (int i = 0; i < area; ++i, ++next) {
            for (int f = 0; f < F; ++f) x[next * F + f] = h[f * area + i];
            y[next] = static_cast<uint8_t>(stoneClass(games[s.game], s.t, i));
        }
    });
    assert(next == n);

    // Standardize each channel, so one step size suits every network.
    std::vector<double> mean(F, 0.0), sd(F, 0.0);
    for (size_t j = 0; j < n; ++j)
        for (int f = 0; f < F; ++f) mean[f] += x[j * F + f];
    for (int f = 0; f < F; ++f) mean[f] /= std::max<size_t>(n, 1);
    for (size_t j = 0; j < n; ++j)
        for (int f = 0; f < F; ++f) sd[f] += (x[j * F + f] - mean[f]) * (x[j * F + f] - mean[f]);
    for (int f = 0; f < F; ++f) {
        sd[f] = std::sqrt(sd[f] / std::max<size_t>(n, 1));
        if (sd[f] < 1e-6) sd[f] = 1.0;   // a constant channel carries nothing; leave it at 0
    }
    for (size_t j = 0; j < n; ++j)
        for (int f = 0; f < F; ++f) x[j * F + f] = static_cast<float>((x[j * F + f] - mean[f]) / sd[f]);

    // Gradient descent on the mean cross-entropy. Its Hessian's largest
    // eigenvalue is at most 0.5 * (F + 1) on standardized features (0.5 for the
    // softmax, F + 1 bounding the feature covariance plus the bias), so this
    // step is safely below the stability limit 2 / that.
    const double lr = 1.0 / (F + 1);
    std::vector<double> w(3 * F, 0.0), b(3, 0.0), gw(3 * F), gb(3);
    std::vector<size_t> order(n);
    for (size_t j = 0; j < n; ++j) order[j] = j;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t j = n; j > 1; --j) std::swap(order[j - 1], order[rng.below(static_cast<int>(j))]);
        for (size_t first = 0; first < n; first += kProbeBatch) {
            const size_t last = std::min(n, first + kProbeBatch);
            std::fill(gw.begin(), gw.end(), 0.0);
            std::fill(gb.begin(), gb.end(), 0.0);
            for (size_t o = first; o < last; ++o) {
                const float* xi = x.data() + order[o] * F;
                double z[3], zmax = -1e300, sum = 0;
                for (int c = 0; c < 3; ++c) {
                    z[c] = b[c];
                    for (int f = 0; f < F; ++f) z[c] += w[c * F + f] * xi[f];
                    zmax = std::max(zmax, z[c]);
                }
                for (int c = 0; c < 3; ++c) sum += (z[c] = std::exp(z[c] - zmax));
                for (int c = 0; c < 3; ++c) {
                    const double d = z[c] / sum - (y[order[o]] == c ? 1.0 : 0.0);   // dLoss/dlogit
                    gb[c] += d;
                    for (int f = 0; f < F; ++f) gw[c * F + f] += d * xi[f];
                }
            }
            const double scale = lr / static_cast<double>(last - first);
            for (int c = 0; c < 3; ++c) b[c] -= scale * gb[c];
            for (int j = 0; j < 3 * F; ++j) w[j] -= scale * gw[j];
        }
    }

    // Fold the standardization in: w.(x - mean)/sd + b = (w/sd).x + (b - w.mean/sd).
    BoardProbe probe;
    probe.w.resize(3 * F);
    probe.b.resize(3);
    for (int c = 0; c < 3; ++c) {
        double bias = b[c];
        for (int f = 0; f < F; ++f) {
            probe.w[c * F + f] = static_cast<float>(w[c * F + f] / sd[f]);
            bias -= w[c * F + f] * mean[f] / sd[f];
        }
        probe.b[c] = static_cast<float>(bias);
    }
    return probe;
}

std::vector<BoardRow> boardProbe(const Model& model, const BoardProbe& probe, const std::vector<GameRecord>& games,
                                 int maxK) {
    const ModelDims dims = model.dims();
    const int F = dims.filters, area = dims.area();
    assert(probe.w.size() == static_cast<size_t>(3 * F) && probe.b.size() == 3);
    std::vector<BoardRow> rows;
    for (int k = 0; k <= maxK; ++k) rows.push_back({k, 0, 0.0, 0, 0.0});
    forEachImagined(model, games, startsAlong(games, maxK, 0), [&](const Start& s, int k, const ModelOutput& out, int row) {
        const GameRecord& r = games[s.game];
        const int p = s.t + k;
        const float* h = out.hidden.data() + static_cast<size_t>(row) * dims.hiddenSize();
        BoardRow& br = rows[k];
        for (int i = 0; i < area; ++i) {
            const bool right = probeArgmax(probe, F, area, h, i) == stoneClass(r, p, i);
            br.points += 1;
            br.accuracy += right;
            if (r.stones[static_cast<size_t>(s.t) * area + i] != r.stones[static_cast<size_t>(p) * area + i]) {
                br.changedPoints += 1;
                br.changedAccuracy += right;
            }
        }
    });
    for (BoardRow& r : rows) {
        if (r.points > 0) r.accuracy /= static_cast<double>(r.points);
        if (r.changedPoints > 0) r.changedAccuracy /= static_cast<double>(r.changedPoints);
    }
    return rows;
}

std::string analyzePosition(const Model& model, const Game& game, const MctsConfig& cfg, int topMoves) {
    assert(!game.isTerminal());
    MctsConfig quiet = cfg;
    quiet.addNoise = false;
    Search search(quiet, model.dims());
    std::unique_ptr<ModelSession> session = model.newSession(1);
    Rng rng(0);
    const SearchResult result = runSearch(search, *session, game, model.history(), model.passPlane(), rng);
    const int n = game.size();
    const Color toMove = game.board().toMove();
    const char* side = toMove == Color::Black ? "Black" : "White";
    const char* other = toMove == Color::Black ? "White" : "Black";

    std::ostringstream os;
    char line[256];
    os << game.board().toString();   // ends with the side to move
    std::snprintf(line, sizeof line, "%d moves played; %d simulations\n", game.moveCount(), quiet.simulations);
    os << line;
    std::snprintf(line, sizeof line, "root value %+.3f (for %s)\n\n", result.rootValue, side);
    os << line;

    std::vector<RootEdge> edges = search.rootEdges();
    std::stable_sort(edges.begin(), edges.end(), [](const RootEdge& a, const RootEdge& b) {
        return a.visits != b.visits ? a.visits > b.visits : a.prior > b.prior;
    });
    os << "  move  visits   prior       Q\n";
    for (int i = 0; i < std::min(topMoves, static_cast<int>(edges.size())); ++i) {
        const RootEdge& e = edges[i];
        std::snprintf(line, sizeof line, "  %4s  %6d  %6.3f  %+6.3f\n", actionToString(e.action, n).c_str(), e.visits,
                      e.prior, e.q);
        os << line;
    }
    std::snprintf(line, sizeof line,
                  "\nsearch depth: max %d, mean %.2f; simulations going beyond the real game's end (two passes): %.1f%%\n",
                  result.maxDepth, result.meanDepth, 100.0 * result.pastEndFraction);
    os << line;

    // Each PV node's value is from the view of its own side to move, and an
    // edge's reward and Q from the view of the player making that move; all
    // three are signed here for the side to move at the root, so the
    // columns read as one story. The reward column shows where the model
    // puts the game's payoff (on the move that ends it, ideally).
    os << "\nprincipal variation (imagined by the dynamics; reward, Q and value for " << side << "):\n";
    os << "  ply  player  move   reward        Q    value\n";
    const std::vector<PvStep> pv = search.principalVariation(kPvMaxLen);
    for (size_t i = 0; i < pv.size(); ++i) {
        const bool rootSideMoved = i % 2 == 0;
        const float sign = rootSideMoved ? 1.0f : -1.0f;   // mover's view -> root side's view
        std::snprintf(line, sizeof line, "  %3zu  %6s  %4s  %+7.3f  %+7.3f  %+7.3f\n", i + 1,
                      rootSideMoved ? side : other, actionToString(pv[i].action, n).c_str(), sign * pv[i].reward,
                      sign * pv[i].q, -sign * pv[i].value);
        os << line;
    }
    return os.str();
}

} // namespace mzgo
