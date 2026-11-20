#include "mzgo/match.hpp"
#include "mzgo/parallel.hpp"

namespace mzgo {

double MatchResult::scoreA() const {
    const int total = winsA + winsB + draws;
    return total == 0 ? 0.0 : (winsA + draws / 2.0) / total;
}

MatchResult playMatch(const PlayerFactory& a, const PlayerFactory& b, int games, int boardSize, float komi,
                      int maxMoves, uint64_t seed, int workers) {
    const int cap = maxMoves == 0 ? defaultMaxMoves(boardSize) : maxMoves;
    MatchResult out;
    out.games.resize(games);
    parallelFor(games, workers, [&](int i, int) {
        auto pa = a();
        auto pb = b();
        pa->newGame();
        pb->newGame();
        Rng rng(deriveSeed(seed, static_cast<uint64_t>(i)));
        const bool aBlack = i % 2 == 0;
        Game g(boardSize, komi, cap);
        while (!g.isTerminal()) {
            const bool blackToMove = g.board().toMove() == Color::Black;
            Player& p = (blackToMove == aBlack) ? *pa : *pb;
            g.play(p.selectMove(g, rng));
        }
        GameRecordLite& rec = out.games[i];
        rec.moves = g.moves();
        rec.resultForBlack = g.resultForBlack();
        rec.aWasBlack = aBlack;
    });
    for (const auto& r : out.games) {
        if (r.resultForBlack == 0) ++out.draws;
        else if ((r.resultForBlack > 0) == r.aWasBlack) ++out.winsA;
        else ++out.winsB;
    }
    return out;
}

} // namespace mzgo
