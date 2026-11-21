#include "mzgo/league.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>

namespace mzgo {

std::string LeagueMember::name() const {
    return isRandom() ? "random" : "gen_" + std::to_string(generation);
}

double LeagueResult::score() const { return games > 0 ? points / games : 0.0; }

int leagueGameCount(int gamesPerGeneration, float fraction, int size) {
    if (size <= 0) return 0;
    const long n = std::lround(static_cast<double>(gamesPerGeneration) * static_cast<double>(fraction));
    return static_cast<int>(std::clamp<long>(n, 0, gamesPerGeneration));
}

std::vector<LeagueResult> leagueResults(const std::vector<GameRecord>& leagueGames,
                                        const std::vector<int>& opponentOf, int members) {
    assert(leagueGames.size() == opponentOf.size());
    std::vector<LeagueResult> results(static_cast<size_t>(std::max(0, members)));
    for (size_t i = 0; i < leagueGames.size(); ++i) {
        const int m = opponentOf[i];
        assert(m >= 0 && m < members);
        const float black = leagueGames[i].resultForBlack;
        const float learner = i % 2 == 0 ? black : -black;   // the learner is Black in even games
        LeagueResult& r = results[static_cast<size_t>(m)];
        ++r.games;
        r.points += learner > 0 ? 1.0 : learner < 0 ? 0.0 : 0.5;
    }
    return results;
}

int chooseReplacement(const std::vector<LeagueResult>& results, double threshold, int minGames) {
    int best = -1;
    for (size_t i = 0; i < results.size(); ++i) {
        const LeagueResult& r = results[i];
        if (r.games < std::max(1, minGames)) continue;
        if (static_cast<float>(r.score()) < static_cast<float>(threshold)) continue;
        if (best < 0 || r.score() > results[static_cast<size_t>(best)].score()) best = static_cast<int>(i);
    }
    return best;
}

} // namespace mzgo
