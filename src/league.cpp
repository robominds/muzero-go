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

std::vector<int> leagueOpponents(int leagueGames, int size) {
    std::vector<int> opponentOf(static_cast<size_t>(std::max(0, leagueGames)));
    for (int j = 0; j < leagueGames; ++j) opponentOf[static_cast<size_t>(j)] = (j / 2) % size;
    return opponentOf;
}

int leagueFewestGames(int leagueGames, int size) {
    if (size <= 0) return 0;
    std::vector<int> games(static_cast<size_t>(size), 0);
    for (int m : leagueOpponents(leagueGames, size)) ++games[static_cast<size_t>(m)];
    return *std::min_element(games.begin(), games.end());
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

bool leagueBeaten(const LeagueResult& r, double threshold, int minGames) {
    return r.games >= std::max(1, minGames) && static_cast<float>(r.score()) >= static_cast<float>(threshold);
}

int chooseReplacement(const std::vector<LeagueMember>& members, const std::vector<LeagueResult>& results,
                      double threshold, int minGames, int networkWins) {
    assert(members.size() == results.size());
    int best = -1;
    for (size_t i = 0; i < results.size(); ++i) {
        const LeagueResult& r = results[i];
        if (!leagueBeaten(r, threshold, minGames)) continue;
        const bool oneWinEnough = !members[i].isRandom() && networkWins == 1;
        if (!members[i].beatenLastGeneration && !oneWinEnough) continue;   // its first strike
        if (best < 0 || r.score() > results[static_cast<size_t>(best)].score()) best = static_cast<int>(i);
    }
    return best;
}

int advanceLeague(std::vector<LeagueMember>& members, const std::vector<LeagueResult>& results, double threshold,
                  int minGames, int networkWins, bool resetOnReplace, int newMember) {
    const int slot = chooseReplacement(members, results, threshold, minGames, networkWins);
    assert(members.size() == results.size());
    for (size_t i = 0; i < members.size(); ++i)
        members[i].beatenLastGeneration = leagueBeaten(results[i], threshold, minGames);
    if (slot >= 0) {
        members[static_cast<size_t>(slot)] = LeagueMember{newMember};
        if (resetOnReplace)
            for (LeagueMember& m : members) m.beatenLastGeneration = false;
    }
    return slot;
}

} // namespace mzgo
