// League play (docs/superpowers/specs/2026-11-20-league-design.md): opponents
// for the network being trained that do not play like it.
//
// The league has league_size slots, each held by the random-move bot or by
// an earlier generation's network. A new run starts with every slot held by
// the random bot. Each generation, round(games_per_generation *
// league_fraction) of its games are league games: game j plays the current
// network (the learner) against the member in slot j mod league_size, the
// learner taking Black in even games (selfplay.hpp's playLeagueGames). The
// games themselves decide who is beaten: a member against whom the learner
// scored at least league_replace_score (a draw counts half) over at least
// league_min_games games is beaten, and the most clearly beaten one (one per
// generation) is replaced by the network that played, generation g - 1's
// during generation g. The league is part of state.bin, so a resume
// continues it exactly.
#pragma once
#include <string>
#include <vector>
#include "mzgo/game_record.hpp"

namespace mzgo {

// One slot: the random-move bot (generation = -1) or the network saved as
// checkpoints/gen_GGGG.mzn.
struct LeagueMember {
    int generation = -1;
    bool isRandom() const { return generation < 0; }
    std::string name() const;   // "random" or "gen_N"
};

// The learner's results against one member.
struct LeagueResult {
    int games = 0;
    double points = 0;          // win 1, draw 0.5, loss 0
    double score() const;       // points / games, 0 if none
};

// Number of league games this generation: round(gamesPerGeneration * fraction),
// 0 when size == 0.
int leagueGameCount(int gamesPerGeneration, float fraction, int size);

// Per-member results from league records (learner Black in even games; win 1,
// draw 0.5, loss 0). opponentOf[i] is game i's member, in 0..members-1.
std::vector<LeagueResult> leagueResults(const std::vector<GameRecord>& leagueGames,
                                        const std::vector<int>& opponentOf, int members);

// The slot to replace: among members with games >= minGames and score() >=
// threshold, the highest score; ties go to the lowest slot; -1 if none. A
// member that played no games is never chosen (even with minGames 0). The
// score is held against the threshold at float precision, the config's, so
// 11 of 20 games meets a threshold of 0.55f.
int chooseReplacement(const std::vector<LeagueResult>& results, double threshold, int minGames);

} // namespace mzgo
