#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include "mzgo/game_record.hpp"
#include "mzgo/league.hpp"

using namespace mzgo;

static void test_member_names() {
    assert(LeagueMember{}.isRandom() && LeagueMember{}.name() == "random");
    assert(!LeagueMember{12}.isRandom() && LeagueMember{12}.name() == "gen_12");
    assert(LeagueMember{0}.name() == "gen_0");
    assert(LeagueResult{}.score() == 0);
    assert((LeagueResult{4, 3}.score() == 0.75));
    std::printf("test_member_names passed\n");
}

static void test_game_count() {
    assert(leagueGameCount(200, 0.5f, 3) == 100);
    assert(leagueGameCount(200, 0.5f, 0) == 0);
    assert(leagueGameCount(3, 0.1f, 2) == 0);
    assert(leagueGameCount(4, 0.5f, 2) == 2);
    assert(leagueGameCount(4, 1.0f, 2) == 4);
    assert(leagueGameCount(4, 0.0f, 2) == 0);
    std::printf("test_game_count passed\n");
}

// Every slot meets the learner in both colours (Black in even games), for an
// odd league size and an even one, and pairs share a slot.
static void test_opponents() {
    assert((leagueOpponents(8, 2) == std::vector<int>{0, 0, 1, 1, 0, 0, 1, 1}));
    assert((leagueOpponents(5, 3) == std::vector<int>{0, 0, 1, 1, 2}));
    assert(leagueOpponents(0, 2).empty());
    for (int size : {1, 2, 3, 4}) {
        const std::vector<int> o = leagueOpponents(4 * size, size);
        for (int slot = 0; slot < size; ++slot) {
            bool asBlack = false, asWhite = false;
            for (size_t j = 0; j < o.size(); ++j)
                if (o[j] == slot) (j % 2 == 0 ? asBlack : asWhite) = true;
            assert(asBlack && asWhite);
        }
    }
    assert(leagueFewestGames(100, 3) == 32);   // 50 pairs: 17, 17, 16
    assert(leagueFewestGames(4, 2) == 2);
    assert(leagueFewestGames(2, 2) == 0);
    assert(leagueFewestGames(5, 2) == 2);      // slot 0: 0, 1, 4; slot 1: 2, 3
    assert(leagueFewestGames(0, 3) == 0 && leagueFewestGames(10, 0) == 0);
    std::printf("test_opponents passed\n");
}

static GameRecord withResult(float resultForBlack) {
    GameRecord r;
    r.boardSize = 5;
    r.resultForBlack = resultForBlack;
    return r;
}

static void test_results() {
    // Game 0: learner Black, Black wins (a learner win) against member 0.
    // Game 1: learner White, Black wins (a learner loss) against member 1.
    // Game 2: learner Black, a draw against member 0.
    // Game 3: learner White, White wins (a learner win) against member 1.
    const std::vector<GameRecord> games = {withResult(1), withResult(1), withResult(0), withResult(-1)};
    const std::vector<LeagueResult> r = leagueResults(games, {0, 1, 0, 1}, 3);
    assert(r.size() == 3);
    assert(r[0].games == 2 && r[0].points == 1.5 && r[0].score() == 0.75);
    assert(r[1].games == 2 && r[1].points == 1.0 && r[1].score() == 0.5);
    assert(r[2].games == 0 && r[2].points == 0 && r[2].score() == 0);
    // A learner loss as Black.
    const std::vector<LeagueResult> loss = leagueResults({withResult(-1)}, {0}, 1);
    assert(loss[0].games == 1 && loss[0].points == 0);
    std::printf("test_results passed\n");
}

// n network members (generation 0), none beaten last generation.
static std::vector<LeagueMember> networks(int n) { return std::vector<LeagueMember>(static_cast<size_t>(n), LeagueMember{0}); }

static void test_beaten() {
    assert(leagueBeaten({20, 11}, 0.55, 20));
    assert(leagueBeaten({20, 11}, 0.55f, 20));     // the config's float threshold: 11 of 20 still meets it
    assert(!leagueBeaten({20, 10.5}, 0.55f, 20));  // below the threshold
    assert(leagueBeaten({20, 12}, 0.6f, 20));      // the default, 0.6f, is a little above 0.6 too
    assert(!leagueBeaten({20, 11.5}, 0.6f, 20));
    assert(!leagueBeaten({19, 19}, 0.55, 20));     // too few games
    assert(!leagueBeaten({0, 0}, 0.0, 0));         // no games: nothing to judge by
    assert(leagueBeaten({1, 0}, 0.0, 0));
    std::printf("test_beaten passed\n");
}

// Network members: replaced after one win, as before the two-strike rule.
static void test_choose_replacement() {
    // Below the threshold: none.
    assert(chooseReplacement(networks(2), {{20, 10}, {20, 10.5}}, 0.55, 20) == -1);
    // Enough score but too few games: none.
    assert(chooseReplacement(networks(2), {{19, 19}, {10, 10}}, 0.55, 20) == -1);
    // Two eligible: the higher score.
    assert(chooseReplacement(networks(3), {{20, 12}, {20, 16}, {20, 5}}, 0.55, 20) == 1);
    // A tie goes to the lower slot.
    assert(chooseReplacement(networks(3), {{20, 5}, {20, 16}, {20, 16}}, 0.55, 20) == 1);
    // At the threshold and at min games exactly: eligible.
    assert(chooseReplacement(networks(1), {{20, 11}}, 0.55, 20) == 0);
    assert(chooseReplacement(networks(1), {{20, 11}}, 0.55f, 20) == 0);
    assert(chooseReplacement(networks(1), {{20, 10.5}}, 0.55f, 20) == -1);
    // A member that played no games is never replaced, even with min games 0
    // and threshold 0 (there is nothing to judge it by).
    assert(chooseReplacement(networks(2), {{0, 0}, {1, 0}}, 0.0, 0) == 1);
    assert(chooseReplacement(networks(1), {{0, 0}}, 0.0, 0) == -1);
    assert(chooseReplacement({}, {}, 0.55, 20) == -1);
    // A random member is eligible only when it was beaten last generation too.
    std::vector<LeagueMember> mixed = {LeagueMember{}, LeagueMember{3}};
    assert(chooseReplacement(mixed, {{20, 20}, {20, 5}}, 0.55, 20) == -1);
    mixed[0].beatenLastGeneration = true;
    assert(chooseReplacement(mixed, {{20, 20}, {20, 5}}, 0.55, 20) == 0);
    std::printf("test_choose_replacement passed\n");
}

static const LeagueResult kWin{20, 16}, kLoss{20, 4};

// advanceLeague over several generations (threshold 0.55, 20 games).
static void test_two_strikes() {
    // Beaten once: not replaced, but marked; beaten again next generation: replaced.
    std::vector<LeagueMember> league(1);
    assert(advanceLeague(league, {kWin}, 0.55, 20, 4) == -1);
    assert(league[0].isRandom() && league[0].beatenLastGeneration);
    assert(advanceLeague(league, {kWin}, 0.55, 20, 5) == 0);
    assert(league[0].generation == 5 && !league[0].beatenLastGeneration);

    // Beaten, not beaten, beaten: the wins are not consecutive, so it stays.
    league = std::vector<LeagueMember>(1);
    assert(advanceLeague(league, {kWin}, 0.55, 20, 1) == -1 && league[0].beatenLastGeneration);
    assert(advanceLeague(league, {kLoss}, 0.55, 20, 2) == -1 && !league[0].beatenLastGeneration);
    assert(advanceLeague(league, {kWin}, 0.55, 20, 3) == -1 && league[0].isRandom());
    assert(league[0].beatenLastGeneration);

    // A network member beaten once is replaced at once.
    league = {LeagueMember{2}};
    assert(advanceLeague(league, {kWin}, 0.55, 20, 7) == 0);
    assert(league[0].generation == 7 && !league[0].beatenLastGeneration);

    // A random member on its second strike, but a network member scored
    // higher: the network member goes, and the random one stays marked.
    league = {LeagueMember{}, LeagueMember{2}};
    league[0].beatenLastGeneration = true;
    assert(advanceLeague(league, {{20, 13}, {20, 18}}, 0.55, 20, 8) == 1);
    assert(league[0].isRandom() && league[0].beatenLastGeneration);
    assert(league[1].generation == 8 && !league[1].beatenLastGeneration);
    // ... so the next win replaces it.
    assert(advanceLeague(league, {{20, 13}, {20, 4}}, 0.55, 20, 9) == 0 && league[0].generation == 9);

    // Two random members beaten in the same generation: both marked, then
    // only one (the higher score) replaced; the other stays marked.
    league = std::vector<LeagueMember>(2);
    assert(advanceLeague(league, {{20, 12}, {20, 14}}, 0.55, 20, 1) == -1);
    assert(league[0].beatenLastGeneration && league[1].beatenLastGeneration);
    assert(advanceLeague(league, {{20, 12}, {20, 14}}, 0.55, 20, 2) == 1);
    assert(league[0].isRandom() && league[0].beatenLastGeneration && league[1].generation == 2);

    // A generation without games (league_fraction too small): nobody is
    // beaten, so a first strike is forgotten.
    league = std::vector<LeagueMember>(1);
    advanceLeague(league, {kWin}, 0.55, 20, 1);
    assert(advanceLeague(league, {LeagueResult{}}, 0.55, 20, 2) == -1 && !league[0].beatenLastGeneration);
    std::printf("test_two_strikes passed\n");
}

int main() {
    test_member_names();
    test_game_count();
    test_opponents();
    test_results();
    test_beaten();
    test_choose_replacement();
    test_two_strikes();
    return 0;
}
