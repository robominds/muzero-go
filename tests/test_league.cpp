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

static void test_choose_replacement() {
    // Below the threshold: none.
    assert(chooseReplacement({{20, 10}, {20, 10.5}}, 0.55, 20) == -1);
    // Enough score but too few games: none.
    assert(chooseReplacement({{19, 19}, {10, 10}}, 0.55, 20) == -1);
    // Two eligible: the higher score.
    assert(chooseReplacement({{20, 12}, {20, 16}, {20, 5}}, 0.55, 20) == 1);
    // A tie goes to the lower slot.
    assert(chooseReplacement({{20, 5}, {20, 16}, {20, 16}}, 0.55, 20) == 1);
    // At the threshold and at min games exactly: eligible.
    assert(chooseReplacement({{20, 11}}, 0.55, 20) == 0);
    // The config's threshold is a float (0.55f is a little above 0.55):
    // 11 of 20 still meets it.
    assert(chooseReplacement({{20, 11}}, 0.55f, 20) == 0);
    assert(chooseReplacement({{20, 10.5}}, 0.55f, 20) == -1);
    // A member that played no games is never replaced, even with min games 0
    // and threshold 0 (there is nothing to judge it by).
    assert(chooseReplacement({{0, 0}, {1, 0}}, 0.0, 0) == 1);
    assert(chooseReplacement({{0, 0}}, 0.0, 0) == -1);
    assert(chooseReplacement({}, 0.55, 20) == -1);
    std::printf("test_choose_replacement passed\n");
}

int main() {
    test_member_names();
    test_game_count();
    test_opponents();
    test_results();
    test_choose_replacement();
    return 0;
}
