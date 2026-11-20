#include <cassert>
#include <cstdio>
#include "mzgo/match.hpp"

using namespace mzgo;

static void test_match_alternates_and_deterministic() {
    PlayerFactory r = [] { return std::unique_ptr<Player>(new RandomPlayer()); };
    PlayerFactory m = [] { return std::unique_ptr<Player>(new RolloutMctsPlayer(20)); };
    MatchResult one = playMatch(m, r, 8, 5, 0.5f, 0, 99, 1);
    MatchResult four = playMatch(m, r, 8, 5, 0.5f, 0, 99, 4);
    assert(one.games.size() == 8 && four.games.size() == 8);
    for (int i = 0; i < 8; ++i) {
        assert(one.games[i].aWasBlack == (i % 2 == 0));
        assert(one.games[i].moves == four.games[i].moves);
        assert(one.games[i].resultForBlack == four.games[i].resultForBlack);
        assert(!one.games[i].moves.empty());
    }
    assert(one.winsA == four.winsA && one.winsB == four.winsB && one.draws == four.draws);
    assert(one.winsA + one.winsB + one.draws == 8);
    assert(MatchResult().scoreA() == 0.0);
    // A different seed gives different games.
    MatchResult other = playMatch(m, r, 8, 5, 0.5f, 0, 100, 2);
    bool differs = false;
    for (int i = 0; i < 8; ++i) differs |= other.games[i].moves != one.games[i].moves;
    assert(differs);
    std::printf("test_match_alternates_and_deterministic passed\n");
}

int main() {
    test_match_alternates_and_deterministic();
    return 0;
}
