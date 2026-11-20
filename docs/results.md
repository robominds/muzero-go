# Training results

## Bottom line, stated plainly

**The paper's Appendix G reading for board games does not work here.**
Appendix G says "in board games without intermediate rewards, we omit the
reward prediction loss". Built that way (`reward_head = false`), the model
has no way to value the move that ends the game: the state after the last
pass is an absorbing state with value target 0, and no reward is carried on
the edge into it. So to the search a game-ending pass is worth about 0, a
draw, whatever the score. On 5×5 with komi 0.5 a winning pass got a prior of
0.028 and no visits at all. That run was stopped after 12 generations. Its
generation-12 network lost 1–39 to the reward-head run's generation 12 and
0–40 to its generation 40.

**With the reward head (the reading in §3 of the paper and in its
pseudocode), the model learns to beat itself on 5×5, but not yet outside
opponents.** Its generation 40 beats its own generations 0, 5, 20 and 30
(0.80, 0.68, 0.88, 0.70 over 40 games each). Yet it loses to the random
bot (0.33) and to the 200-simulation rollout bot (0.10). The search's
values go above 1, which no real game result can do. When the random bot
passes and passing back would win, the search rates passing at about +0.4
and rates playing on at about +1.4. In one recorded game it kept playing inside
its own territory until it filled its own last eye, and the random bot
captured all 23 of its stones on the last move before the move cap.

**The spec's own 5×5 preset (komi 24.5, no reward head) fails for a
different reason.** With komi 24.5, Black wins only by owning all 25 points.
From a cold start Black never won a recorded self-play game (0 of 120). All
of Black's moves end up with the same Q (−0.996), so its visits follow the
prior, and the prior's favourite is pass. In the recorded games after
generation 20, three of every four Black moves are passes. The arena against
generation g − 5 sits at 0.5 because the colour decides every game.

`reward_head = true` is now set in both presets (see "What changed" at the
end). The 9×9 run is in progress with it on and has nothing to report yet.

## How these were measured

All runs used the `build/` binaries (Release, BLAS) on a Mac Studio (Apple
M5 Max, 18 cores) that other training jobs were sharing. Training used 6
self-play workers and 6 training threads. Every measurement below ran at
`nice -n 10` on 2 workers. `mzgo_match` alternates colours (A is Black in
games 0, 2, 4, ...) and prints only the totals, so these results are not
split by colour. Over 40 games, a score has a standard error of about
±0.08. Unless a command says otherwise, matches use 100 simulations and 2
opening moves sampled in proportion to visits (`--temperature-moves 2`).
These are the arena settings used during training (`arena_simulations`,
`arena_temperature_moves`). The per-generation arena columns in `log.csv`
come from 20-game matches (±0.11).

Outputs below are verbatim, except that header lines and board diagrams are
sometimes trimmed. `...` marks a cut.

The three 5×5 runs share every key except these three:

| run | komi | `reward_head` | generations | wall time (self-play + training, from `log.csv`) |
|---|---|---|---|---|
| `runs/5x5-s1` | 24.5 | false | 60 | 790 s + 345 s = 18.9 min |
| `runs/5x5-k05` | 0.5 | false | 12 (stopped; configured for 40) | 79 s + 66 s = 2.4 min |
| `runs/5x5-k05-rh` | 0.5 | true | 40 | 390 s + 229 s = 10.3 min |

The other keys are those of `configs/5x5.cfg`: 3 + 3 blocks × 32 filters, 200
games and 200 simulations per generation, 8 temperature moves, a replay
window of 2000 games, 200 steps of batch 128, K = 5, lr 0.02.

Black's self-play record comes from the recorded SGFs (`sgf_per_generation
= 2`, so two games per generation, not all 200):

```
$ python3 - <<'EOF'
import glob, re
for run in ["5x5-s1", "5x5-k05", "5x5-k05-rh"]:
    games = [open(f).read() for f in sorted(glob.glob(f"runs/{run}/sgf/*.sgf"))]
    black_wins = sum("RE[B+" in g for g in games)
    b_moves = b_passes = pass_answers = pass_answered_by_pass = 0
    for g in games:
        mv = re.findall(r";([BW])\[([a-z]{0,2})\]", g)
        for i, (c, m) in enumerate(mv):
            if c == "B": b_moves += 1; b_passes += (m == "")
            if i > 0 and mv[i-1][1] == "":
                pass_answers += 1; pass_answered_by_pass += (m == "")
    print(f"{run}: {len(games)} games, Black won {black_wins}; "
          f"{b_passes}/{b_moves} Black moves are passes; "
          f"{pass_answered_by_pass}/{pass_answers} passes answered by a pass")
EOF
5x5-s1: 120 games, Black won 0; 1764/2444 Black moves are passes; 117/1778 passes answered by a pass
5x5-k05: 24 games, Black won 16; 34/248 Black moves are passes; 24/79 passes answered by a pass
5x5-k05-rh: 80 games, Black won 46; 243/1311 Black moves are passes; 78/473 passes answered by a pass
```

So Black wins 0%, 67% and 57% of the recorded games. Two passes in a row
end the game, so "answered by a pass" counts how games ended. The other
endings are the move cap (2n² = 50 moves).

## Run 1: `5x5-s1`, the spec's preset (komi 24.5, no reward head)

`runs/5x5-s1/log.csv`, selected generations (value and policy losses at
k = 0; arena scores for the newer network over 20 games):

```
generation  mean_length  loss_total  policy_k0  value_k0     arena_prev  arena_rollout  arena_random
         1       36.555     6.24221    3.04922  0.161947           0.15           0.05             0
         2        19.22     5.93422    3.00032  0.0579466           0.5              0             0
        10       19.925     4.83207    2.54189  0.00804269          0.5              0          0.05
        20        46.97      3.2261    1.62824  0.00619525         0.55           0.05          0.15
        30        47.26     3.34057    1.66127  3.27651e-06         0.5              0           0.3
        40       47.395      3.4479    1.71343  0.000157887         0.5              0          0.35
        60         46.3     3.46787    1.65542  0.00678492          0.5              0          0.35
```

Over generations 20–60, `arena_prev` is 0.5 in 40 of the 41 generations
(0.55 at generation 20). `arena_random` ranges from 0.10 to 0.45, and
`arena_rollout` never goes above 0.05. The value loss at k = 0 is between
1.6e-06 and 0.008. The network predicts the result almost perfectly,
because the result never varies: White wins.

A self-play game from generation 30 (`runs/5x5-s1/sgf/gen_0030_0.sgf`, `RE[W+49.5]`):
Black plays three stones, then passes on every move while White fills the
board:

```
;B[cb];W[ad];B[ce];W[be];B[ab];W[eb];B[];W[db];B[];W[bb];B[];W[cc];B[] ... ;W[ac];B[];W[]
```

What the network thinks after `C3 B2 D4 pass`:

```
$ ./build/mzgo_analyze --checkpoint runs/5x5-s1/checkpoints/gen_0040.mzn --moves "C3 B2 D4 pass"
root value -0.996 (for Black)

  move  visits   prior       Q
  pass      30   0.144  -0.996
    C2      20   0.095  -0.996
    A5      14   0.068  -0.995
    C4      12   0.060  -0.995
    A1      11   0.055  -0.995
    E1       9   0.043  -0.996
    B4       9   0.042  -0.994
    A2       8   0.042  -0.998
    D5       8   0.041  -0.996
    B5       8   0.040  -0.994

search depth: max 8, mean 3.14; simulations past the real game's end (two passes): 15.0%
```

The opponent has just passed, so this pass ends the game. It has the same Q
as every other move, and the largest prior.

Matches of the final network:

```
$ ./build/mzgo_match --a ckpt:runs/5x5-s1/checkpoints/gen_0060.mzn --b ckpt:runs/5x5-s1/checkpoints/gen_0010.mzn --games 40 --board-size 5 --komi 24.5 --simulations 100 --temperature-moves 2 --workers 2 --seed 1
A ckpt:runs/5x5-s1/checkpoints/gen_0060.mzn: 21 wins / B ckpt:runs/5x5-s1/checkpoints/gen_0010.mzn: 19 wins / 0 draws, scoreA 0.5250
$ ./build/mzgo_match --a ckpt:runs/5x5-s1/checkpoints/gen_0060.mzn --b random --games 40 --board-size 5 --komi 24.5 --simulations 100 --temperature-moves 2 --workers 2 --seed 1
A ckpt:runs/5x5-s1/checkpoints/gen_0060.mzn: 13 wins / B random: 27 wins / 0 draws, scoreA 0.3250
```

After 50 more generations, generation 60 scores no better than even
against generation 10. And 13 wins in 40 games against the random bot means
it lost at least 7 of its 20 games as White. With komi 24.5, White loses
only if Black ends up owning every point.

**Diagnosis.** A cold start with komi 24.5 means Black never wins a
self-play game, so every Black value target is −1. The value head learns
that well (value loss near 0), and every Black move gets the same Q. With
equal Q, the visits follow the prior. The visit distribution is the next
policy target, so whatever the prior favours gets reinforced. After the
first `temperature_moves` = 8 moves, self-play picks the most visited move.
Here the favourite is pass, and Black passes on 75% of its moves in the
recorded games from generations 20–60 (1464 of 1963). This is a result about
the komi on a cold start, not about MuZero's model. Nothing in the signal
ever separates one Black move from another, and the reward head would not
change that (Black still never wins). Both later runs therefore use komi
0.5.

## Run 2: `5x5-k05` (komi 0.5, no reward head; stopped at generation 12)

The controller stopped this run after generation 12. By then, its generation
5 had lost to its own random initialization:

```
$ ./build/mzgo_match --a ckpt:runs/5x5-k05/checkpoints/gen_0005.mzn --b ckpt:runs/5x5-k05/checkpoints/gen_0000.mzn --games 20 --board-size 5 --komi 0.5 --workers 2
A ckpt:runs/5x5-k05/checkpoints/gen_0005.mzn: 2 wins / B ckpt:runs/5x5-k05/checkpoints/gen_0000.mzn: 18 wins / 0 draws, scoreA 0.1000
```

(That match used the defaults of 200 simulations and 4 temperature moves.)
`log.csv` shows `arena_random` 0 at generation 5 and 0.25 at generation 12.
`arena_prev` (against generation g − 5) is 0.65, 0.55 and 0.70 at
generations 10–12. So the run was not standing still against itself when
it was stopped. At generation 12:

```
$ ./build/mzgo_match --a ckpt:runs/5x5-k05/checkpoints/gen_0012.mzn --b ckpt:runs/5x5-k05/checkpoints/gen_0000.mzn --games 40 --board-size 5 --komi 0.5 --simulations 100 --temperature-moves 2 --workers 2 --seed 1
A ckpt:runs/5x5-k05/checkpoints/gen_0012.mzn: 29 wins / B ckpt:runs/5x5-k05/checkpoints/gen_0000.mzn: 11 wins / 0 draws, scoreA 0.7250
$ ./build/mzgo_match --a ckpt:runs/5x5-k05/checkpoints/gen_0012.mzn --b random --games 40 --board-size 5 --komi 0.5 --simulations 100 --temperature-moves 2 --workers 2 --seed 1
A ckpt:runs/5x5-k05/checkpoints/gen_0012.mzn: 7 wins / B random: 33 wins / 0 draws, scoreA 0.1750
```

In the position after `C3 pass`, a pass by Black ends the game. Under
Tromp-Taylor rules Black then owns all 25 points and wins by 24.5:

```
$ ./build/mzgo_analyze --checkpoint runs/5x5-k05/checkpoints/gen_0005.mzn --moves "C3 pass" --komi 0.5
root value +0.058 (for Black)

  move  visits   prior       Q
    D3     126   0.092  +0.164
    D4      17   0.045  +0.074
    E3      15   0.062  +0.037
    D1       5   0.056  -0.245
    ...
search depth: max 38, mean 10.33; simulations past the real game's end (two passes): 46.0%

principal variation (imagined by the dynamics; value for Black):
  ply  player  move    value
    1   Black    D3   +0.164
    2   White    C4   +0.058
    3   Black    D2   +0.014
    4   White    B4   +0.011
    5   Black  pass   +0.006
    6   White  pass   +0.012
    7   Black  pass   +0.006
    ...
   20   White  pass   +0.011
$ ./build/mzgo_analyze --checkpoint runs/5x5-k05/checkpoints/gen_0005.mzn --moves "C3 pass" --komi 0.5 --top 26 | sed -n '/move  visits/,/^$/p' | grep -E "move|pass"
  move  visits   prior       Q
  pass       0   0.028  +0.000
```

The winning pass has prior 0.028 and gets no visits. (The brief behind this
write-up recorded "≤ 3 visits" for it. This run gives 0.) In the
principal variation, once both sides have passed, the values flatten to
about 0, which is what the absorbing-state target teaches.

**Diagnosis.** There is no reward head, so the game's result is never
attached to the move that ends the game. The state after the final pass is
absorbing, and its value target is 0. So the search values a game-ending
move at about 0 − 0 = 0, whatever the score
([`algorithm-explained.md`](algorithm-explained.md), chapter 06). The
result still reaches the model through the value of the positions before
the end. But choosing *when* to end the game, the one decision in Go that
the score depends on directly, gets no signal. The evidence above shows the
mechanism (a pass worth +24.5 is never examined). The run is too short to
show how much it costs over a long run. The cross-run matches below show it
was well behind the reward-head run at the same generation.

## Run 3: `5x5-k05-rh` (komi 0.5, reward head on)

`runs/5x5-k05-rh/log.csv`, selected generations:

```
generation  mean_length  loss_total  policy_k0  value_k0  reward_k1   arena_prev  arena_rollout  arena_random
         1       34.605     7.18988    3.07579  0.615895  0.0344401           0              0             0
         5        25.21     6.23963    2.80976  0.405648  0.0277385         0.4              0           0.1
        10       30.625     5.69494     2.4981  0.426599  0.0203102        0.55              0           0.5
        15       28.745      5.2424       2.23  0.432133  0.0130433         0.9              0          0.55
        20       31.485     4.93705    2.07528  0.417316  0.0106087        0.75              0          0.45
        25        39.18     4.53963    1.88619  0.365835  0.00809366        0.9           0.05          0.35
        30       32.685     4.30412    1.74729  0.353343  0.00816324        0.8              0          0.35
        35       32.515     4.36139    1.70243  0.420536  0.00774412        0.5              0           0.2
        40        27.43     4.55903    1.69492   0.53924  0.0114487           1           0.05          0.25
```

Against generation g − 5, the score is 0.5 or more in 33 of the 40
generations (all values: `.00 .10 .20 .20 .40 .85 1.00 .80 .75 .55 .90 .50
.90 .60 .90 .90 .95 .80 .40 .75 .35 .75 .85 .65 .90 .95 .80 .60 .50 .80
.65 1.00 .95 .60 .50 1.00 .55 .50 .50 1.00`). Against the outside bots, the
score never settles above even. `arena_random` ranges from 0 to 0.75 (0.75
once, at generation 9) and is 0.05–0.50 from generation 20 on.
`arena_rollout` stays between 0 and 0.10.

Generation 40 in 40-game matches:

```
$ ./build/mzgo_match --a ckpt:runs/5x5-k05-rh/checkpoints/gen_0040.mzn --b ckpt:runs/5x5-k05-rh/checkpoints/gen_0000.mzn --games 40 --board-size 5 --komi 0.5 --simulations 100 --temperature-moves 2 --workers 2 --seed 1
A ckpt:runs/5x5-k05-rh/checkpoints/gen_0040.mzn: 32 wins / B ckpt:runs/5x5-k05-rh/checkpoints/gen_0000.mzn: 8 wins / 0 draws, scoreA 0.8000
$ ./build/mzgo_match --a ckpt:runs/5x5-k05-rh/checkpoints/gen_0040.mzn --b ckpt:runs/5x5-k05-rh/checkpoints/gen_0005.mzn --games 40 --board-size 5 --komi 0.5 --simulations 100 --temperature-moves 2 --workers 2 --seed 1
A ckpt:runs/5x5-k05-rh/checkpoints/gen_0040.mzn: 27 wins / B ckpt:runs/5x5-k05-rh/checkpoints/gen_0005.mzn: 13 wins / 0 draws, scoreA 0.6750
$ ./build/mzgo_match --a ckpt:runs/5x5-k05-rh/checkpoints/gen_0040.mzn --b ckpt:runs/5x5-k05-rh/checkpoints/gen_0020.mzn --games 40 --board-size 5 --komi 0.5 --simulations 100 --temperature-moves 2 --workers 2 --seed 1
A ckpt:runs/5x5-k05-rh/checkpoints/gen_0040.mzn: 35 wins / B ckpt:runs/5x5-k05-rh/checkpoints/gen_0020.mzn: 5 wins / 0 draws, scoreA 0.8750
$ ./build/mzgo_match --a ckpt:runs/5x5-k05-rh/checkpoints/gen_0040.mzn --b ckpt:runs/5x5-k05-rh/checkpoints/gen_0030.mzn --games 40 --board-size 5 --komi 0.5 --simulations 100 --temperature-moves 2 --workers 2 --seed 1
A ckpt:runs/5x5-k05-rh/checkpoints/gen_0040.mzn: 28 wins / B ckpt:runs/5x5-k05-rh/checkpoints/gen_0030.mzn: 12 wins / 0 draws, scoreA 0.7000
$ ./build/mzgo_match --a ckpt:runs/5x5-k05-rh/checkpoints/gen_0040.mzn --b rollout:200 --games 40 --board-size 5 --komi 0.5 --simulations 100 --temperature-moves 2 --workers 2 --seed 1
A ckpt:runs/5x5-k05-rh/checkpoints/gen_0040.mzn: 4 wins / B rollout:200: 36 wins / 0 draws, scoreA 0.1000
$ ./build/mzgo_match --a ckpt:runs/5x5-k05-rh/checkpoints/gen_0040.mzn --b random --games 40 --board-size 5 --komi 0.5 --simulations 100 --temperature-moves 2 --workers 2 --seed 1
A ckpt:runs/5x5-k05-rh/checkpoints/gen_0040.mzn: 13 wins / B random: 27 wins / 0 draws, scoreA 0.3250
```

Early on it was not monotone either. Generation 5 scored 0.425 against
generation 0, and generation 12 scored 0.35 (and 0.275 against random):

```
$ ./build/mzgo_match --a ckpt:runs/5x5-k05-rh/checkpoints/gen_0005.mzn --b ckpt:runs/5x5-k05-rh/checkpoints/gen_0000.mzn --games 40 --board-size 5 --komi 0.5 --simulations 100 --temperature-moves 2 --workers 2 --seed 1
A ckpt:runs/5x5-k05-rh/checkpoints/gen_0005.mzn: 17 wins / B ckpt:runs/5x5-k05-rh/checkpoints/gen_0000.mzn: 23 wins / 0 draws, scoreA 0.4250
$ ./build/mzgo_match --a ckpt:runs/5x5-k05-rh/checkpoints/gen_0012.mzn --b ckpt:runs/5x5-k05-rh/checkpoints/gen_0000.mzn --games 40 --board-size 5 --komi 0.5 --simulations 100 --temperature-moves 2 --workers 2 --seed 1
A ckpt:runs/5x5-k05-rh/checkpoints/gen_0012.mzn: 14 wins / B ckpt:runs/5x5-k05-rh/checkpoints/gen_0000.mzn: 26 wins / 0 draws, scoreA 0.3500
$ ./build/mzgo_match --a ckpt:runs/5x5-k05-rh/checkpoints/gen_0012.mzn --b random --games 40 --board-size 5 --komi 0.5 --simulations 100 --temperature-moves 2 --workers 2 --seed 1
A ckpt:runs/5x5-k05-rh/checkpoints/gen_0012.mzn: 11 wins / B random: 29 wins / 0 draws, scoreA 0.2750
```

### Playing the random bot

To see how it loses to the random bot, I played games through `mzgo_gtp`
(100 simulations, komi 0.5). A small Python driver plays the random side.
It picks uniformly among the legal points and passes only when none is
left. Like `mzgo_match`, it stops at the engine's move cap of 50 moves.
(`mzgo_gtp` itself accepts moves past the cap. An earlier version of the
driver ran to 60 plies, and those games are not used here.) In six games,
three as each colour, MuZero won four. That is too few to estimate
anything; `mzgo_match` above is the measurement. One loss shows the
failure plainly. MuZero is Black, seed 3:

```
MZ:C3 R:B2 MZ:C2 R:E1 MZ:B1 R:C1 MZ:D2 R:A5 MZ:D3 R:D1 MZ:E3 R:D4 MZ:A2 R:A4 MZ:B4 R:C4
MZ:B5 R:A1 MZ:B3 R:E2 MZ:B1 R:E1 MZ:D1 R:E4 MZ:C1 R:E5 MZ:D5 R:C5 MZ:B2 R:pass MZ:A1 R:pass
MZ:D5 R:C5 MZ:A3 R:E4 MZ:A4 R:C4 MZ:A5 R:D4 MZ:E5 R:C4 MZ:C5 R:D4 MZ:E4 R:C4 MZ:E2 R:pass
MZ:E1 R:D4
= W+25.5
```

Replaying the game through `mzgo_gtp` (`showboard`, `final_score`) gives
the position after the random bot's first pass (ply 30), when passing back
would have won by 4.5, and the last two positions:

```
after ply 30 (R:pass):              after ply 49 (MZ:E1):               after ply 50 (R:D4):
   A B C D E                           A B C D E                           A B C D E
 5 O X O . O  5                      5 X X X X X  5                      5 . . . . .  5
 4 O X O O O  4                      4 X X O . X  4                      4 . . O O .  4
 3 . X X X X  3                      3 X X X X X  3                      3 . . . . .  3
 2 X X X X .  2                      2 X X X X X  2                      2 . . . . .  2
 1 . X X X O  1                      1 X X X X X  1                      1 . . . . .  1
score if the game ended now: B+4.5  score if the game ended now: B+21.5  score if the game ended now: W+25.5
```

The random bot passed at plies 30, 32 and 48, and each time MuZero played
on. At ply 47 MuZero captured the White stone on E1 by playing E2. At ply
49, with B+21.5 on the board and the random bot having just passed, it
played E1 and filled its own second eye. The one remaining liberty was D4,
and the random bot played it, capturing all 23 Black stones.

What the search sees after ply 30:

```
$ ./build/mzgo_analyze --checkpoint runs/5x5-k05-rh/checkpoints/gen_0040.mzn --komi 0.5 --top 26 --moves "C3 B2 C2 E1 B1 C1 D2 A5 D3 D1 E3 D4 A2 A4 B4 C4 B5 A1 B3 E2 B1 E1 D1 E4 C1 E5 D5 C5 B2 pass" | grep -vE "^\s+[A-E][1-5]\s+0 "
root value +1.198 (for Black)

  move  visits   prior       Q
    A1     137   0.272  +1.353
  pass      20   0.409  +0.408
    A3      20   0.145  +1.062
    D5      18   0.104  +1.160
    E2       5   0.071  +0.800

search depth: max 14, mean 5.91; simulations past the real game's end (two passes): 71.5%
```

(The `grep` drops the moves with no visits.) In a second game (seed 1, MuZero as
Black, won B+24.5 at the cap), the same thing happens after the random
bot's pass at ply 42, when passing would have won by 15.5:

```
$ ./build/mzgo_analyze --checkpoint runs/5x5-k05-rh/checkpoints/gen_0040.mzn --komi 0.5 --top 26 --moves "C3 C1 C2 A5 B2 E4 B3 A2 D3 A4 D2 A3 B1 E3 C5 D5 B4 B5 E1 C4 D1 E5 C1 C5 A1 E2 D4 D5 E4 E3 A2 C4 C5 A3 pass A5 pass B5 A4 A5 A3 pass"
root value +1.424 (for Black)

  move  visits   prior       Q
    B5     106   0.413  +1.465
    E5      85   0.299  +1.486
  pass       9   0.215  +0.351
    C4       0   0.047  +0.000
    E2       0   0.027  +0.000

search depth: max 19, mean 7.62; simulations past the real game's end (two passes): 85.5%
```

**Diagnosis (partial).** A real game result is −1, 0 or +1, and one
result is all a board game ever pays. So under the absorbing-state targets,
no Q above +1 should exist. A reward on the final edge is followed by value
0, and every other edge carries reward 0. The search reports +1.35 and
+1.49 for moves that keep the game going, and root values of +1.20 and
+1.42. Meanwhile it rates the pass that would really end the game with a
win at +0.35 to +0.41. And 72–86% of its simulations run past the point
where the real game would have ended. So the learned reward and value
together do not respect the structure the targets describe. Somewhere past
the real end, imagined steps are paying out reward again, or non-final
edges carry reward. The model then rates "keep playing" above "end the game
and win", and keeps playing until it fills its own last eye. In the games
above, the visits went to the inflated moves (A1 got 137 of 200 visits at
200 simulations). I have not yet measured which edges carry the reward:
no probe reports the predicted rewards.

**A hypothesis this write-up does not confirm.** The brief behind this
write-up proposed a different mechanism. Self-play endings have no dead
stones on the board, and Tromp-Taylor counts every stone, so after an
invasion the reward head might predict a win that the real count does not
give. (`alphazero-go`'s search scores real terminal positions, so it has to
capture dead stones.) This is plausible: only 78 of 473 passes in the
recorded games were answered by a pass, so the reward head sees few endings
of any kind. But the positions I examined show the opposite error. The real
count gave Black the win, and the model *under*-rated the pass. Testing the
hypothesis needs a position where passing loses because of dead stones
while the model rates the pass highly. I have not built one.

## The probes

All on 20 rollout-bot games, K = 5, seed 1, 2 workers. The whole set took
7.3 s.

### `mzgo_latent_probe`: does the imagined state agree with a fresh look?

```
$ ./build/mzgo_latent_probe --checkpoint runs/5x5-k05-rh/checkpoints/gen_0010.mzn --games 20 --k 5 --workers 2 --seed 1 --komi 0.5
   k   samples   mean |value error|   mean policy TV distance
   0       703               0.0000                    0.0000
   1       683               0.6155                    0.2669
   2       663               0.2765                    0.2624
   3       643               0.7034                    0.3335
   4       623               0.3775                    0.3357
   5       603               0.7321                    0.3844

$ ./build/mzgo_latent_probe --checkpoint runs/5x5-k05-rh/checkpoints/gen_0040.mzn --games 20 --k 5 --workers 2 --seed 1 --komi 0.5
   k   samples   mean |value error|   mean policy TV distance
   0       703               0.0000                    0.0000
   1       683               0.5085                    0.3154
   2       663               0.3469                    0.3544
   3       643               0.5597                    0.4313
   4       623               0.4503                    0.4631
   5       603               0.6038                    0.5121

$ ./build/mzgo_latent_probe --checkpoint runs/5x5-s1/checkpoints/gen_0060.mzn --games 20 --k 5 --workers 2 --seed 1 --komi 24.5
   k   samples   mean |value error|   mean policy TV distance
   0       850               0.0000                    0.0000
   1       830               0.0670                    0.2882
   2       810               0.0965                    0.3188
   3       790               0.1588                    0.3652
   4       770               0.1909                    0.3962
   5       750               0.2160                    0.4305
```

The policy drifts steadily with k in all three. By k = 5, about half of the
reward-head model's prior is on different moves from what a fresh look
would give. In the reward-head run, the drift grows from generation 10 to
generation 40, while the model gets better at self-play. The value errors in
the reward-head run alternate with k: odd steps are worse (0.51–0.73) than
even steps (0.28–0.45). I don't know why. A step that changes the side to
move is the obvious suspect, but I haven't tested it. The `5x5-s1` value
errors are small for the same reason its value loss is: the value is almost
constant.

### `mzgo_rules_probe`: prior mass on illegal moves

```
$ ./build/mzgo_rules_probe --checkpoint runs/5x5-k05-rh/checkpoints/gen_0010.mzn --games 20 --k 5 --workers 2 --seed 1 --komi 0.5
   k   samples    illegal   occupied    superko   captured
   0       703     0.1091     0.0754     0.0007     0.0000
   1       683     0.1822     0.1349     0.0005     0.0036
   2       663     0.1921     0.1511     0.0004     0.0097
   3       643     0.2626     0.2116     0.0006     0.0176
   4       623     0.2678     0.2261     0.0006     0.0223
   5       603     0.3351     0.2869     0.0008     0.0306

$ ./build/mzgo_rules_probe --checkpoint runs/5x5-k05-rh/checkpoints/gen_0040.mzn --games 20 --k 5 --workers 2 --seed 1 --komi 0.5
   k   samples    illegal   occupied    superko   captured
   0       703     0.0454     0.0113     0.0016     0.0000
   1       683     0.0739     0.0243     0.0006     0.0020
   2       663     0.1138     0.0593     0.0011     0.0124
   3       643     0.1768     0.1121     0.0009     0.0246
   4       623     0.2104     0.1565     0.0011     0.0350
   5       603     0.2784     0.2208     0.0010     0.0491

$ ./build/mzgo_rules_probe --checkpoint runs/5x5-s1/checkpoints/gen_0060.mzn --games 20 --k 5 --workers 2 --seed 1 --komi 24.5
   k   samples    illegal   occupied    superko   captured
   0       850     0.0605     0.0166     0.0006     0.0000
   1       830     0.1281     0.0491     0.0001     0.0032
   2       810     0.1575     0.0804     0.0001     0.0095
   3       790     0.2165     0.1295     0.0001     0.0201
   4       770     0.2520     0.1752     0.0001     0.0312
   5       750     0.3014     0.2250     0.0002     0.0491
```

Training does teach legality on the real position. Between generations 10
and 40, the illegal mass at k = 0 falls from 0.109 to 0.045 (occupied
points from 0.075 to 0.011). But the lesson fades with depth: 5 imagined
moves ahead, 28–34% of the prior is on moves the real rules forbid, mostly
occupied points. "Captured" (playing on a point whose stones the move
before captured) grows from about 0 to 0.03–0.05. The model is slowly
learning that captures free points.

### `mzgo_board_probe`: can the stones be read back out of the state?

```
$ ./build/mzgo_board_probe --checkpoint runs/5x5-k05-rh/checkpoints/gen_0010.mzn --games 20 --k 5 --workers 2 --seed 1 --komi 0.5
   k     points   accuracy    changed   changed acc.
   0      18075     0.7665          0              -
   1      17575     0.4084        855         0.3778
   2      17075     0.4201       1568         0.3463
   3      16575     0.4034       2241         0.3525
   4      16075     0.4030       2753         0.3200
   5      15575     0.3939       3243         0.3250

$ ./build/mzgo_board_probe --checkpoint runs/5x5-k05-rh/checkpoints/gen_0040.mzn --games 20 --k 5 --workers 2 --seed 1 --komi 0.5
   k     points   accuracy    changed   changed acc.
   0      18075     0.7414          0              -
   1      17575     0.3808        855         0.4596
   2      17075     0.3781       1568         0.3438
   3      16575     0.3795       2241         0.3498
   4      16075     0.3737       2753         0.3153
   5      15575     0.3758       3243         0.3235

$ ./build/mzgo_board_probe --checkpoint runs/5x5-s1/checkpoints/gen_0060.mzn --games 20 --k 5 --workers 2 --seed 1 --komi 24.5
   k     points   accuracy    changed   changed acc.
   0      21750     0.6671          0              -
   1      21250     0.3880       1049         0.4480
   2      20750     0.3686       1945         0.4267
   3      20250     0.3692       2791         0.3898
   4      19750     0.3595       3470         0.4023
   5      19250     0.3583       4115         0.3701
```

A linear readout fitted on fresh `h` states reads the board back at
67–77% (three classes per point: empty, own, opponent). Applied to the
state one imagined step later, it drops to 36–42%, and it stays there at
deeper steps. So the dynamics' states are not in the representation's
format, at least not linearly. A one-step drop that does not keep falling
looks like a change of format, not a gradual loss of information. The probe
cannot tell those two apart, and I did not measure a majority-class
baseline.

## Across runs: the same generation, with and without the reward head

```
$ ./build/mzgo_match --a ckpt:runs/5x5-k05-rh/checkpoints/gen_0012.mzn --b ckpt:runs/5x5-k05/checkpoints/gen_0012.mzn --games 40 --board-size 5 --komi 0.5 --simulations 100 --temperature-moves 2 --workers 2 --seed 1
A ckpt:runs/5x5-k05-rh/checkpoints/gen_0012.mzn: 39 wins / B ckpt:runs/5x5-k05/checkpoints/gen_0012.mzn: 1 wins / 0 draws, scoreA 0.9750
$ ./build/mzgo_match --a ckpt:runs/5x5-k05-rh/checkpoints/gen_0040.mzn --b ckpt:runs/5x5-k05/checkpoints/gen_0012.mzn --games 40 --board-size 5 --komi 0.5 --simulations 100 --temperature-moves 2 --workers 2 --seed 1
A ckpt:runs/5x5-k05-rh/checkpoints/gen_0040.mzn: 40 wins / B ckpt:runs/5x5-k05/checkpoints/gen_0012.mzn: 0 wins / 0 draws, scoreA 1.0000
$ ./build/mzgo_match --a ckpt:runs/5x5-k05-rh/checkpoints/gen_0040.mzn --b ckpt:runs/5x5-k05/checkpoints/gen_0000.mzn --games 40 --board-size 5 --komi 0.5 --simulations 100 --temperature-moves 2 --workers 2 --seed 1
A ckpt:runs/5x5-k05-rh/checkpoints/gen_0040.mzn: 23 wins / B ckpt:runs/5x5-k05/checkpoints/gen_0000.mzn: 17 wins / 0 draws, scoreA 0.5750
```

The two generation-0 networks are not the same file (`cmp` reports a
difference, and the reward-head file is 3,835 bytes larger). The pattern is
not transitive. The reward-head networks beat trained networks of either
kind decisively. Against the untrained networks they win less decisively
(0.80 against their own generation 0, 0.575 against the other run's), and
they lose to the random bot. What the reward-head run has learned is how to beat
opponents that play like itself.

The same `C3 pass` test as in run 2 does not separate the two runs.
The reward-head generation 40 also gives the winning pass no visits there
(prior 0.000), because a White pass on move 2 never happens in its
self-play:

```
$ ./build/mzgo_analyze --checkpoint runs/5x5-k05-rh/checkpoints/gen_0040.mzn --moves "C3 pass" --komi 0.5
root value +0.643 (for Black)

  move  visits   prior       Q
    C2     194   0.893  +0.647
    B2       6   0.059  +0.510
    B4       0   0.011  +0.000
    ...
search depth: max 17, mean 6.53; simulations past the real game's end (two passes): 0.0%
$ ./build/mzgo_analyze --checkpoint runs/5x5-k05-rh/checkpoints/gen_0040.mzn --moves "C3 pass" --komi 0.5 --top 26 | sed -n '/move  visits/,/^$/p' | grep -E "move|pass"
  move  visits   prior       Q
  pass       0   0.000  +0.000
```

## Self-play search statistics

From `log.csv` (`past_end_fraction` is the fraction of simulations that
went past two imagined passes; `capped_games` counts games that hit the
50-move cap):

```
5x5-s1      g1: past_end 0.030 capped 34 | g10: 0.666 capped 0 | g20: 0.204 capped 1 | g40: 0.187 capped 1 | g60: 0.182 capped 28
5x5-k05     g1: past_end 0.030 capped 34 | g5: 0.593 capped 0  | g12: 0.589 capped 0
5x5-k05-rh  g1: past_end 0.030 capped 17 | g5: 0.620 capped 0  | g10: 0.444 capped 6 | g20: 0.343 capped 9 | g40: 0.184 capped 1
```

With the reward head, the fraction of self-play search spent past the end
falls steadily, from 0.62 at generation 5 to 0.18 at 40. Without it, the
fraction stays near 0.6 through generation 12. In the analyzed positions
against the random bot, though, the fraction is 0.72 and 0.86. Those
positions are the ones where the model goes wrong, and they look nothing
like its self-play games.

## What is not yet known

- **Where the values above 1 come from.** It could be reward predicted on
  imagined steps after the end, or on non-final edges, or value not
  reaching 0 after a final edge. A probe that prints the predicted reward
  along real and imagined lines would settle it.
- **Whether the dead-stones hypothesis holds** (see run 3).
- **Whether the no-reward-head run would have recovered.** It was stopped
  at generation 12, while it was still improving against itself (0.725
  against its generation 0).
- **Seeds.** Every run is a single seed (seed 1). The non-monotone arena
  scores (run 3's generation 12 lost to generation 0) could be noise or
  could be real. Nothing here separates the two.
- **Komi 24.5 with the reward head.** The 5×5 preset now has the reward
  head on but still komi 24.5, and that pair has never been run. Run 1's
  failure did not depend on the reward head, so the reward head should not
  fix it. With komi 24.5, Black still has to own all 25 points to win.
- **Longer training.** 40 generations is 10 minutes on this machine. The
  learning-rate schedule had reached 0.0045 by then (`lr` column).

## 9×9

In progress, not yet measured. `runs/9x9-rh-s1.log` starts with `starting
runs/9x9-rh-s1 at generation 0 of 150`. Its configuration
(`runs/9x9-rh-s1/config.cfg`) is the 9×9 preset with `reward_head = true`,
6 workers and 6 training threads. The first generation took 349 s of
self-play and 200 s of training:

```
gen 1: selfplay 349.166s (200 games, mean length 120.445), train 199.784s (400 steps, loss 8.98326, lr 0.02), arena vs gen 0 0, rollout 0, random 0
```

At about 9 minutes per generation, 150 generations would take roughly a
day. That is only a projection, since self-play time changes with game
length. The head-to-head against `alphazero-go` will be reported here once
the run has a checkpoint worth comparing.

## What changed because of these results

- `configs/5x5.cfg` and `configs/9x9.cfg` set `reward_head = true`. The
  built-in default (`RunConfig::rewardHead`, `include/mzgo/config.hpp`)
  stays `false`, so a config file that does not mention the key keeps its
  old meaning. `test_presets_parse` in `tests/test_config.cpp` checks both.
- The spec's Dynamics bullet and deviations list record the choice. The
  paper's §3 loss and pseudocode include the reward term for every game;
  Appendix G drops it for board games. Here the pseudocode reading is the
  default, and Appendix G's reading is measured above as a failure.

---

Mark Castelluccio, designed and implemented with Claude Code (AI-assisted).
