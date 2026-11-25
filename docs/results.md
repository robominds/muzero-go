# Training results

## Bottom line, stated plainly

**The first reward-head results were distorted by a batch-norm defect.**
Every batch norm in `g`, `f` and the reward head kept one set of running
statistics, updated at every step of the unroll. Training normalized step
k with step k's own batch statistics, but the search normalized every
depth with the blend of all steps. So the search computed something
training never produced. The final review measured the cost on the
pre-fix run's generation 40 (chapter 08 of
[`algorithm-explained.md`](algorithm-explained.md)): the reward error at
k = 1 was 0.032 at inference against 0.0046 with the batch's statistics,
and a non-final move's predicted reward was +0.888. Commit `37df3a1` keeps
one set of statistics per unroll step and uses the leaf's depth at
inference. Commit `90cbaba` also changed the targets: a game stopped by the
move cap is now treated as truncated, not ended. Runs 1–3 below were
measured before both fixes, and their checkpoints no longer load.

**Re-run on the fixed code, the same 5×5 setup (komi 0.5, reward head on,
40 generations) now beats the random bot.** Run 4 (`5x5-k05-rh2`) scores
0.825 against the random bot over 40 games and 0.79 over 200; run 3 scored
0.33. Split by colour, it won 47 of 50 games as Black and 40 of 50 as White.
Its generation 40 beats its own generations 0, 5, 20 and 30 (0.825, 1.00,
0.875, 0.75). Predicted rewards on ordinary moves are now about 0 (within
±0.03 in every principal variation below). In 12 games against the random
bot played through `mzgo_gtp`, it won 11. It still often plays on after the
random bot passes, but in these games it never turned a won position into
a loss.

**What is still weak:**

- **The rollout bot.** Generation 40 scores 0.05 against `rollout:200` over
  40 games and 0.06 over 200 (5 of 50 games won as Black, 1 of 50 as
  White).
- **Values above 1 remain.** The batch-norm defect was not the whole
  story. In 19 of the 21 positions just after a random-bot pass, the root
  value or a visited move's Q is above 1, up to +1.70. The same holds in 17
  of 41 self-play positions just after a pass, but in none of 20 self-play
  positions at move 20. Where the search's best line starts with the
  real winning pass (9 positions), the model's reward on that pass is
  +0.23 to +0.79 instead of +1, and the value after it is +0.64 to +1.01
  instead of 0. Their sum goes above 1. Past the end, imagined passes pay
  out rewards again (up to +0.63).
- **A winning pass with a low prior is never tried.** After `C3 pass`,
  Black's pass ends the game and wins by 24.5. Its prior is 0.004 and it gets
  0 visits at 200 and at 800 simulations. An unvisited child counts as a
  loss on the normalized scale (the pseudocode's rule), so a low-prior move
  is never looked at.
- **Positions self-play rarely reaches.** A White pass on move 2, as in
  `C3 pass`, occurs in 1 of run 4's 80 recorded self-play games (at
  generation 2). The one GTP game lost to the random bot was lost because
  MuZero, as White, passed with its 13-stone group in atari, and the random
  bot captured it. The capturing move (A1) had the higher Q, but the pass
  had the higher prior and got more visits. The imagined line after that
  pass does not show the capture.

**Without a reward head (the paper's Appendix G reading for board games),
the move that ends the game is worth about 0 to the search.** Appendix G says
"in board games without intermediate rewards, we omit the reward
prediction loss". Built that way (`reward_head = false`), the state after
the last pass is an absorbing state with value target 0, and no reward is
carried on the edge into it. So to the search a game-ending pass is worth
about 0, a draw, whatever the score. On 5×5 with komi 0.5 a winning pass got
a prior of 0.028 and no visits at all (run 2, before the fix). That run
was stopped after 12 generations, and its generation 12 lost 1–39 to the
pre-fix reward-head run's generation 12. This argument does not involve
batch norm, so it should still hold. But it has not been re-run on the
fixed code.

**The spec's own 5×5 preset (komi 24.5, no reward head) fails for a
different reason.** With komi 24.5, Black wins only by owning all 25 points.
From a cold start Black never won a recorded self-play game (0 of 120). All
of Black's moves end up with the same Q (−0.996), so its visits follow the
prior, and the prior's favourite is pass (run 1, before the fix). In the
recorded games after generation 20, three of every four Black moves are
passes, and the arena against generation g − 5 sits at 0.5 because the
colour decides every game. The reasoning rests on the value targets being constant (Black never wins), not
on batch norm. So it should still hold, but it too is untested on the fixed
code.

`reward_head = true` is the default in both presets. The 9×9 run
(`runs/9x9-s1`, fixed code) is in progress and has no results yet.

## How these were measured

All runs used the `build/` binaries (Release, BLAS) on a Mac Studio (Apple
M5 Max, 18 cores) that other training jobs were sharing. Training used 6
self-play workers and 6 training threads. Every measurement below ran at
`nice -n 10` on 2 workers or fewer. `mzgo_match` alternates colours (A is
Black in games 0, 2, 4, ...) and prints only the totals. For a split by colour,
run 4 uses one-game matches (A is Black in game 0). Over 40 games, a score
has a standard error of about ±0.08, and over 200 about ±0.03. Unless a
command says otherwise, matches use 100 simulations and 2
opening moves sampled in proportion to visits (`--temperature-moves 2`).
These are the arena settings used during training (`arena_simulations`,
`arena_temperature_moves`). The per-generation arena columns in `log.csv`
come from 20-game matches (±0.11).

Outputs below are verbatim, except that header lines and board diagrams are
sometimes trimmed. `...` marks a cut.

The four 5×5 runs share every key except these:

| run | code | komi | `reward_head` | generations | wall time (self-play + training, from `log.csv`) |
|---|---|---|---|---|---|
| `runs/5x5-s1` | before the fix | 24.5 | false | 60 | 790 s + 345 s = 18.9 min |
| `runs/5x5-k05` | before the fix | 0.5 | false | 12 (stopped; configured for 40) | 79 s + 66 s = 2.4 min |
| `runs/5x5-k05-rh` | before the fix | 0.5 | true | 40 | 390 s + 229 s = 10.3 min |
| `runs/5x5-k05-rh2` | after `37df3a1` and `90cbaba` | 0.5 | true | 40 | 440 s + 247 s = 11.5 min |

"Before the fix" means one set of batch-norm running statistics per layer,
and a capped game scored as if it had ended. The checkpoints of runs 1–3
use checkpoint format version 1 and no longer load, so everything measured
on them below is history; their `log.csv` files and SGFs are still readable.

The other keys are those of `configs/5x5.cfg`: 3 + 3 blocks × 32 filters, 200
games and 200 simulations per generation, 8 temperature moves, a replay
window of 2000 games, 200 steps of batch 128, K = 5, lr 0.02.

Black's self-play record comes from the recorded SGFs (`sgf_per_generation
= 2`, so two games per generation, not all 200):

```
$ python3 - <<'EOF'
import glob, re
for run in ["5x5-s1", "5x5-k05", "5x5-k05-rh", "5x5-k05-rh2"]:
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
5x5-k05-rh2: 80 games, Black won 46; 151/1402 Black moves are passes; 73/268 passes answered by a pass
```

So Black wins 0%, 67%, 57% and 57% of the recorded games. Two passes in a
row end the game, so "answered by a pass" counts how games ended. The other
endings are the move cap (2n² = 50 moves). Seven of run 4's 80 recorded
games hit the cap; their `RE` is the score at the cap.

## Run 1: `5x5-s1`, the spec's preset (komi 24.5, no reward head)

> **Note.** Measured before the per-depth batch-norm fix and before capped
> games were treated as truncated. The checkpoints no longer load. The
> diagnosis below rests on the value targets being constant, not on batch
> norm, so I expect it to hold on the fixed code. That is untested.

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

> **Note.** Measured before the per-depth batch-norm fix. The checkpoints
> no longer load. With no reward head, the edge into the absorbing state
> carries no reward at all, whatever batch norm does, so the diagnosis
> below should still hold. It has not been re-run on the fixed code.

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

## Run 3: `5x5-k05-rh` (komi 0.5, reward head on; before the fix)

> **Note.** Measured before the per-depth batch-norm fix and before games
> stopped by the move cap were treated as truncated. The checkpoints use
> checkpoint format version 1 and no longer load. Run 4 re-measures the
> same setup on the fixed code. Some of this section's conclusions turned
> out wrong; ["Pre-fix diagnoses, revisited"](#pre-fix-diagnoses-revisited)
> says which.

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
no probe reported the predicted rewards at the time (`mzgo_analyze` now
prints each principal-variation edge's predicted reward and Q).

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

## Run 4: `5x5-k05-rh2`, re-measured after the per-depth batch-norm fix

Same configuration as run 3 (`runs/5x5-k05-rh2/config.cfg`: komi 0.5,
`reward_head = true`, 40 generations, seed 1), trained on the fixed code
(batch-norm statistics per unroll step, capped games truncated).

### Training log

`runs/5x5-k05-rh2/log.csv`, selected generations:

```
generation  mean_length   loss_total    policy_k0     value_k0    reward_k1   arena_prev arena_rollout arena_random
         1       34.605      7.21315       3.0801       0.6201    0.0329439            0            0            0
         5       21.815      6.31661      2.86378     0.417855    0.0374021          0.4            0          0.1
        10        31.61      6.01088      2.59742     0.511012     0.029658          0.8          0.1          0.3
        15        35.25      5.69512      2.27549     0.583512    0.0210082         0.45         0.15         0.55
        20       34.635      5.45873      2.09021      0.61444    0.0208704         0.65         0.05         0.55
        25        37.75      5.34636       1.9774     0.641987    0.0194618         0.55         0.05         0.65
        30         38.5      5.23118      1.90428     0.645818    0.0174586          0.7            0          0.8
        35       36.555      5.05569      1.84875     0.612454    0.0189945          0.3          0.1          0.8
        40       39.415      4.97994      1.79868     0.624217    0.0174896         0.75          0.2         0.85
```

Against generation g − 5, the score is 0.5 or more in 30 of the 40
generations. `arena_random` first reaches 0.5 at generation 9 and is
0.55–0.90 from generation 20 on (run 3: 0.05–0.50). `arena_rollout` stays
between 0 and 0.20 (0.20 at generations 29 and 40). The losses are training
losses on batch statistics, and capped games are now masked past the cap,
so they are not directly comparable with run 3's.

More games hit the move cap than in run 3: 761 of the 8,000 self-play
games (`capped_games`, summed over generations) against 214. In
generations 31–40 the counts are 221 against 69, and the mean game length
is 36.9 against 29.9 moves. The count is computed the same way in both runs
(a game that did not end with two passes). I don't know why it rose. One
candidate is that a capped game no longer pays its result as a reward, so
nothing in the targets rewards ending a game before the cap rather than at
it, but I have not tested that.

### Matches

```
$ ./build/mzgo_match --a ckpt:runs/5x5-k05-rh2/checkpoints/gen_0040.mzn --b ckpt:runs/5x5-k05-rh2/checkpoints/gen_0000.mzn --games 40 --board-size 5 --komi 0.5 --simulations 100 --temperature-moves 2 --workers 2 --seed 1
A ckpt:runs/5x5-k05-rh2/checkpoints/gen_0040.mzn: 33 wins / B ckpt:runs/5x5-k05-rh2/checkpoints/gen_0000.mzn: 7 wins / 0 draws, scoreA 0.8250
$ ./build/mzgo_match --a ckpt:runs/5x5-k05-rh2/checkpoints/gen_0040.mzn --b ckpt:runs/5x5-k05-rh2/checkpoints/gen_0005.mzn --games 40 --board-size 5 --komi 0.5 --simulations 100 --temperature-moves 2 --workers 2 --seed 1
A ckpt:runs/5x5-k05-rh2/checkpoints/gen_0040.mzn: 40 wins / B ckpt:runs/5x5-k05-rh2/checkpoints/gen_0005.mzn: 0 wins / 0 draws, scoreA 1.0000
$ ./build/mzgo_match --a ckpt:runs/5x5-k05-rh2/checkpoints/gen_0040.mzn --b ckpt:runs/5x5-k05-rh2/checkpoints/gen_0020.mzn --games 40 --board-size 5 --komi 0.5 --simulations 100 --temperature-moves 2 --workers 2 --seed 1
A ckpt:runs/5x5-k05-rh2/checkpoints/gen_0040.mzn: 35 wins / B ckpt:runs/5x5-k05-rh2/checkpoints/gen_0020.mzn: 5 wins / 0 draws, scoreA 0.8750
$ ./build/mzgo_match --a ckpt:runs/5x5-k05-rh2/checkpoints/gen_0040.mzn --b ckpt:runs/5x5-k05-rh2/checkpoints/gen_0030.mzn --games 40 --board-size 5 --komi 0.5 --simulations 100 --temperature-moves 2 --workers 2 --seed 1
A ckpt:runs/5x5-k05-rh2/checkpoints/gen_0040.mzn: 30 wins / B ckpt:runs/5x5-k05-rh2/checkpoints/gen_0030.mzn: 10 wins / 0 draws, scoreA 0.7500
$ ./build/mzgo_match --a ckpt:runs/5x5-k05-rh2/checkpoints/gen_0040.mzn --b rollout:200 --games 40 --board-size 5 --komi 0.5 --simulations 100 --temperature-moves 2 --workers 2 --seed 1
A ckpt:runs/5x5-k05-rh2/checkpoints/gen_0040.mzn: 2 wins / B rollout:200: 38 wins / 0 draws, scoreA 0.0500
$ ./build/mzgo_match --a ckpt:runs/5x5-k05-rh2/checkpoints/gen_0040.mzn --b random --games 40 --board-size 5 --komi 0.5 --simulations 100 --temperature-moves 2 --workers 2 --seed 1
A ckpt:runs/5x5-k05-rh2/checkpoints/gen_0040.mzn: 33 wins / B random: 7 wins / 0 draws, scoreA 0.8250
```

Each 40-game match took 3–6 s. Against the outside bots, 200 more games
each, with another seed:

```
$ ./build/mzgo_match --a ckpt:runs/5x5-k05-rh2/checkpoints/gen_0040.mzn --b rollout:200 --games 200 --board-size 5 --komi 0.5 --simulations 100 --temperature-moves 2 --workers 2 --seed 2
A ckpt:runs/5x5-k05-rh2/checkpoints/gen_0040.mzn: 12 wins / B rollout:200: 188 wins / 0 draws, scoreA 0.0600
$ ./build/mzgo_match --a ckpt:runs/5x5-k05-rh2/checkpoints/gen_0040.mzn --b random --games 200 --board-size 5 --komi 0.5 --simulations 100 --temperature-moves 2 --workers 2 --seed 2
A ckpt:runs/5x5-k05-rh2/checkpoints/gen_0040.mzn: 158 wins / B random: 42 wins / 0 draws, scoreA 0.7900
$ ./build/mzgo_match --a random --b rollout:200 --games 40 --board-size 5 --komi 0.5 --workers 2 --seed 1
A random: 0 wins / B rollout:200: 40 wins / 0 draws, scoreA 0.0000
```

The rollout bot is a strong opponent at this size: it beat the random bot
in all 40 games.

By colour, from one-game matches with seeds 1–50 (with `--games 1`, A is
Black; the loop below swaps `--a` and `--b` for the White games, and the
rollout games use the same loop):

```
$ C=ckpt:runs/5x5-k05-rh2/checkpoints/gen_0040.mzn
$ for s in $(seq 1 50); do ./build/mzgo_match --a $C --b random --games 1 --board-size 5 --komi 0.5 --simulations 100 --temperature-moves 2 --workers 1 --seed $s; done   # and --a random --b $C
MuZero gen 40 as black vs random, seeds 1-50 (one game each): 47 of 50 won
MuZero gen 40 as white vs random, seeds 1-50 (one game each): 40 of 50 won
MuZero gen 40 vs rollout:200, seeds 1-50 (one game each): as black 5 of 50 won, as white 1 of 50 won
```

(The summary lines were printed by the loop, which added up the per-game
`wins` fields.) Black has the advantage at komi 0.5, as the self-play record
(57% for Black) also says.

### Playing the random bot again

The same driver as in run 3 (a scratch script, not in the repository: the
random side plays uniformly over legal points, and the game stops at 50
moves), on generation 40, six seeds as each colour:

```
$ for c in b w; do for s in 1 2 3 4 5 6; do echo "### color $c seed $s"; python3 vsrandom50.py runs/5x5-k05-rh2/checkpoints/gen_0040.mzn $c $s; done; done | grep -E "^###|^= "
### color b seed 1
= B+9.5
### color b seed 2
= B+7.5
### color b seed 3
= B+21.5
### color b seed 4
= B+7.5
### color b seed 5
= B+6.5
### color b seed 6
= B+17.5
### color w seed 1
= B+6.5
### color w seed 2
= W+9.5
### color w seed 3
= W+22.5
### color w seed 4
= W+5.5
### color w seed 5
= W+10.5
### color w seed 6
= W+17.5
```

MuZero won 11 of 12. The seed-3 game as Black, the seed of run 3's
eye-filling loss (the moves differ, since the network does):

```
MZ:B3 R:B2 MZ:C4 R:E1 MZ:A4 R:C1 MZ:E4 R:A5 MZ:D5 R:B1 MZ:B5 R:D4 MZ:D3 R:A3 MZ:D2 R:D1 MZ:C2 R:A1
MZ:C3 R:A2 MZ:B4 R:E2 MZ:D4 R:pass MZ:E3 R:E2 MZ:B2 R:A1 MZ:D1 R:C1 MZ:A2 R:pass MZ:B1 R:pass
MZ:C5 R:pass MZ:pass
= B+21.5
```

It still plays on while the random bot passes and invades, but here playing
on only added points. For every random-bot pass in the 12 games, the score
if MuZero had passed back, and what it played instead (replayed through
`mzgo_gtp`'s `final_score`):

```
MZ b seed 1: ply 26: pass-back score B+2.5, MZ played D5
MZ b seed 2: ply 26: pass-back score B+5.5, MZ played B1; ply 28: pass-back score B+5.5, MZ played D1
MZ b seed 3: ply 24: pass-back score B+5.5, MZ played E3; ply 32: pass-back score B+16.5, MZ played B1; ply 34: pass-back score B+21.5, MZ played C5; ply 36: pass-back score B+21.5, MZ played pass
MZ b seed 4: ply 30: pass-back score B+7.5, MZ played E3; ply 32: pass-back score B+7.5, MZ played pass
MZ b seed 5: no random-bot pass before the end
MZ b seed 6: ply 40: pass-back score B+14.5, MZ played D4; ply 42: pass-back score B+17.5, MZ played D5; ply 44: pass-back score B+17.5, MZ played pass
MZ w seed 1: ply 43: pass-back score B+6.5, MZ played pass
MZ w seed 2: ply 29: pass-back score W+6.5, MZ played E1; ply 31: pass-back score W+9.5, MZ played pass
MZ w seed 3: ply 31: pass-back score W+16.5, MZ played E3; ply 33: pass-back score W+16.5, MZ played B1; ply 37: pass-back score W+22.5, MZ played pass
MZ w seed 4: ply 27: pass-back score W+5.5, MZ played pass
MZ w seed 5: ply 33: pass-back score W+10.5, MZ played pass
MZ w seed 6: ply 35: pass-back score W+17.5, MZ played pass
```

In 20 of these 21 positions passing back would have won. MuZero passed in
8 and played on in 12, and every one of those games ended in a win.

**The one loss** (MuZero White, seed 1) came from somewhere else. The
scores after each ply show White ahead by 3.5 after ply 27. MuZero then
passed at ply 28, and the random bot's E5 turned the count to B+24.5:

```
... 26:MZ:E3=W+4.5 27:R:E1=W+3.5 28:MZ:pass=W+3.5 29:R:E5=B+24.5 30:MZ:C3=B+10.5 ...
```

The position MuZero passed in:

```
$ ./build/mzgo_analyze --checkpoint runs/5x5-k05-rh2/checkpoints/gen_0040.mzn --komi 0.5 --top 26 --moves "C1 C4 A5 E4 C2 C3 A2 B3 A4 D3 A3 B2 E3 E2 D5 D2 B5 B4 B1 D4 E5 C5 E5 D5 D1 E3 E1" | grep -vE "^\s+[A-E][1-5]\s+0 "
   A B C D E
 5 X X O O .  5
 4 X O O O O  4
 3 X O O O O  3
 2 X O X O O  2
 1 . X X X X  1
   A B C D E
White to move
27 moves played; 200 simulations
root value +0.861 (for White)

  move  visits   prior       Q
  pass     106   0.746  +0.735
    A1      94   0.254  +1.002

search depth: max 15, mean 5.86; simulations going beyond the real game's end (two passes): 36.5%

principal variation (imagined by the dynamics; reward, Q and value for White):
  ply  player  move   reward        Q    value
    1   White  pass   +0.014   +0.735   +0.721
    2   Black    E5   -0.002   +0.315   +0.317
    3   White    E4   -0.000   +0.369   +0.369
    ...
```

Both sides are in atari. White's 13 stones have one liberty, E5. Both
Black groups have one liberty, A1. White's A1 captures 10 stones; a White
pass lets Black's E5 capture 13. The search gives A1 the higher Q, but
the pass has three times the prior and gets more visits, and the move is
chosen by visits. In the imagined line, Black's E5 after White's pass
costs White only about 0.4 of value, and White is still rated +0.317
after losing 13 stones: the dynamics does not show the capture. (The game used 100 simulations; this analysis uses 200.)

### The search's numbers: rewards, Q and the untried pass

The `C3 pass` test, with the predicted rewards now printed:

```
$ ./build/mzgo_analyze --checkpoint runs/5x5-k05-rh2/checkpoints/gen_0040.mzn --moves "C3 pass" --simulations 200 --komi 0.5 --top 26
root value +0.186 (for Black)

  move  visits   prior       Q
    B3      68   0.324  +0.175
    E4      48   0.211  +0.187
    D4      39   0.151  +0.206
    B2      25   0.141  +0.149
    C4      20   0.066  +0.229
    B5       0   0.017  +0.000
    ...
  pass       0   0.004  +0.000
    ...

search depth: max 11, mean 5.09; simulations going beyond the real game's end (two passes): 0.0%

principal variation (imagined by the dynamics; reward, Q and value for Black):
  ply  player  move   reward        Q    value
    1   Black    B3   -0.001   +0.175   +0.176
    2   White    B2   +0.002   +0.213   +0.211
    3   Black    C4   -0.002   +0.198   +0.200
    4   White    A4   -0.003   +0.189   +0.191
    5   Black    E4   +0.002   +0.186   +0.184
    6   White    D5   +0.003   +0.167   +0.165
    7   Black    D3   +0.004   +0.140   +0.136
    8   White    B5   +0.002   +0.110   +0.108
    9   Black    B4   -0.001   +0.100   +0.101
   10   White    D4   +0.001   +0.098   +0.097
   11   Black    B3   -0.002   +0.054   +0.056
$ ./build/mzgo_analyze --checkpoint runs/5x5-k05-rh2/checkpoints/gen_0020.mzn --moves "C3 pass" --simulations 200 --komi 0.5 --top 26 | grep -E "root value|move  visits|pass|search depth"
root value +0.248 (for Black)
  move  visits   prior       Q
  pass       0   0.010  +0.000
search depth: max 12, mean 5.48; simulations going beyond the real game's end (two passes): 0.0%
$ ./build/mzgo_analyze --checkpoint runs/5x5-k05-rh2/checkpoints/gen_0040.mzn --moves "C3 pass" --simulations 800 --komi 0.5 --top 26 | grep -E "root value|move  visits|pass|search depth"
root value +0.168 (for Black)
  move  visits   prior       Q
  pass       0   0.004  +0.000
search depth: max 15, mean 6.25; simulations going beyond the real game's end (two passes): 0.0%
```

The rewards on ordinary moves are within ±0.004, where run 3 had a
non-final reward of +0.888. But the pass that wins by 24.5 is never
visited, even at 800 simulations. A White pass on move 2 is rare in
self-play (1 of the 80 recorded games, at generation 2), and the prior on
answering it with a pass stays near 0. An unvisited child is scored 0 on the
normalized scale, which is a loss. So the search has no reason to try it
(`src/mcts.cpp`, the PUCT score; see "The unvisited child" in chapter 05 of
[`algorithm-explained.md`](algorithm-explained.md)).

Run 3's seed-3 position after ply 30 (passing back wins by 4.5), on the new
network:

```
$ ./build/mzgo_analyze --checkpoint runs/5x5-k05-rh2/checkpoints/gen_0040.mzn --komi 0.5 --top 26 --moves "C3 B2 C2 E1 B1 C1 D2 A5 D3 D1 E3 D4 A2 A4 B4 C4 B5 A1 B3 E2 B1 E1 D1 E4 C1 E5 D5 C5 B2 pass" | grep -vE "^\s+[A-E][1-5]\s+0 "
root value +0.921 (for Black)

  move  visits   prior       Q
    E2     134   0.539  +0.972
  pass      37   0.214  +0.879
    A3      14   0.112  +0.740
    D5       8   0.075  +0.731
    A1       7   0.061  +0.749

search depth: max 18, mean 6.49; simulations going beyond the real game's end (two passes): 63.0%

principal variation (imagined by the dynamics; reward, Q and value for Black):
  ply  player  move   reward        Q    value
    1   Black    E2   -0.000   +0.972   +0.972
    2   White    A3   -0.010   +0.682   +0.693
    3   Black    D5   -0.001   +0.756   +0.757
    4   White    A1   +0.002   +0.529   +0.528
    5   Black  pass   -0.001   +0.570   +0.571
    6   White  pass   +0.653   +0.592   -0.062
    7   Black  pass   -0.086   -0.067   +0.019
    8   White  pass   +0.024   +0.025   +0.001
    9   Black  pass   -0.016   +0.005   +0.021
   10   White  pass   +0.244   +0.023   -0.221
   11   Black  pass   -0.313   -0.257   +0.057
   12   White  pass   +0.139   +0.083   -0.057
   13   Black  pass   -0.093   -0.078   +0.014
   14   White  pass   +0.059   +0.012   -0.047
```

Here nothing is above 1, and the pass (+0.879) is close to the best move,
E2, which captures the White stone on E1. Run 3 had A1 at +1.353
and the pass at +0.408. But look past ply 6, where the imagined game has
ended: the passes keep paying rewards (+0.244, −0.313, +0.139), where the
targets say 0.

The seed-3 game's first random-bot pass (ply 24, passing back wins by 5.5):

```
$ ./build/mzgo_analyze --checkpoint runs/5x5-k05-rh2/checkpoints/gen_0040.mzn --komi 0.5 --top 26 --moves "B3 B2 C4 E1 A4 C1 E4 A5 D5 B1 B5 D4 D3 A3 D2 D1 C2 A1 C3 A2 B4 E2 D4 pass" | grep -vE "^\s+[A-E][1-5]\s+0 "
root value +1.098 (for Black)

  move  visits   prior       Q
    E3     120   0.670  +1.053
  pass      41   0.134  +1.181
    C5      39   0.142  +1.149

search depth: max 12, mean 5.31; simulations going beyond the real game's end (two passes): 58.0%

principal variation (imagined by the dynamics; reward, Q and value for Black):
  ply  player  move   reward        Q    value
    1   Black    E3   +0.002   +1.053   +1.051
    2   White  pass   +0.022   +1.314   +1.292
    3   Black  pass   +0.976   +1.312   +0.336
    4   White  pass   +0.170   +0.368   +0.198
    5   Black  pass   +0.115   +0.294   +0.179
    6   White  pass   +0.037   +0.178   +0.140
    7   Black  pass   -0.062   +0.252   +0.314
    8   White    A5   -0.003   +0.359   +0.362
    9   Black  pass   -0.002   +0.373   +0.375
   10   White  pass   +0.383   +0.381   -0.002
```

Every Q is above 1. In the principal variation, ply 3 ends the imagined
game. Its reward is about right (+0.976 for a win), but the value after it
is +0.336 where the target is 0, so its Q is +1.312.

From the seed-6 game as Black (ply 40, passing back wins by 14.5):

```
$ ./build/mzgo_analyze --checkpoint runs/5x5-k05-rh2/checkpoints/gen_0040.mzn --komi 0.5 --top 26 --moves "B3 B5 C4 D5 A4 E3 E4 C5 B2 A2 D3 A1 D2 B4 D1 D4 E2 C3 C2 C1 E3 A5 C4 A3 B1 A4 C1 C3 pass C4 E5 B5 B4 D5 A2 A4 C4 A5 C5 pass" | grep -vE "^\s+[A-E][1-5]\s+0 "
root value +1.276 (for Black)

  move  visits   prior       Q
  pass      90   0.191  +1.453
    D4      62   0.366  +1.196
    A1      24   0.170  +1.116
    A3      18   0.143  +1.078
    C3       6   0.095  +0.664

search depth: max 14, mean 5.47; simulations going beyond the real game's end (two passes): 71.5%

principal variation (imagined by the dynamics; reward, Q and value for Black):
  ply  player  move   reward        Q    value
    1   Black  pass   +0.590   +1.453   +0.863
    2   White    D4   -0.020   +0.728   +0.748
    3   Black  pass   -0.004   +0.787   +0.791
    4   White    A3   -0.008   +0.263   +0.271
    5   Black  pass   -0.013   +0.285   +0.299
    6   White  pass   +0.630   +0.771   +0.141
    7   Black  pass   +0.102   +0.172   +0.070
    8   White  pass   +0.189   +0.118   -0.071
    9   Black  pass   -0.110   -0.126   -0.016
```

Here the pass is the real end of the game. The model gives it reward +0.590
and value after it +0.863, where the targets are +1 and 0. The sum,
+1.453, is the top Q, so in this position the inflation favours the right
move. In the line, White then plays D4 after a game that has already ended.

**How often.** To count, a script (not in the repository) ran `mzgo_analyze`
(200 simulations, `--top 26`) on every position just after a random-bot pass
in the 12 games above. For each it printed the root value, the largest Q among
visited root moves, the pass's visits and Q, and the reward and the value
after it on the first edge in the principal variation that ends the
imagined game (a pass after a pass). Values are for the side to move.
Generation 40:

```
MZ b seed 1 ply 26: root +0.968, max Q +1.049 (pass), pass 55 visits Q +1.049; ending edge at PV ply 6: reward +0.616, value after -0.016
MZ b seed 2 ply 26: root +0.918, max Q +0.989 (A5), pass 18 visits Q +0.674; ending edge at PV ply 4: reward +0.953, value after +0.264
MZ b seed 2 ply 28: root +0.997, max Q +1.054 (D1), pass 55 visits Q +0.952; ending edge at PV ply 3: reward +0.965, value after +0.319
MZ b seed 3 ply 24: root +1.098, max Q +1.181 (pass), pass 41 visits Q +1.181; ending edge at PV ply 3: reward +0.976, value after +0.336
MZ b seed 3 ply 32: root +1.131, max Q +1.262 (C5), pass 24 visits Q +1.115; ending edge at PV ply 4: reward +0.907, value after +0.158
MZ b seed 3 ply 34: root +1.199, max Q +1.269 (E5), pass 16 visits Q +0.919; ending edge at PV ply 3: reward +0.954, value after +0.952
MZ b seed 3 ply 36: root +1.369, max Q +1.503 (E5), pass 53 visits Q +1.298; ending edge at PV ply 1: reward +0.434, value after +0.864
MZ b seed 4 ply 30: root +1.144, max Q +1.264 (D5), pass 25 visits Q +0.970; ending edge at PV ply 3: reward +0.951, value after +0.889
MZ b seed 4 ply 32: root +1.240, max Q +1.410 (D5), pass 61 visits Q +1.231; ending edge at PV ply 3: reward +0.952, value after +0.904
MZ b seed 6 ply 40: root +1.276, max Q +1.453 (pass), pass 90 visits Q +1.453; ending edge at PV ply 1: reward +0.590, value after +0.863
MZ b seed 6 ply 42: root +1.482, max Q +1.689 (pass), pass 82 visits Q +1.689; ending edge at PV ply 1: reward +0.686, value after +1.002
MZ b seed 6 ply 44: root +1.561, max Q +1.697 (pass), pass 98 visits Q +1.697; ending edge at PV ply 1: reward +0.792, value after +0.905
MZ w seed 1 ply 43: root +0.348, max Q +0.464 (B2), pass 73 visits Q +0.233; ending edge at PV ply 1: reward -0.003, value after +0.236
MZ w seed 2 ply 29: root +1.175, max Q +1.238 (pass), pass 95 visits Q +1.238; ending edge at PV ply 1: reward +0.227, value after +1.011
MZ w seed 2 ply 31: root +1.199, max Q +1.282 (pass), pass 110 visits Q +1.282; ending edge at PV ply 1: reward +0.351, value after +0.931
MZ w seed 3 ply 31: root +1.503, max Q +1.594 (C5), pass 32 visits Q +1.485; ending edge at PV ply 3: reward +0.957, value after +1.226
MZ w seed 3 ply 33: root +1.452, max Q +1.519 (C5), pass 57 visits Q +1.511; ending edge at PV ply 3: reward +0.970, value after +0.882
MZ w seed 3 ply 37: root +1.431, max Q +1.494 (pass), pass 83 visits Q +1.494; ending edge at PV ply 1: reward +0.487, value after +1.007
MZ w seed 4 ply 27: root +1.136, max Q +1.255 (pass), pass 88 visits Q +1.255; ending edge at PV ply 1: reward +0.378, value after +0.877
MZ w seed 5 ply 33: root +0.998, max Q +1.051 (pass), pass 90 visits Q +1.051; ending edge at PV ply 1: reward +0.410, value after +0.640
MZ w seed 6 ply 35: root +1.135, max Q +1.211 (pass), pass 43 visits Q +1.211; ending edge at PV ply 5: reward +0.963, value after +0.139
19 of 21 positions have a root value or visited-move Q above 1
```

Generation 20 on the same positions: `20 of 21 positions have a root value
or visited-move Q above 1`. The values above 1 are not confined to games
against the random bot. The same count on self-play positions of
generations 31–40 (from the SGFs, generation 40's network):

```
$ python3 qscan_sgf.py runs/5x5-k05-rh2/checkpoints/gen_0040.mzn runs/5x5-k05-rh2/sgf/gen_003[1-9]_*.sgf runs/5x5-k05-rh2/sgf/gen_0040_*.sgf
after a pass: 17 of 41 positions above 1 (largest root value or visited Q +1.766)
ply 20: 0 of 20 positions above 1 (largest root value or visited Q +0.944)
```

So in the middle of a game Q stays below 1. It is near the end, after a
pass, that the model's picture of the ending goes wrong. In 9 of the
positions the principal variation starts with the real pass back, which
wins. There the reward on that pass is +0.23 to +0.79 instead of +1, and
the value after it +0.64 to +1.01 instead of 0. Over all 21, the value after
the first imagined ending ranges from −0.02 to +1.23; the target is 0
every time. The two errors add, often to more than 1. In the one losing
position (White, seed 1, ply 43, B+6.5 on the board), the PV also starts
with the pass, and its reward is −0.003 where the target is −1. The search
no longer rates the winning pass low: it is the top-Q move in 11 of the 21
positions. In the two run-3 positions above, it was rated about 1 below
the moves that played on.

**A dead-stone test that did not test anything.** Run 3 left open the
hypothesis that the model would think a pass wins when invading stones
would make it lose under Tromp-Taylor. I built a position for it. Black owns
columns A–C and White D–E. In one version a White stone on A1 sits in
Black's area, so Black loses if it passes (W+4.5). In the other version the
stone is on E5 instead, and passing wins (B+4.5). White has just passed in
both:

```
$ ./build/mzgo_analyze --checkpoint runs/5x5-k05-rh2/checkpoints/gen_0040.mzn --komi 0.5 --top 26 --moves "C3 D3 C2 D2 C4 D4 C1 D1 C5 D5 B5 A1 B4 pass" | ...
root value +0.444 (for Black)
  move  visits   prior       Q
    E4     126   0.244  +0.513
    E3      46   0.327  +0.338
    B2      18   0.130  +0.340
    E2      10   0.095  +0.262
  pass       0   0.029  +0.000
$ ./build/mzgo_analyze --checkpoint runs/5x5-k05-rh2/checkpoints/gen_0040.mzn --komi 0.5 --top 26 --moves "C3 D3 C2 D2 C4 D4 C1 D1 C5 D5 B5 E5 B4 pass" | ...
root value +0.348 (for Black)
  move  visits   prior       Q
    E4      88   0.186  +0.427
    E2      38   0.237  +0.281
    B3      34   0.111  +0.388
    E3      33   0.229  +0.259
    B2       6   0.097  -0.000
    B1       1   0.044  -0.436
  pass       0   0.018  +0.000
```

The pass gets no visits in either version, so the test says nothing about
the hypothesis. These positions are unlike anything in self-play, which
also shows: the search invests in E4, inside White's area, in both.

### The probes

The same probe commands as for run 3 (20 rollout-bot games, K = 5, seed 1).
The probes now pass each imagined step's depth, so step k uses step k's
batch-norm statistics. That is a change in what they compute, not only in
the network.

```
$ ./build/mzgo_latent_probe --checkpoint runs/5x5-k05-rh2/checkpoints/gen_0010.mzn --games 20 --k 5 --workers 2 --seed 1 --komi 0.5
   k   samples   mean |value error|   mean policy TV distance
   0       703               0.0000                    0.0000
   1       683               0.4391                    0.2478
   2       663               0.3183                    0.2337
   3       643               0.4997                    0.2995
   4       623               0.4530                    0.3095
   5       603               0.5895                    0.3440
$ ./build/mzgo_latent_probe --checkpoint runs/5x5-k05-rh2/checkpoints/gen_0040.mzn --games 20 --k 5 --workers 2 --seed 1 --komi 0.5
   k   samples   mean |value error|   mean policy TV distance
   0       703               0.0000                    0.0000
   1       683               0.3715                    0.3140
   2       663               0.2677                    0.3038
   3       643               0.4268                    0.3785
   4       623               0.3684                    0.3918
   5       603               0.4877                    0.4290
$ ./build/mzgo_rules_probe --checkpoint runs/5x5-k05-rh2/checkpoints/gen_0010.mzn --games 20 --k 5 --workers 2 --seed 1 --komi 0.5
   k   samples    illegal   occupied    superko   captured
   0       703     0.1858     0.1360     0.0017     0.0000
   1       683     0.2383     0.1707     0.0008     0.0068
   2       663     0.2469     0.1894     0.0009     0.0144
   3       643     0.2830     0.2213     0.0009     0.0209
   4       623     0.2998     0.2467     0.0009     0.0271
   5       603     0.3324     0.2764     0.0010     0.0325
$ ./build/mzgo_rules_probe --checkpoint runs/5x5-k05-rh2/checkpoints/gen_0040.mzn --games 20 --k 5 --workers 2 --seed 1 --komi 0.5
   k   samples    illegal   occupied    superko   captured
   0       703     0.0755     0.0174     0.0020     0.0000
   1       683     0.1155     0.0377     0.0006     0.0068
   2       663     0.1431     0.0705     0.0006     0.0155
   3       643     0.1868     0.1094     0.0008     0.0295
   4       623     0.2146     0.1473     0.0008     0.0416
   5       603     0.2546     0.1846     0.0010     0.0531
$ ./build/mzgo_board_probe --checkpoint runs/5x5-k05-rh2/checkpoints/gen_0010.mzn --games 20 --k 5 --workers 2 --seed 1 --komi 0.5
   k     points   accuracy    changed   changed acc.
   0      18075     0.7054          0              -
   1      17575     0.4487        855         0.3158
   2      17075     0.4391       1568         0.3559
   3      16575     0.4364       2241         0.3182
   4      16075     0.4275       2753         0.3244
   5      15575     0.4204       3243         0.3198
$ ./build/mzgo_board_probe --checkpoint runs/5x5-k05-rh2/checkpoints/gen_0040.mzn --games 20 --k 5 --workers 2 --seed 1 --komi 0.5
   k     points   accuracy    changed   changed acc.
   0      18075     0.7488          0              -
   1      17575     0.4675        855         0.3871
   2      17075     0.4448       1568         0.3948
   3      16575     0.4326       2241         0.3677
   4      16075     0.4289       2753         0.3727
   5      15575     0.4259       3243         0.3534
```

Each probe took 1 s or less. Compared with run 3's generation 40:

- **Latent probe.** The value errors are smaller (k = 1–5: 0.37, 0.27,
  0.43, 0.37, 0.49, against 0.51, 0.35, 0.56, 0.45, 0.60). The policy drift
  is a little smaller (0.43 at k = 5 against 0.51). The odd/even
  alternation is still there, so it was not a batch-norm artefact.
- **Rules probe.** Illegal mass on the real position is higher than run 3's
  (0.076 against 0.045 at k = 0) and a little lower at depth (0.255
  against 0.278 at k = 5). As before, it grows with depth, mostly on
  occupied points, and the "captured" column grows to 0.05.
- **Board probe.** The readout still drops after one imagined step (0.75
  on fresh states, 0.43–0.47 at k ≥ 1, against run 3's 0.38) and stays flat
  after it. The same caveats apply (in-sample, no majority-class baseline).

## Pre-fix diagnoses, revisited

What runs 1–3 concluded, and what run 4 says about it:

- **"The search's values go above 1" (run 3).** Still true after the fix,
  in positions just after a pass. The notes added after the final review
  said the review "traced most of it to the BN train/inference mismatch".
  That was too strong. The fix removed the reward on ordinary moves and
  the large gap between the pass and playing on, but not the values above
  1.
- **"Somewhere past the real end, imagined steps are paying out reward
  again, or non-final edges carry reward" (run 3's partial diagnosis).**
  Both halves were possible, and they had different causes. Reward on
  non-final edges was the batch-norm defect: the review measured +0.888 on
  one, and now ordinary moves get within ±0.03. The other half is still
  there after the fix, and is now measured. Past the end, imagined passes
  pay rewards of up to +0.63 (and down to −0.31). The value after the
  ending edge is far from 0 (up to +1.23), and the ending edge's reward is
  often well short of +1.
- **"It rates the pass that would end a won game at about +0.4 and playing
  on at about +1.4; it keeps playing until it fills its own last eye"
  (run 3).** Not reproduced after the fix. The pass is now rated close to
  or above playing on, and in 12 games against the random bot no won
  position was lost by playing on.
- **"What the reward-head run has learned is how to beat opponents that
  play like itself" (across runs).** Wrong after the fix, as far as the
  random bot goes: 0.79–0.825. It still holds for the rollout bot
  (0.05–0.06).
- **The dead-stones hypothesis (run 3).** Still untested. The position built
  for it in run 4 gives the pass no visits, so it measures nothing.
- **Run 1 (komi 24.5 cold start).** The reasoning does not depend on batch
  norm: the Black value target is −1 in every game, so no network can
  separate Black's moves by value. I expect it to hold, but the run has not
  been repeated on the fixed code.
- **Run 2 (no reward head).** With no reward head, nothing is ever paid on
  the ending edge, so the argument does not depend on batch norm either.
  Also not re-run. The cross-run matches below (39–1 at generation 12)
  compare two pre-fix networks.

## The probes (runs 3 and 1, before the fix)

> **Note.** Measured before the per-depth batch-norm fix; the probes then
> applied one blended set of statistics at every depth. Run 4's probes are
> above.

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
67–77% (three classes per point: empty, own, opponent). These accuracies
are in-sample: the readout is fitted and scored on the same positions. Applied to the
state one imagined step later, it drops to 36–42%, and it stays there at
deeper steps. So the dynamics' states are not in the representation's
format, at least not linearly. A one-step drop that does not keep falling
looks like a change of format, not a gradual loss of information. The probe
cannot tell those two apart, and I did not measure a majority-class
baseline.

## Across runs: the same generation, with and without the reward head

> **Note.** Both runs measured here are from before the per-depth
> batch-norm fix, and neither has been re-run without the reward head.

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
opponents that play like itself. (Run 4, after the fix, beats the random
bot; see above.)

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
50-move cap). Runs 1–3 counted a simulation whose leaf *was* the second
pass as past the end too; run 4 counts only paths that go on beyond the
double pass, so its numbers are not directly comparable with the others:

```
5x5-s1      g1: past_end 0.030 capped 34 | g10: 0.666 capped 0 | g20: 0.204 capped 1 | g40: 0.187 capped 1 | g60: 0.182 capped 28
5x5-k05     g1: past_end 0.030 capped 34 | g5: 0.593 capped 0  | g12: 0.589 capped 0
5x5-k05-rh  g1: past_end 0.030 capped 17 | g5: 0.620 capped 0  | g10: 0.444 capped 6 | g20: 0.343 capped 9 | g40: 0.184 capped 1
5x5-k05-rh2 g1: past_end 0.029 capped 17 | g5: 0.492 capped 0  | g10: 0.379 capped 6 | g20: 0.199 capped 21 | g40: 0.121 capped 36
```

With the reward head, the fraction of self-play search spent past the end
falls steadily, from 0.62 at generation 5 to 0.18 at 40 in run 3, and from
0.49 to 0.12 in run 4. Without it, the fraction stays near 0.6 through
generation 12. In the analyzed positions just after a random-bot pass,
though, it is much higher: 0.72 and 0.86 in run 3, and 0.58–0.72 in the
three such positions shown for run 4.
Those are the positions where the values above 1 appear.

## What is not yet known

- **Why the value after the ending edge is not 0.** Every sample that
  reaches the end of a game trains the state after the last pass toward
  value 0, with random actions past it. Run 4 shows the model has not
  learned that for positions just after a pass, in self-play as well as
  against the random bot. I have not measured whether there is too little
  such data, or whether the states the search reaches after an imagined
  pass look unlike the ones training builds with random padding actions.
- **Why more games hit the move cap after the fix** (761 against 214 of
  8,000).
- **Whether the dead-stones hypothesis holds** (see run 3; run 4's attempt
  did not test it).
- **Whether the no-reward-head run would have recovered.** It was stopped
  at generation 12, while it was still improving against itself (0.725
  against its generation 0). It has not been re-run on the fixed code.
- **Seeds.** Every run is a single seed (seed 1). The non-monotone arena
  scores (run 3's generation 12 lost to generation 0; run 4's `arena_prev`
  is below 0.5 in 10 generations) could be noise or could be real.
  Nothing here separates the two.
- **Komi 24.5 with the reward head.** The 5×5 preset now has the reward
  head on but still komi 24.5, and that pair has never been run. Run 1's
  failure did not depend on the reward head, so the reward head should not
  fix it. With komi 24.5, Black still has to own all 25 points to win.
- **Longer training.** 40 generations is about 11 minutes on this machine.
  The learning-rate schedule had reached 0.0045 by then (`lr` column).

## 9×9: `9x9-s1` (stopped at generation 94)

`runs/9x9-s1` was the 9×9 preset on the fixed code (komi 7.5, 6 + 6
blocks × 64 filters, `reward_head = true`, 200 games and 200 simulations
per generation, 400 training steps of batch 256, K = 5, a replay window of
2500 games), started as

```
./build/mzgo_train --config configs/9x9.cfg --run runs/9x9-s1 --set workers=6 --set train_threads=6
```

It was stopped at generation 94 because it never learned to beat the
random bot: over the last 30 generations it scored 0.05–0.25 against it
(20 games each), and 0–0.1 against `rollout:200`. The last lines of the log:

```
gen 93: selfplay 250.193s (200 games, mean length 94.735), train 179.197s (400 steps, loss 5.67254, lr 0.00487188), arena vs gen 88 0.4, rollout 0.05, random 0.15
gen 94: selfplay 214.731s (200 games, mean length 81.69), train 179.849s (400 steps, loss 5.62784, lr 0.00479767), arena vs gen 89 0.9, rollout 0.1, random 0.25
```

### Diagnosis: the network cannot see that the opponent just passed

Self-play had settled into short, pass-heavy games with nearly identical
openings. The random bot never passes and fills the board, which leads to
positions self-play never visits. Within self-play, the search often
failed to play the winning pass that ends the game after the opponent
passed.

The cause is in the input. The representation function sees AlphaZero's
planes: 8 history boards × (own, opponent) plus 2 colour planes. A pass
shows up only as two identical consecutive history boards. A probe that
starts the model k imagined steps before a real game end (generation 86,
self-play games) measured:

| k | reward × sign of the outcome | value after the end (should be 0) |
|---|---|---|
| 1 | 0.2 of the target | 0.77 |
| ≥ 2 | about 0.7 of the target | about 0.06 |

At k = 1 the real position is the one right after the opponent's pass, so
the network has to read "the last move was a pass" from the input. At
k ≥ 2 the pass happens inside the dynamics function, which knows the
action. Training agrees: the `reward_k1` loss was 4–8 times the k ≥ 2
losses. The same pattern was measured on 5×5 (run 4).

The paper's Appendix E describes the same input for Go, with no pass
feature, so this is not a deviation from the paper. KataGo adds explicit
"previous move was a pass" features for this reason.

## Run 5: `5x5-k05-pp`, the pass plane

The fix is an optional input plane, `pass_plane = true`: one extra plane
after the colour planes, all ones if the last move was a pass. It is off by
default, so the paper's input stays the default and older checkpoints
still load (checkpoint format version 3; version 2 loads with the plane
off). Run 5 is run 4's config with only `pass_plane = true` added, same
seed, 40 generations.

Measured at generation 40 (matches: 200 games, 200 simulations,
`--temperature-moves 2`, komi 0.5, seed 7):

| | Run 4 (no plane) | Run 5 (pass plane) |
|---|---|---|
| Reward at k = 1, fraction of the target | 0.35 | 0.77 |
| Value after the end at k = 1 | 0.52 | 0.19 |
| Reward at k = 2, fraction of the target | 0.87 | 0.82 |
| `reward_k1` loss at generation 33 | 0.019 | 0.005 |
| Score against the random bot | 0.85 | 0.79 |
| Score against `rollout:200` | 0.125 | 0.065 |

Head to head, run 5 beat run 4 148–52 (0.74).

The plane fixes the blind spot: the ending one step ahead is now learned
almost as well as the ending two steps ahead. Run 5 is clearly stronger
at self-play Go. It is not better against the two bots, which never pass
and lead to crowded, capture-heavy positions. That gap is about 1.6–2
standard errors from a single training seed, so it may be noise. If it is
real, the likely cause is narrow self-play (too little exploration), not
passes.

## 9×9: `9x9-pp-s1` (stopped at generation 56)

`runs/9x9-pp-s1` is the 9×9 preset with only `pass_plane = true` added,
started as

```
./build/mzgo_train --config configs/9x9.cfg --run runs/9x9-pp-s1 --set pass_plane=true --set workers=6 --set train_threads=6
```

The plane did its job: `reward_k1` fell to the level of the deeper reward
losses (about 0.003). The run still stalled like `9x9-s1`. Over generations
41–53 it averaged 0.02 against the random bot (`9x9-s1`: 0.07; 20 games
each) and 0.06 against `rollout:200`. Self-play games first collapsed to
13–20 moves (generations 5–20, nearly all White wins by komi after a
pass-pass ending), then recovered to 40–60 moves, the same path the
earlier run took. It was stopped at generation 56. So the pass blind spot
was real but is not what keeps 9×9 from beating the random bot.

## Runs 6 and 7: league play on 5×5

The random bot never passes and drops stones everywhere, so it leads to
positions self-play never reaches. League play
(`docs/superpowers/specs/2026-11-20-league-design.md`) gives part of each
generation's games to opponents that do not play like the current network.
The league has `league_size` slots, which all start as the random bot. Half
the games (`league_fraction = 0.5`) are league games. A member is replaced
by the network that beat it (score ≥ 0.6 over at least 20 of the
generation's league games). A random-bot member must be beaten in two
generations in a row. In league games only the learner's own moves train
the policy.

Both runs use run 5's config (pass plane on) plus `league_size = 3`.
Run 7 adds `league_value = false`, so league games give no value targets.
They still train policy (the learner's moves), reward and dynamics.

| 200 games each (200 simulations, `--temperature-moves 2`, komi 0.5, seed 7) | Run 5 (no league) | Run 6 (league) | Run 7 (league, no value) |
|---|---|---|---|
| Generation 40 vs the random bot | 0.79 | 0.695 | **0.865** |
| Generation 40 vs `rollout:200` | 0.065 | 0.065 | **0.16** |
| Generation 40 vs run 5's generation 40 | | 0.00 | 0.48 |
| Generation 40 vs run 6's generation 40 | | | 0.725 |
| Generation 80 vs the random bot | 0.90 | 0.745 | |
| Generation 80 vs `rollout:200` | 0.135 | 0.04 | |
| Generation 80 vs run 5's generation 80 | | 0.055 | |

Run 6 is clearly worse. The likely reason: a league game trains the value
toward its outcome, and against the random bot that is the outcome against
a random opponent. The search then uses those values as if the opponent
played well. Run 7 tests that explanation. With the value targets removed,
the league matches run 5 head-to-head, and it is better against both bots.
The gain against `rollout:200`, which the league never plays, is about 3
standard errors. All three are single seeds.

The two-wins rule for random-bot members filtered out single lucky
generations about ten times in run 6. Run 6's random slots were all
replaced by generation 42, run 7's by about generation 22. After that both
leagues hold recent networks, and a member is replaced every one or two
generations.

## 9×9: `9x9-lg-s1` (stopped at generation 8, then resumed)

`runs/9x9-lg-s1` is the 9×9 preset with the pass plane and run 7's league
(`league_size = 3`, `league_value = false`), started as

```
./build/mzgo_train --config configs/9x9.cfg --run runs/9x9-lg-s1 --set pass_plane=true --set league_size=3 --set league_value=false --set workers=6 --set train_threads=6
```

It was stopped at generation 8. Self-play games had collapsed to 5.9 moves,
and newer networks lost to older ones (0.1–0.25 against the network five
generations back). In the saved league games the learner passed 54 of its
about 65 moves while the random bot filled the board. Those are the
learner's own moves, so they become policy targets that reinforce passing.
This was judged too early: both earlier 9×9 runs also collapsed in their
first 20 generations and recovered later. The run was resumed as the
league-only comparison for the next run; see "`9x9-lg-s1` resumed" below.

## 9×9: `9x9-nep-s1`, league plus no early passes

`runs/9x9-nep-s1` is `9x9-lg-s1` plus `no_pass_before = 30`: in self-play
and league games, a searching player may not pass before move 30 unless
pass is its only legal move. The random bot, the arena and the match tools
are unaffected. Started as

```
./build/mzgo_train --config configs/9x9.cfg --run runs/9x9-nep-s1 --set pass_plane=true --set league_size=3 --set league_value=false --set no_pass_before=30 --set workers=6 --set train_threads=6
```

It ran all 150 generations, about 19 hours on 6 cores. This is the first 9×9
run that learned to beat the random bot. Ten-generation means of the arena
(20 games each):

| Generations | Self-play mean length | vs random | vs `rollout:200` |
|---|---|---|---|
| 1–10 | 38 | 0.00 | 0.03 |
| 21–30 | 34 | 0.01 | 0.10 |
| 41–50 | 69 | 0.11 | 0.15 |
| 61–70 | 87 | 0.24 | 0.12 |
| 81–90 | 98 | 0.45 | 0.15 |
| 101–110 | 99 | 0.52 | 0.20 |
| 121–130 | 98 | 0.65 | 0.28 |
| 141–150 | 95 | 0.74 | 0.27 |

Generation 150 in 200-game matches (200 simulations, `--temperature-moves 4`,
komi 7.5, seed 7):

| Opponent | Score |
|---|---|
| The random bot | 0.73 (146–54) |
| `rollout:200` | 0.30 (60–140) |
| `9x9-s1` generation 94 | 0.865 (173–27) |

For the first 30 generations self-play games ended just after move 30,
with a pass at the first legal chance. They grew to 95–100 moves by about
generation 80. The random-bot slots stayed in the league until generations
89–96, about when the arena score against the random bot passed 0.45; after
that the league held recent networks. The score was still rising at
generation 150.

Two things changed at once compared with `9x9-pp-s1`, the league and the
no-early-pass rule, so this run does not say which one was needed. The
resumed `9x9-lg-s1` (league only) answers that. `no_pass_before` is a
training-only deviation from the paper. It uses no Go knowledge beyond
"pass can end the game", so it is a minimum-game-length rule, the same kind
of exploration aid as root noise and temperature. A stronger rule ("no pass
while a non-eye-filling move remains") was considered and rejected, because
it builds hand-written Go knowledge (what an eye is) into the player.

## 9×9: `9x9-lg-s1` resumed, league only (generations 8–450)

Was the no-early-pass rule needed, or was `9x9-lg-s1` stopped too early?
It was resumed from generation 8 with the same settings (`no_pass_before =
0`), to 150 generations, then extended 50 generations at a time:

```
./build/mzgo_train --resume runs/9x9-lg-s1 --set workers=6 --set train_threads=6
./build/mzgo_train --resume runs/9x9-lg-s1 --set generations=200 --set workers=6 --set train_threads=6
./build/mzgo_train --resume runs/9x9-lg-s1 --set generations=250 --set league_network_wins=2 --set league_reset_on_replace=true --set workers=6 --set train_threads=6
./build/mzgo_train --resume runs/9x9-lg-s1 --set generations=300 --set workers=6 --set train_threads=6
./build/mzgo_train --resume runs/9x9-lg-s1 --set generations=350 --set workers=6 --set train_threads=6
./build/mzgo_train --resume runs/9x9-lg-s1 --set generations=450 --set lr_init=0.002 --set lr_decay_rate=1 --set workers=6 --set train_threads=6
```

Generations 1–200 used the league rules of `9x9-nep-s1` (a beaten network
member is replaced at once). Generations 201 on used the two rules added
after generation 200 (below): a network member must also be beaten in two
generations in a row, and any replacement clears every slot's strike.
Ten-generation means of the arena (20 games each); the last column counts
league replacements:

| Generations | Self-play mean length | vs random | vs `rollout:200` | Replacements |
|---|---|---|---|---|
| 1–10 | 20 | 0.00 | 0.01 | 0 |
| 11–20 | 5 | 0.00 | 0.00 | 0 |
| 41–50 | 7 | 0.00 | 0.00 | 0 |
| 61–70 | 36 | 0.01 | 0.01 | 0 |
| 81–90 | 62 | 0.08 | 0.01 | 0 |
| 101–110 | 65 | 0.26 | 0.06 | 0 |
| 121–130 | 85 | 0.54 | 0.14 | 0 |
| 141–150 | 95 | 0.51 | 0.22 | 4 |
| 161–170 | 97 | 0.61 | 0.26 | 5 |
| 181–190 | 95 | 0.66 | 0.28 | 8 |
| 191–200 | 93 | 0.75 | 0.35 | 7 |
| 201–210 | 90 | 0.71 | 0.35 | 1 |
| 221–230 | 90 | 0.80 | 0.42 | 2 |
| 241–250 | 92 | 0.78 | 0.49 | 2 |
| 251–260 | 91 | 0.79 | 0.40 | 4 |
| 271–280 | 95 | 0.82 | 0.55 | 4 |
| 291–300 | 98 | 0.82 | 0.47 | 2 |
| 321–330 | 98 | 0.87 | 0.51 | 4 |
| 341–350 | 98 | 0.89 | 0.55 | 4 |
| 351–360 | 98 | 0.83 | 0.61 | 2 |
| 381–390 | 97 | 0.86 | 0.63 | 2 |
| 401–410 | 96 | 0.96 | 0.73 | 4 |
| 421–430 | 96 | 0.94 | 0.67 | 5 |
| 441–450 | 96 | 0.93 | 0.68 | 3 |

200-game matches (same settings as for `9x9-nep-s1`):

| Generation | Opponent | Score |
|---|---|---|
| 150 | The random bot | 0.65 (130–70) |
| 150 | `rollout:200` | 0.275 (55–145) |
| 150 | `9x9-s1` generation 94 | 0.855 (171–29) |
| 150 | `9x9-nep-s1` generation 150 | 0.215 (43–157) |
| 200 | The random bot | 0.815 (163–37) |
| 200 | `rollout:200` | 0.40 (80–120) |
| 200 | `9x9-lg-s1` generation 150 | 0.925 (185–15) |
| 200 | `9x9-nep-s1` generation 150 | 0.43 (86–114) |
| 250 | The random bot | 0.76 (152–48) |
| 250 | `rollout:200` | 0.465 (93–107) |
| 250 | `9x9-lg-s1` generation 200 | 0.995 (199–1) |
| 250 | `9x9-nep-s1` generation 150 | 0.91 (182–18) |
| 300 | The random bot | 0.83 (166–34) |
| 300 | `rollout:200` | 0.455 (91–109) |
| 300 | `9x9-lg-s1` generation 250 | 1.00 (200–0) |
| 300 | `9x9-nep-s1` generation 150 | 0.985 (197–3) |
| 350 | The random bot | 0.95 (190–10) |
| 350 | `rollout:200` | 0.56 (112–88) |
| 450 | The random bot | 0.935 (187–13) |
| 450 | `rollout:200` | 0.755 (151–49) |
| 450 | `9x9-lg-s1` generation 400 | 0.50 (100–100) |
| 450 | `9x9-lg-s1` generation 350 | 0.50 (100–100) |
| 450 | `9x9-lg-s1` generation 300 | 0.98 (196–4) |
| 450 | `9x9-nep-s1` generation 150 | 0.985 (197–3) |

The league alone escaped the early-pass trap. Self-play games stayed at
5–8 moves for generations 11–50, then grew to about 95 by generation 140,
and the score against the random bot rose once games had passed about 60
moves (generation 90 on), as in `9x9-nep-s1`. The first random slot was replaced at generation 132
(`9x9-nep-s1`: 89) and the last at about 150. So `no_pass_before` was not
needed, but it shortened the stall by about 50 generations: at generation
150 the no-early-pass network wins 157 of 200 games against this one.

After generation 200 the replacement rule changed (Mark's choice, for noise
rejection): `league_network_wins = 2` and `league_reset_on_replace = true`.
Replacements fell from 32 in generations 151–200 (most generations replaced
someone, so members rarely lasted more than a few generations) to 11 in
generations 201–250. The score against `rollout:200` kept rising, to 0.49 in
the arena over generations 241–250, the best of any 9×9 run so far; against
the random bot it levelled off near 0.8. In the 200-game matches,
generation 250 beats generation 200 in 199 of 200 games and
`9x9-nep-s1`'s generation 150 in 182, the largest generation-to-generation
gain seen on 9×9, and comes within 0.035 of an even score against
`rollout:200`. Against the random bot it scores no better than generation
200 (0.76 against 0.815, a gap within about two standard errors at 200
games): it gets stronger against opponents that play Go while still losing
about a quarter of its games to random moves. Since the rule
change and the extra generations come together, this does not show that
the new rule caused the gain, only that it did not stop it.

Generations 251–300 kept the same settings. Generation 300 beats generation
250 in all 200 games and `9x9-nep-s1`'s generation 150 in 197; against the
random bot it recovers to 0.83, the best so far. Against `rollout:200` it
has held at about 0.46 since generation 250 (0.40–0.55 in the arena). The
league replaced a member 14 times in these 50 generations. The learning
rate is now low: `lr_decay_steps = 60000` with `lr_decay_rate = 0.1`
(the 9×9 preset) takes it from 0.02 to 0.0002 at generation 300 (400 steps a
generation), which may be why the rollout score has stopped rising.

Generations 301–350 kept the decaying rate (0.0002 down to 0.00009). In
the 200-game matches generation 350 scores 0.95 against the random bot and
0.56 against `rollout:200`. A separate 100-game look at generation 350
against `rollout:200` (a scratch program; 0.53 there) found that the losses
are lost fights, not passes: in its 47 losses the rollout player captured
21 of the network's stones on average (1 in its wins), 26 losses had 10 or
more captured, and at the network's first pass it was already about 10
points behind; only 4 losses had it passing while ahead. It was weaker as
Black (20–30) than as White (33–17). A likely cause is reading: a capturing
race longer than `unroll_steps = 5` is read through imagined states that
training never checked, while the rollout player plays every fight out on
the real board.

Generations 351–450 held the learning rate at 0.002 (`lr_init = 0.002`,
`lr_decay_rate = 1`). The training loss fell at once (3.58 over
generations 341–350, 3.38 over 351–360, 3.25 over 441–450), and the arena
score against `rollout:200` rose from about 0.55 to about 0.68. In the
200-game matches generation 450 scores 0.755 against `rollout:200` (generation
350: 0.56, generation 300: 0.455) and 0.935 against the random bot, and
beats generation 300 in 196 games of 200.

Against generations 350 and 400 it scores exactly 0.50, and this says
little: in a 40-game replay of the generation-400 match (same seed, a
scratch program that keeps the colors), Black won every game, whichever
network had it, and the games repeated: 20 of the 20 with generation 450 as
White ended with the same margin, −12.5, and the other 20 with only two
margins. With `temperature-moves = 4`, two similar networks play almost the
same few games, so a head-to-head match between them measures little more
than which color wins those games. The earlier 199–1 and 200–0 results were
the newer network winning with both colors. The matches against
`rollout:200` and the random bot, whose moves vary, are the measures to use
from here.

An earlier 9×9 run, `runs/9x9-rh-s1`, was started on the pre-fix code and
stopped after 3 generations. Its checkpoints are format version 1 and no
longer load. The head-to-head against `alphazero-go` will be reported here
once it has been played.

### Generation 450: what limits it

Generation 450 scores 0.755 against `rollout:200` but 0.365 against
`rollout:1000`. Four checks looked for the cause, all on generation 450.

**More search does not help.** 200-game matches at different simulation
counts:

| Simulations | vs `rollout:200` | vs `rollout:1000` |
|---|---|---|
| 1 | 0.695 | — |
| 25 | 0.715 | — |
| 200 (the training setting) | 0.755 | 0.365 |
| 800 | 0.72 | 0.41 |

One simulation (the policy head's move) is already within 0.06 of 200,
and 800 is no better than 200 within noise (two standard errors at 200
games are about ±0.07). The strength is in the network's single look, and
the search adds little to it.

**The imagined states do not break down at `unroll_steps`.**
`mzgo_latent_probe` on 100 rollout-bot games, out to k = 12, compares f on
the state imagined k steps ahead with f on a fresh look at the real
position:

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 12 |
|---|---|---|---|---|---|---|---|---|---|
| \|value difference\| | 0.54 | 0.24 | 0.56 | 0.32 | 0.58 | 0.38 | 0.59 | 0.42 | 0.49 |
| policy TV distance | 0.34 | 0.32 | 0.35 | 0.35 | 0.37 | 0.38 | 0.39 | 0.39 | 0.43 |

Policy drift grows smoothly with no jump past k = 5, so the fight-reading
explanation offered at generation 350 is not supported. The value
difference is large at every k and alternates: about 0.54–0.61 at odd k
and 0.24–0.49 at even k. One untested reading is that after an odd number
of imagined moves the value comes out close to the right size with the
wrong sign.

**The value head memorizes the games it trains on.** Value mean squared
error against the game's result z (±1, so predicting 0 everywhere scores
1.0) on fixed sets of self-play games, for three consecutive checkpoints.
Generation G plays its games with checkpoint G − 1 and then trains on
them, so checkpoint G has trained once on generation G's games:

| Games | gen 449 | gen 450 | gen 451 |
|---|---|---|---|
| Generation 451's 100 self-play games | 1.03 | 1.04 | 0.43 |
| Generation 450's 100 self-play games | 0.89 | 0.52 | 0.40 |

On games it has not yet trained on, a checkpoint does about as well as
predicting 0 (0.89–1.04); one generation of training on them halves the
error (0.52, 0.43). The
correlation between value and result, by third of the game, is 0.74 /
0.98 / 0.997 on games already trained on and 0.18 / 0.33 / 0.58 on games
not yet seen: the value head is weakly informative on new games, but much
too confident. Each position stays in the 2500-game buffer for about 12
generations and serves as a value target about 30 times in that span
(400 steps × 256 samples × 6 unroll targets a generation), which is
enough to learn individual games by heart. A scratch check of batch norm
ruled out the other suspect: value loss on the same batches is about the
same with training-mode and inference-mode statistics.

**The search cannot correct it.** Root search values track the game
results no better than the raw value. At depth 1, the imagined values
barely depend on the move: across the top five moves their spread is
0.05 (a fresh look at the same five positions: 0.14), they correlate
with the fresh values at about 0.05, and they sit close to minus the root
value. So the tree's first ply sees almost the same value under every
move, and the choice falls to the prior. That is consistent with one
simulation playing nearly as well as 800, though not shown to cause it.

The fixes to try, one at a time and measured by the held-out value error
above (about 1.0 now) and by `rollout:1000` (0.365 now):

1. `symmetry_augment = true`: transform each training sample by one of the
   board's eight symmetries, chosen at random, so the same game is seen
   eight ways and is harder to memorize.
2. Less reuse: fewer `train_steps` a generation, or more games.
3. Value targets that mix in search values, or Reanalyse.

Generations 451–453 ran with the learning rate falling from 0.002 to reach
0.0002 at generation 550 (`lr_init = 63.245553`, `lr_decay_rate = 0.1`,
`lr_decay_steps = 40000`). The run was then restarted from generation 453
with fix 1 added:

```
./build/mzgo_train --resume runs/9x9-lg-s1 --set generations=550 --set symmetry_augment=true --set lr_init=63.245553 --set lr_decay_rate=0.1 --set lr_decay_steps=40000 --set workers=6 --set train_threads=6
```

The probe's value columns against z and its `--replay` option (probe along
a run's own self-play games) were added for this investigation.

## What changed because of these results

- `configs/5x5.cfg` and `configs/9x9.cfg` set `reward_head = true`. The
  built-in default (`RunConfig::rewardHead`, `include/mzgo/config.hpp`)
  stays `false`, so a config file that does not mention the key keeps its
  old meaning. `test_presets_parse` in `tests/test_config.cpp` checks both.
- The spec's Dynamics bullet and deviations list record the choice. The
  paper's §3 loss and pseudocode include the reward term for every game;
  Appendix G drops it for board games. Here the pseudocode reading is the
  default, and Appendix G's reading is measured above as a failure.
- After the final review, batch norm keeps running statistics per unroll
  step and inference uses the leaf's depth (`37df3a1`; checkpoint format
  version 2). A game stopped by the move cap is now truncated: no reward on
  its last move, the score at the cap as the last position's value target,
  and no targets past it (`90cbaba`). Run 4 is the 5×5 reward-head run
  repeated on that code.
- The optional `pass_plane` input (run 5) was added because of the 9×9
  diagnosis. It is off by default; the 9×9 restart turns it on.
- League play (runs 6 and 7) is optional and off by default
  (`league_size = 0`). Run 7's setting, `league_value = false`, is the one to
  use. The default stays `true`, the plain design, so that runs 6 and 7 can
  each be reproduced from their configs.
- `no_pass_before` (off by default) was added for `9x9-nep-s1`. With 0, every
  output is byte-identical to the build before it.
- `league_network_wins` (default 2) and `league_reset_on_replace` (default
  true) were added after `9x9-lg-s1` generation 200 to make league
  replacements less noisy. Their defaults change the league's behavior;
  `league_network_wins = 1` with `league_reset_on_replace = false` restores
  the rule that runs 6 and 7, `9x9-nep-s1` and `9x9-lg-s1` up to generation
  200 used.

---

Mark Castelluccio, designed and implemented with Claude Code (AI-assisted).
