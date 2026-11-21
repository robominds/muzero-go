# MuZero Go

A from-scratch C++17 implementation of MuZero (Schrittwieser et al. 2020)
for the game of Go on 5×5 and 9×9 boards. Three networks — representation
`h`, dynamics `g` and prediction `f` — learn a model of the game together,
and Monte Carlo tree search plans inside that learned model: the real rules
are consulted only at the root of each search (the observation and the
legal-move mask) and to play the chosen move. Below the root the search
never sees a board, so it has to have learned what captures, illegal moves,
ko and the end of the game look like.

It is a **teaching example**, the sibling of `alphazero-go` (AlphaZero on
the same game, whose rules, encoding, layers and optimizer were copied in)
and `muzero-tictactoe` (MuZero on a game too small for the model to have
much to learn). Every part — the forward and backward passes, including
backpropagation through the unrolled dynamics, the search, self-play,
training and the diagnostics — is written directly, with no
machine-learning library. It follows the paper and its pseudocode closely;
every deviation is listed in the design
([`docs/superpowers/specs/2026-11-19-muzero-go-design.md`](docs/superpowers/specs/2026-11-19-muzero-go-design.md))
and in the code where it happens. It is not meant to be a strong Go engine.

## Build

Requires CMake ≥ 3.16 and a C++17 compiler. The default build uses only
the C++ standard library:

```bash
cmake -S . -B build-naive -DCMAKE_BUILD_TYPE=Release
cmake --build build-naive -j
ctest --test-dir build-naive --output-on-failure
```

Matrix multiplication can use BLAS instead (Apple Accelerate on macOS,
OpenBLAS elsewhere; on Ubuntu `apt install libopenblas-dev`). This is the
build to train with:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMZ_USE_BLAS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The tests are one executable per component (`tests/test_*.cpp`, `assert`
based, including finite-difference gradient checks of every function and
of the whole unrolled loss) plus `tests/integration_smoke.sh`, a tiny 5×5
run that is resumed and then probed, analyzed, played through GTP and
matched, in under a minute.

Every run config has `backend = blas` by default; with the naive build pass
`--set backend=naive` to `mzgo_train`. The other programs default to BLAS
when the build has it and to the naive backend otherwise.

## Quick start

```bash
./build/mzgo_train --config configs/5x5.cfg --run runs/five
./build/mzgo_rules_probe --checkpoint runs/five/checkpoints/gen_0060.mzn
./build/mzgo_analyze --checkpoint runs/five/checkpoints/gen_0060.mzn --moves "C3 B3"
```

Interrupt `mzgo_train` at any time and continue with
`./build/mzgo_train --resume runs/five`.

## Presets

`configs/5x5.cfg` and `configs/9x9.cfg`. Any key can be overridden with
`--set key=value`. Defaults not listed: K = `unroll_steps` = 5, 200
simulations, root Dirichlet α = `dirichlet_scale` / n² = 10 / n² at 25%,
SGD momentum 0.9, weight decay 1e-4, no symmetry augmentation, no batch
norm calibration.

| | 5×5 | 9×9 |
|---|---|---|
| komi | 24.5 | 7.5 |
| `repr_blocks` / `dyn_blocks` × `filters` | 3 / 3 × 32 | 6 / 6 × 64 |
| `value_hidden` | 32 | 64 |
| `reward_head` | true | true |
| `generations` | 60 | 150 |
| `games_per_generation` | 200 | 200 |
| `simulations` | 200 | 200 |
| `temperature_moves` | 8 | 7 |
| `replay_games` (window) | 2000 | 2500 |
| `train_steps` per generation | 200 | 400 |
| `batch_size` | 128 | 256 |
| `lr_init`, decaying ×0.1 over `lr_decay_steps` | 0.02 over 12,000 | 0.02 over 60,000 |
| design estimate of wall time on one Mac | about 1 hour | 1.5–2 days |

The 9×9 network matches `alphazero-go`'s 9×9 trunk (6 blocks × 64), so the
two can be compared at matched settings. 5×5 uses komi 24.5 as in
`alphazero-go`: 5×5 Go is a win for Black by the whole board, and at 7.5
Black wins nearly every game. Measured here, 24.5 also has a cost: from a
cold start Black never wins, and its policy collapses to passing
([`docs/results.md`](docs/results.md)). The later 5×5 runs use
`--set komi=0.5`. Checkpoints do not store komi; the programs below
default to 24.5 on 5×5 and 7.5 otherwise.

## Usage

Every program prints its usage with `--help`.

### `mzgo_train` — the generation loop

```
mzgo_train --config FILE --run DIR [--set key=value]...
mzgo_train --resume DIR [--set key=value]...
```

Each generation plays `games_per_generation` self-play games with the
current network (every move a full MuZero search), adds them to the replay
window of the last `replay_games` games, runs `train_steps` SGD steps on
batches of K-step unrolls, saves a checkpoint, and every `arena_every`
generations plays `arena_games` games each against generation g −
`arena_gap`, a rollout-MCTS bot (`rollout_bot_simulations`, searching with
the real rules) and the random bot. A run directory holds:

| file | contents |
|---|---|
| `config.cfg` | the resolved config, every key |
| `log.csv` | one row per generation: self-play seconds, mean game length, `capped_games` (stopped by the 2n² move cap; their targets are truncated, not ended), search `mean_depth` / `max_depth`, `past_end_fraction` (simulations whose imagined path goes on beyond two passes in a row), lr, the loss per unroll step (`policy_k0..K`, `value_k0..K`, `reward_k1..K`), the arena scores, the replay window's size; with a league also the league columns (below) |
| `checkpoints/gen_GGGG.mzn` | the network after generation G (`gen_0000` is the starting one); format version 3, with batch norm statistics per unroll step and the pass plane flag (version 2 files still load) |
| `sgf/gen_GGGG_I.sgf` | the first `sgf_per_generation` self-play games of each generation (and `gen_GGGG_league_I.sgf`, as many league games) |
| `state.bin` | what a resume needs: generation, step, network, optimizer velocity, replay window, league (format version 2; version 1 files still load) |

A new run refuses a non-empty directory. `--resume` continues from
`state.bin`; overrides may change any key except the architecture keys
(`board_size`, `history`, `repr_blocks`, `dyn_blocks`, `filters`,
`value_hidden`, `reward_head`, `pass_plane`, `unroll_steps`: batch norm keeps running
statistics per unroll step, so K is part of the network) and `league_size`. To go on past the
config's last generation: `--resume DIR --set generations=200`. Every random
choice is derived from (seed, generation, phase, step), so a resumed run
writes exactly the checkpoints and `state.bin` an uninterrupted one would,
and results do not depend on `workers` or `train_threads` (0 = all cores).

`reward_head` (a reward prediction on each dynamics step, trained on the
result at the last move of a game the players ended, 0 for a game the move
cap stopped) is on in both presets. The built-in default
is off, which is the paper's Appendix G reading, so a config without the
key keeps that meaning. `docs/results.md` measures why the presets turn it
on. Opt-in switches, off by default (the paper's version is the default):
`bn_calibrate` (set generation 0's batch norm statistics, every unroll
step's set, from one K-step pass over random-move positions),
`symmetry_augment` (one random board symmetry per unroll), `pass_plane` (off
by default (paper input); on adds a plane marking that the last move was a
pass, so the input has 2·history + 3 planes; an architecture key).

League play (off by default; the design is
[`docs/superpowers/specs/2026-11-20-league-design.md`](docs/superpowers/specs/2026-11-20-league-design.md))
plays part of each generation's games against a league of opponent slots,
all held by the random bot at first. Only the network's own moves in those
games train the policy; every position trains value, reward and dynamics.
When the network scores well enough against a member, the network that
played (generation g − 1's during generation g) takes its slot, at most one
slot per generation. A random-bot member must be beaten in two generations
in a row first (it never comes back once replaced, so one noisy win should
not remove it); a network member goes after one. The summary line marks a
random member's first strike with `*` after its score (`random 0.7*`);
`state.bin` keeps the strikes, so a resume continues them. The keys:

- `league_size` (default 0 = off, 0..8): the number of slots; cannot change on resume.
- `league_fraction` (default 0.5, 0..1): round(`games_per_generation` × this) of each generation's games are league games; games 2k and 2k + 1 play slot k mod `league_size`, the network Black in the first and White in the second.
- `league_replace_score` (default 0.6, 0..1): the score (a draw counts half) that beats a member.
- `league_min_games` (default 20, ≥ 0): the fewest games against a member for its score to count; a config where some slot gets fewer league games a generation than this is rejected (that member could never be replaced).

`league_fraction` counts games, not positions: league games against the
random bot run long, so they can be most of a generation's new positions
(`league_positions` below). And since the opponent's positions carry no
policy target, `policy_k` losses, averaged over every batch slot including
masked ones, drop as more positions are masked; compare them across runs
with care.

`log.csv` then has six more columns: `league_members` (the slots during
the generation, before its replacement, `random` or `gen_N`, `;`-joined),
`league_scores` (the network's score against each, empty for a slot
without games), `league_replaced` (the slot, or −1; the new member shows in
the next row), `league_mean_length`, `league_capped_games` and
`league_positions` (positions the league games added). `mean_length`,
`capped_games` and the search columns stay the self-play games' alone, so a
league run reads like any other; `seconds_selfplay` and `games` cover both.
The rollout bot is never a member, so `arena_rollout` stays a held-out
measure; `arena_random` no longer is once the league plays the random bot.

Checkpoints and `state.bin` files written before batch norm kept its
statistics per unroll step (checkpoint format version 1) cannot be loaded
or resumed; loading one says the format changed. Version 2 files (before
`pass_plane`) still load, with the plane off.

### `mzgo_match` — two players, N games

```
mzgo_match --a SPEC --b SPEC --games N --board-size n --komi k
           [--simulations 200] [--temperature-moves 4] [--workers 0] [--seed S] [--backend blas]
```

SPEC is `ckpt:PATH` (a network searching `--simulations` per move, the
first `--temperature-moves` moves sampled in proportion to visits, no
noise), `rollout:SIMS` (the rollout MCTS bot) or `random`. Colors
alternate; prints A wins / B wins / draws and A's score. Writes no files.

```bash
./build/mzgo_match --a ckpt:runs/nine/checkpoints/gen_0150.mzn \
    --b ckpt:runs/nine/checkpoints/gen_0050.mzn --games 100 --board-size 9 --komi 7.5
```

### `mzgo_gtp` — a checkpoint as a GTP engine

```
mzgo_gtp --checkpoint CKPT [--simulations 200] [--backend blas|naive] [--seed 1] [--komi 7.5]
```

Speaks GTP version 2 on stdin/stdout (`protocol_version`, `name`,
`version`, `known_command`, `list_commands`, `quit`, `boardsize` — the
checkpoint's size only — `clear_board`, `komi`, `play`, `genmove`, `undo`,
`showboard`, `final_score`), the same command set as `alphazero-go`'s `gtp`.
`genmove` searches at temperature 0 without root noise and never resigns.
One line per move (its visits, the root value) goes to stderr. Works with
any GTP controller or board program (Sabaki, gogui).

### `tools/gtp_match.py` — a match between two GTP engines

Copied from `alphazero-go`. Plays `--games` games between two engine
command lines with colors alternating, scores them with the `--scorer`
engine's `final_score`, writes an SGF per game, `results.csv` and both
engines' stderr to `--out`, and prints A's score with a 95% Wilson interval
and the implied Elo difference. The head-to-head with `alphazero-go`:

```bash
AZ=../projects/alphazero-go     # the sibling checkout; AZ_RUN: its matched 9x9 run
python3 tools/gtp_match.py \
    --a "./build/mzgo_gtp --checkpoint runs/nine/checkpoints/gen_0150.mzn --simulations 200" \
    --b "$AZ/build-blas/gtp --checkpoint $AZ/runs/$AZ_RUN/checkpoints/gen_0150.azn --backend blas --simulations 200" \
    --games 100 --size 9 --komi 7.5 --out runs/vs-alphazero
```

Both engines search deterministically at temperature 0, so repeated games
between them are identical; the summary counts distinct games
(`alphazero-go`'s `gtp --random-symmetry` varies its side).

### The diagnostics: `mzgo_latent_probe`, `mzgo_rules_probe`, `mzgo_board_probe`

```
mzgo_latent_probe --checkpoint P [--games 100] [--k 5] [--backend blas|naive] [--seed 1] [--workers 0] [--komi K]
mzgo_rules_probe  --checkpoint P [--games 100] [--k 5] [--backend blas|naive] [--seed 1] [--workers 0] [--komi K]
mzgo_board_probe  --checkpoint P [--games 100] [--k 5] [--epochs 20] [--backend blas|naive] [--seed 1] [--workers 0] [--komi K]
```

Each plays `--games` games of the rollout bot against itself (real, legal
Go that the network did not choose), starts at every position t, runs `h`
there and then `g` along the moves really played for k = 1..`--k` steps,
and compares each imagined state with the real position t + k:

- `mzgo_latent_probe`: `f` on the imagined state vs `f` on a fresh `h` of
  the real position — mean value difference and policy total-variation
  distance per k. (Consistency with the network's own view, not
  correctness.)
- `mzgo_rules_probe`: the policy mass on moves that are illegal in the real
  position — all illegal moves, occupied points, superko — and on points
  freed by a capture within the k steps (legal again: mass there is good).
- `mzgo_board_probe`: a linear readout from the hidden channels at a point
  to its content (empty / own stone / opponent stone), fitted on fresh `h`
  states with no gradient into the network, then applied to imagined
  states: accuracy per k, and on the points whose content changed (new
  stones, captures).

`docs/algorithm-explained.md`, chapter 11, explains how to read each one.

### `mzgo_analyze` — one search, in full

```
mzgo_analyze --checkpoint P [--moves "C3 D4 pass ..."] [--simulations 200] [--top 10] [--backend blas|naive] [--komi K]
```

Plays the moves (GTP coordinates or `pass`, Black first) from the empty
board and runs one search on the position reached: the root value, visits,
prior and Q of the `--top` moves, the search depth, the share of
simulations going beyond the real game's end, and the principal variation —
which below its first move exists only in the model's imagination — with
each edge's predicted reward and Q and the value along it.

### `mzgo_bench` — what the network costs

```
mzgo_bench [--board-size 9] [--blocks 6] [--filters 64] [--batch 256] [--unroll 5]
           [--group 8] [--backend naive|blas] [--threads 0] [--seconds 3]
```

Times one training step (`h`, K steps of `g`, K + 1 of `f`, the loss and
backpropagation through the unroll) at `--batch` on `--threads`, and
recurrent inference (`g` then `f`) at batch `--group` on one thread, as a
self-play worker calls it. `--blocks` sets both trunks; `--filters` also
sets the value head's width.

### `tools/check_citations.py`

Checks that every `file:line` caption in `docs/algorithm-explained.md`
names the line where the quoted function or type is defined; `--fix`
rewrites stale numbers.

## Results

Measurements are reported in [`docs/results.md`](docs/results.md), failures
included. In short, from four 5×5 runs (the first three before the
batch-norm fix, the fourth after it):

- **Without a reward head** (the paper's Appendix G reading for board games,
  and the original default), the move that ends the game is worth about 0
  to the search, whatever the score. A winning pass got no visits.
- **With komi 24.5 on 5×5**, Black never won a self-play game from a cold
  start, and its policy collapsed to passing.
- **With the reward head and komi 0.5, before the fix** (batch norm then
  kept one set of running statistics for all unroll steps), generation 40
  lost to the random bot (0.33) and to the rollout bot (0.10).
- **The same run on the fixed code** (statistics per unroll step, capped
  games truncated): generation 40 beats the random bot (0.79 over 200
  games), beats its own generations 5–30 (0.75–1.00 over 40 games), and
  still loses to the rollout bot (0.06 over 200). Its search values still
  go above 1 just after a pass, because the model's value after the
  game-ending edge is not 0. A winning pass with a low prior is never
  visited.

`reward_head = true` is therefore set in both presets. The 9×9 run
(`runs/9x9-s1`, reward head on, fixed code) is in progress. The
head-to-head with `alphazero-go` will follow it.

Measured so far, with `mzgo_bench` for the 9×9 preset network (6 + 6
blocks, 64 filters, BLAS) on a Mac Studio (M5 Max) while another training
job used 12 of its 18 cores: **453 ms per training step** at batch 256,
K = 5, on 6 threads, and **3,835 recurrent inferences per second** at batch
8 on one thread.

## Learn how it works

- [`docs/algorithm-explained.md`](docs/algorithm-explained.md) — the
  walkthrough, tied to the code: the three functions on Go, action planes,
  min-max normalization, root-only legality and what happens below the root,
  search signs and MinMaxStats, targets and absorbing states, the unrolled
  loss and its 1/K and ½ scalings, batch norm across unroll steps,
  lockstep self-play, the generation loop, and what each probe measures.
- [`docs/muzero-vs-alphazero.md`](docs/muzero-vs-alphazero.md) — every
  place this project differs from `alphazero-go`, and what is identical.
- [`docs/superpowers/specs/2026-11-19-muzero-go-design.md`](docs/superpowers/specs/2026-11-19-muzero-go-design.md)
  — the design, including the complete list of deviations from the paper.

## Project layout

```
include/mzgo/        headers; each starts with a short explanation of its concept
  board, game        Go rules: captures, suicide, positional superko, Tromp-Taylor   (from alphazero-go)
  encode, symmetry   the 18 input planes; the 8 board symmetries                     (from alphazero-go)
  layers, backend    Conv2d, BatchNorm2d, Linear with hand-written backward; naive / BLAS gemm (from alphazero-go)
  parallel, rng      thread pool with deterministic chunking; xoshiro256** and seed derivation (from alphazero-go)
  serialize, sgf, optimizer, params                                                   (from alphazero-go)
  blocks             trunk, heads, min-max normalization, per-call activation caches
  network            MuZeroNetwork: h, g, f; the unrolled forward, loss and BPTT; inference
  model              the Model / ModelSession interface search runs against
  mcts               the search: root-only legality, pUCT, MinMaxStats, backup
  selfplay           lockstep self-play across games; MuZeroPlayer
  game_record, replay_buffer, train_batch   stored games, the replay window, K-step targets
  trainer, config, checkpoint               the generation loop, run config, network files
  league             league play: members, results, which slot to replace
  players, match, player_spec               random and rollout-MCTS bots, matches, player specs
  probes             latent, rules and board probes; analyze
  gtp                the GTP engine
src/                 one .cpp per header
apps/                train, match, gtp, latent_probe, rules_probe, board_probe, analyze, bench (built as mzgo_*)
tests/               one assert-based test per component, integration_smoke.sh, fake_model.hpp
tools/               gtp_match.py, check_citations.py
configs/             5x5.cfg, 9x9.cfg
docs/                algorithm-explained.md, muzero-vs-alphazero.md, results.md, the design and plan
```

## Deliberately out of scope

About strength or speed rather than the algorithm, and left out:

- GPU backends (Metal, CUDA, Core ML), distributed self-play;
- resignation, gating, an ownership head, network growing (all in
  `alphazero-go`);
- MuZero Reanalyze, categorical value and reward supports, prioritized
  replay (Atari-only in the paper), Atari itself;
- training on 13×13 or 19×19 (the code accepts sizes 5–19; only 5×5 and
  9×9 have presets);
- matching `alphazero-go`'s strength. The design does not block growing
  toward it (bigger networks, GPU, distributed self-play), but none of that
  is built.

One thing is ruled out on principle: using the real rules below the root of
the search. That would no longer be MuZero.

## Authorship

Mark Castelluccio, designed and implemented with Claude Code (AI-assisted).
