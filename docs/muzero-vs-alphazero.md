# MuZero vs. AlphaZero on Go: the comparison

This project (`muzero-go`, namespace `mzgo`) and its sibling `alphazero-go`
(namespace `azgo`) play the same game with the same rules code, the same
input encoding, the same layers and the same optimizer — `mzgo`'s were
copied from `azgo` — so that every place the two diverge is a consequence
of one fact: **AlphaZero's search consults the real rules of Go at every
node, and MuZero's consults them only at the root.** Below the root,
`alphazero-go` calls `Board::play` on a copy of the parent's board; MuZero
calls `recurrentInference`, a learned function with no board in its
arguments.

For the guided walkthrough of this codebase see
[`algorithm-explained.md`](algorithm-explained.md); this document is the
side-by-side treatment, with both codebases cited. Paths starting with
`alphazero-go/` are in the sibling repository; the others are here. Line
numbers name the function's definition, as of this writing.

## Summary table

| | `alphazero-go` | `muzero-go` |
|---|---|---|
| Dynamics inside the search | the real board: `descend` plays each move on a copy (`alphazero-go/src/mcts.cpp:87`) | the learned `g`: `MuZeroNetwork::recurrentInference` (`src/network.cpp:363`) |
| A tree node holds | a `Board` | an `F × n × n` hidden state, no decoder |
| Legal moves in the tree | at every node: `Mcts::expand` creates edges for legal moves only (`alphazero-go/src/mcts.cpp:150`) | root only: `Search::beginRoot` (`src/mcts.cpp:36`); every other node has all n² + 1 children (`Search::expandLeaf`, `src/mcts.cpp:114`) |
| End of the game in the tree | exact: `Mcts::isTerminal` (two passes or the move cap) and the Tromp-Taylor score (`alphazero-go/src/mcts.cpp:41, 45`) | none; `past_end_fraction` measures how often the search goes past it |
| Unvisited child (first-play value) | Q = 0 on [−1, 1] (a draw); a game-ending pass gets its exact score at expansion | Q̄ = 0 on the normalized [0, 1] scale (a loss) |
| Exploration term | fixed `c_puct = 1.5` on raw Q, √max(1, N) | c1 = 1.25 + log((N + c2 + 1)/c2), c2 = 19652, on Q̄ = (Q + 1)/2 |
| Batching the search | virtual loss, 8 descents per tree (`mcts_batch = 8`) | lockstep across 8 games, each tree sequential (`selfplay_group = 8`) |
| Tree between moves | the subtree under the move played is kept (`Mcts::advance`, `alphazero-go/src/mcts.cpp:368`) | rebuilt every move (the root must be `h` of the real position) |
| Resignation in self-play | on in the 9×9 preset (calibrated `v_resign`) | none (the paper does not use it for MuZero) |
| Training sample | one position: input, π, z | a position plus the next K = 5 moves: K + 1 sets of targets |
| Value target | the final result z | the final result per unroll step; 0 past the end (absorbing); a game stopped by the move cap is truncated: the score at the cap, no targets past it, no reward |
| Network | one: trunk + policy and value heads | three: `h` (trunk), `g` (trunk on F + 1 planes), `f` (heads) |
| Forward work per training sample | trunk + heads | h + K·g + (K + 1)·f = 6 trunks + 6 heads at K = 5 |
| Symmetry augmentation | a random symmetry per sample, always | off by default (`symmetry_augment`) |
| Batch norm running statistics | one set per layer | one set per layer and unroll step (`g` K, `f` K + 1), chosen by search depth |
| Batch norm calibration of generation 0 | on by default (`bn_calibrate`) | off by default (opt-in switch); one K-step pass sets every step's statistics |
| Learning rate | piecewise constant by generation (`lr_schedule = 0:0.02,90:0.002`) | exponential decay per step, 0.02 · 0.1^(step / 60000) on 9×9 |
| Replay window | the last 200,000 positions | the last 2,500 games |

## Given vs. learned dynamics

`alphazero-go` builds children lazily: the first time a descent takes an
edge, it copies the parent's board and plays the move on it
(`Mcts::descend`, `alphazero-go/src/mcts.cpp:87`). The child is exact. A
capture removes the stones, a ko is forbidden by the board's history, a
second pass makes the node terminal. The network is only ever asked to
judge real positions.

`muzero-go` makes a child by asking the model: the parent's hidden state
and the move, as one extra input plane, go through `g`, then `f`
(`MuZeroNetwork::recurrentInference`, `src/network.cpp:363`). Whether the
child "knows" that stones were captured depends on what `g` learned. The
diagnostics `mzgo_latent_probe`, `mzgo_rules_probe` and `mzgo_board_probe`
exist only because of this difference: they compare imagined states with
the real positions they stand for, a question that has no meaning in
`alphazero-go`.

The cost is approximation error that compounds with depth. The benefit is
the algorithm: the same search code would plan in an environment whose
rules nobody can query.

## Root-only legality on Go

`alphazero-go`'s `Mcts::expand` (`alphazero-go/src/mcts.cpp:150`) creates
edges for the legal moves of the node's real board, so occupied points,
suicides and superko repetitions never appear anywhere in its tree.

`muzero-go` applies the real legal-move list exactly once, in
`Search::beginRoot` (`src/mcts.cpp:36`), renormalizing the prior over it
and mixing the Dirichlet noise only into it. Every deeper node has a child
for each of the n² + 1 actions with `f`'s prior unmasked
(`Search::expandLeaf`, `src/mcts.cpp:114`), as the paper prescribes
("MuZero only masks legal actions at the root"). On tic-tac-toe that was a
curiosity; on Go a large share of the points are occupied in the middle
game, captures reopen points, and ko depends on the previous position. What
keeps the search off illegal moves is the prior alone: the policy targets
are root visit distributions, which are zero on illegal moves, so `f`
learns to avoid them on states it can read, and the pessimistic first-play
value below makes a low-prior child unlikely to be visited.
`mzgo_rules_probe` measures how much prior mass is left on illegal moves k
steps into the imagination.

## Terminal states

`alphazero-go` knows when the game is over inside the tree. A node with two
passes or at the move cap is terminal (`Mcts::isTerminal`,
`alphazero-go/src/mcts.cpp:41`); a descent that reaches it backs up the exact
Tromp-Taylor result without a network call (`Mcts::search`,
`alphazero-go/src/mcts.cpp:213`). And a pass that would end the game has its
exact value before it is ever visited (`e.unvisitedQ = scoreValue(board)` in
`Mcts::expand`).

`muzero-go` has no terminal nodes. After two imagined passes the search
keeps expanding, and each such simulation costs a network call. What the
model says about those states comes from the absorbing-state targets: past
the end of a game the value target is 0 and the policy term is masked
(`fillSample`, `src/game_record.cpp:40`). So, with no reward head, the edge
that ends a game is worth Q ≈ 0 to the search — a draw — whatever the
score; `alphazero-go` values the same pass at exactly ±1. `reward_head`
gives that edge the game's result as a reward. It is on in both presets: the
paper's Appendix G omits the reward loss for board games, and that reading
was measured to fail on 5×5 ([`docs/results.md`](results.md)). Self-play's
`past_end_fraction` and `capped_games` columns in `log.csv` show how much
of the search runs past the end.

### Dead stones: a real terminal position versus a learned one

Tromp-Taylor area scoring counts every stone on the board as alive. There
is no agreement phase where the players remove dead stones. A stone the
opponent could capture still counts for its owner if the game ends with it
on the board, and it also stops the empty points around it from being
anyone's territory. So under these rules a player who has invaded can be
"winning" on the count until somebody captures the invaders.

`alphazero-go` gets this right without being told. When its search reaches
two passes it scores the real board, so a pass that leaves enemy stones
inside its territory is scored with them there. If that loses, the search
sees the loss and prefers to capture first.

`muzero-go` can only know what the reward head was trained on. Its reward
targets come from self-play endings, and in self-play few passes are
answered by a pass (78 of 473 in the recorded games of the 5×5 reward-head
run before the batch norm fix, 73 of 268 in the run after it). An
ending with invading stones still on the board may be rare or absent in
that data. Against an opponent that keeps invading (the random bot does),
nothing forces the model's idea of "a pass now wins" to match what the real
count says. This is a hypothesis, not a measurement. In the positions
examined in `docs/results.md`, the real count favoured the model. Before
the fix, the model *under*-rated the pass and ranked moves that keep the
game going at values above 1. After it, the model rates the pass about as
high as playing on, and values above 1 remain on both. The one position
built to test the hypothesis gave the pass no visits, so it tested
nothing. The general point stands either way. AlphaZero's search
scores the real terminal position, while MuZero's reward head scores the
terminal positions it has seen, and the two agree only where self-play
covered the case.

## The first-play value

The two searches disagree about a move nobody has tried yet.

`alphazero-go`'s `Mcts::selectEdge` (`alphazero-go/src/mcts.cpp:57`):

```cpp
        // Each virtual loss adds a -1 result for the chooser. An unvisited
        // edge has Q = e.unvisitedQ: 0, except a known game-ending pass.
        const float q = n > 0 ? (e.valueSum - static_cast<float>(e.virtualLoss)) / n : e.unvisitedQ;
        const float u = cfg_.cPuct * e.prior * sqrtN / (1 + n);
```

`muzero-go`'s `Search::selectChild` (`src/mcts.cpp:75`), the pseudocode's
`ucb_score`:

```cpp
        // Unvisited: Q̄ = 0 on the normalized scale (as bad as a loss), so
        // only the prior term can draw the search there.
        const float q = c.visits > 0 ? minmax_.normalize(edgeQ(c)) : 0.0f;
        const float score = q + c.prior * sqrtN / (1.0f + static_cast<float>(c.visits)) * pbC;
```

With MinMaxStats initialized to the known bounds [−1, 1] and no reward
head, the normalization is the fixed map Q̄ = (Q + 1)/2. Q̄ = 0 is Q = −1:
MuZero treats an unvisited move as a loss, AlphaZero-Go as a draw. With
MuZero's rule a sibling is tried only when its prior term beats the visited
children's Q̄ plus their own prior terms, so the search follows the prior
more closely and goes deeper along it. That suits a search whose prior is
the only guard against illegal moves; it also means a strong move with a
small prior is found later.

The exploration weights differ too. The log term is about 0.01 for a few
hundred simulations, so MuZero's weight is 1.25 on a [0, 1] Q̄, which is
the same as 2.5 on a [−1, 1] Q: about 1.7 times `alphazero-go`'s
`c_puct = 1.5`.

## Batching: virtual loss vs. lockstep

`alphazero-go` batches inside one tree (`Mcts::search`,
`alphazero-go/src/mcts.cpp:213`): it runs up to `mcts_batch = 8` descents
before evaluating any, each adding a virtual loss along its path so the next
descent goes elsewhere, then evaluates the leaves together. Those descents
do not see each other's results, and the virtual losses bend selection while
they are in flight: the search is not the sequential algorithm.

`muzero-go` batches across games (`playGroup`, `src/selfplay.cpp:32`): a
worker plays `selfplay_group = 8` games, runs one simulation's selection in
each, evaluates all their leaves in one `recurrentInference`, and backs each
up in its own tree. Each game's tree is exactly the sequential pseudocode
search (`test_lockstep_equals_sequential`); the cost is that the batch is at
most the number of live games and shrinks as games finish. A game's record
does not depend on the worker count; it does depend on the group size,
because a batched matrix multiply can round differently.

## Tree reuse

`alphazero-go` keeps the subtree under the move just played
(`Mcts::advance`, `alphazero-go/src/mcts.cpp:368`, called from
`playSelfPlayGame`), so a search starts with the visits the previous one
made below that move, and `simulations` counts only new ones. `muzero-go`
clears the tree at every move (`Search::beginRoot`): its root must be
`h(o)` of the real observation, while the kept child would be an imagined
state. At equal `simulations` the AlphaZero search therefore ends with
somewhat more visits at its root.

## Resignation

`alphazero-go`'s 9×9 preset resigns self-play games (AlphaGo Zero's
`v_resign`, calibrated so at most 5% of the games played out with
resignation disabled would have been resigned by their winner). In its
9×9 experiments it shortened the average game from about 88 plies to
about 51, and it is part of that preset's tuned "recipe". MuZero's paper does not use resignation, so `muzero-go`
plays every game to two passes or the 2n² move cap. That makes self-play
longer (the design's estimate is 1.5–2 times `alphazero-go`'s), and it
means every game teaches the model how games end — including the
absorbing states above. The matched `alphazero-go` run for the head-to-head
comparison turns resignation off.

## Targets

`alphazero-go` trains on single positions (`encodeTrainingSample`,
`alphazero-go/src/replay_buffer.cpp:59`): the input planes, the visit
distribution π and the result z for the side to move, under a random
symmetry.

`muzero-go` trains on unrolls (`fillSample`, `src/game_record.cpp:40`): the
input planes at t, the moves t … t + K − 1, and for every k = 0..K the visit
distribution and result at t + k. Past the end of a game the players
ended: value 0, no policy loss, random actions. A game the move cap stopped
is truncated instead: no reward on any move, the score at the cap as the
value of the last position, and no value, reward or policy target past it.
(`alphazero-go` trains every position of a capped game on its score, the
same as an ended one; with no rewards and no unroll past the end, the
distinction does not arise there.) Both value targets are Monte Carlo returns —
the actual result — since the pseudocode's board-game `td_steps` reaches the
end of every game; MuZero's machinery could bootstrap from stored root
values (and Reanalyze them), but not on board games as published.

The loss weights each unroll step: 1 for k = 0 and 1/K for the rest, and
the gradient into each hidden state after a recurrent step is halved
(`MuZeroNetwork::loss`, `MuZeroNetwork::backward`, `src/network.cpp:190,
261`). `alphazero-go` has one step and no such scaling.

## Network shapes and training cost

| | `alphazero-go` 9×9 | `muzero-go` 9×9 |
|---|---|---|
| trunk(s) | 18 → 64 stem, 6 blocks × 64 | `h`: 18 → 64 stem, 6 blocks; `g`: 65 → 64 stem, 6 blocks |
| heads | policy (1×1 conv → 2, FC → 82), value (1×1 conv → 1, FC 64, FC 1, tanh) | the same, as `f`, read from every hidden state |
| parameters | about 473,000 (the same layers as `h` + `f`) | 954,749 (`h` 454,400, `g` 481,472, `f` 18,877) |
| forward per training sample | 1 trunk + heads | `h` + 5 `g` + 6 `f` |
| network calls per simulation | 1 (trunk + heads); none at a terminal node | 1 (`g` + `f`), always |
| optional extras | ownership head, gating, growing, GPU engines | reward head (on in both presets) |

A `muzero-go` training sample does roughly 5–6 times the work of an
`alphazero-go` one, forward and backward. A search simulation costs about
the same in both, since `g` + `f` has the size of AlphaZero's trunk + heads
(`g`'s stem reads 65 planes instead of 18).

## What is identical

Copied from `alphazero-go` and unchanged in behaviour:

- the rules: `Board` (captures, suicide illegal, Zobrist hashing), `Game`
  (positional superko, two passes or 2n² moves end the game, Tromp-Taylor
  area scoring with komi);
- the input encoding: 18 planes, own/opponent stones for the last 8
  positions and two side-to-move planes, and the eight symmetries;
- the layers (`Conv2d`, `BatchNorm2d`, `Linear`, ReLU) with their
  hand-written backward passes (`BatchNorm2d` here can keep several sets of
  running statistics, one per unroll step; with one set it is the same), the naive and BLAS backends, the thread pool
  and its bit-for-bit determinism;
- the optimizer: SGD, momentum 0.9, weight decay 1e-4 on conv and linear
  weights;
- serialization, SGF output, the GTP command set, `tools/gtp_match.py`;
- the presets' shared keys: 9×9 komi 7.5 (5×5 24.5), 200 simulations, 200
  games and 400 steps per generation, batch 256, Dirichlet α = 10 / n² at
  25%, 7 opening moves at temperature (8 on 5×5), the 6 × 64 trunk.

## Measured

The 5×5 runs are measured in [`docs/results.md`](results.md). Nothing is
compared with `alphazero-go` by measurement yet. The head-to-head match through
GTP (`mzgo_gtp` against `alphazero-go`'s `gtp`, via `tools/gtp_match.py`) at
matched settings, the matched `alphazero-go` run with resignation off, and
timings of both on the same machine will be reported in
[`docs/results.md`](results.md).

---

Mark Castelluccio, designed and implemented with Claude Code (AI-assisted).
