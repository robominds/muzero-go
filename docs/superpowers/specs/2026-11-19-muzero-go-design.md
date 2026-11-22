# MuZero Go — Design

Date: 2026-11-19. Author: Mark Castelluccio, designed with Claude Code.

## Purpose

Mark wants to learn how MuZero works on a problem harder than tic-tac-toe.
`muzero-tictactoe` showed the mechanics but the game is too small for the
learned model to have anything interesting to learn. Go has real rules for
a model to discover — captures, liberties, suicide, ko, passing and the end
of the game — and a sibling AlphaZero implementation (`alphazero-go`) to
compare against.

The deliverable is understanding: a from-scratch, readable implementation
that follows the MuZero paper closely, runs on 5×5 and 9×9 on Mark's
machines, and comes with diagnostics that show what the learned model has
and has not learned about Go, plus a walkthrough document.

## Goals

1. MuZero as published (Schrittwieser et al. 2020, arXiv 1911.08265v2, and
   its official `pseudocode.py`), board-game configuration, scaled down in
   size only. Every deviation from the paper is listed in this document and
   in the code where it happens.
2. 5×5 trains in about an hour (fast tests and smoke runs use smaller
   overrides); a 9×9 preset run takes about 1.5–2 days on one Mac (CPU +
   Accelerate BLAS). Estimate: training K = 5 costs h + 5g + 6f per sample,
   ~5–6× an AlphaZero step, ~10–12 min per generation of 400 steps; self-play
   without resignation ~1.5–2× `alphazero-go`'s; ~15 min per generation in
   all. The MuZero run and the matched `alphazero-go` run (resignation off,
   ~16 h) can run at the same time on two machines (compute-cluster), with
   Mark's go-ahead before the long runs start.
3. Success (option A, agreed): on 9×9 the network clearly and steadily
   improves — later generations beat earlier ones and beat fixed bots —
   and it is compared head-to-head with `alphazero-go` at matched settings
   (same board, komi, network size, simulations, games and steps per
   generation) through GTP. It does not need to match `alphazero-go`'s
   strength.
4. Diagnostics that look inside the learned model (see Diagnostics).
5. Documentation: an algorithm walkthrough tied to the code, a results
   document reporting what was measured (including failures), and a
   MuZero-vs-AlphaZero comparison.

## Non-goals

- Matching `alphazero-go`'s strength (option B). The design must not block
  growing toward it later (bigger nets, GPU, distributed self-play), but
  none of that is built now.
- GPU backends (Metal, CUDA, Core ML), distributed self-play, resignation,
  gating, the ownership head, network growing: all `alphazero-go` features
  that are about strength or speed, not the algorithm.
- MuZero Reanalyze, Atari, categorical value/reward support (Atari only in
  the paper), prioritized replay (Atari only in the paper).
- 13×13 / 19×19 training (the code accepts sizes 5–19; only 5×5 and 9×9
  are presets).

## Anti-goals

- A "MuZero" that quietly uses the real rules below the root of the search.
  The real game is consulted only at the root (observation + legal-move
  mask) and to play the chosen move. Anything else breaks the point of the
  project.
- An implementation whose only output is a strength number. If the model is
  wrong, the diagnostics should show where.

## Constraints

- C++17, no ML library: forward and backward passes written by hand, like
  both siblings. CMake ≥ 3.16. Optional BLAS (Accelerate on macOS,
  OpenBLAS on Linux) for gemm; a naive backend always works.
- Deterministic from a seed (xoshiro256** RNG streams derived per game /
  per generation, as in `alphazero-go`); results must not depend on thread
  count.
- Runs must survive interruption: checkpoint + resume.
- New repo `~/muzero-go`, self-contained: code reused from `alphazero-go`
  is copied in (namespace `mzgo`), not linked.

## Approaches considered

1. **From-scratch C++ reusing `alphazero-go`'s foundation (chosen).** Copy
   the Go rules (Board, Game with positional superko, Tromp-Taylor),
   input encoding, symmetries, RNG, layers (Conv2d / BatchNorm2d / Linear /
   ReLU with hand-written backward), the naive/BLAS backend, thread pool,
   serialization and SGD-with-momentum. Write everything MuZero-specific
   new. Trades: a few thousand lines of proven code for free and an
   apples-to-apples comparison with `alphazero-go`; new code stays focused
   on what MuZero changes.
2. **From scratch, nothing reused.** Most to learn about Go rules and conv
   layers, but those aren't the subject, and rewriting tested code invites
   bugs that would be confused with MuZero failures.
3. **PyTorch.** Fastest to write (autograd handles backprop through the
   unroll), runs on the Mac GPU. Rejected: breaks the house style both
   siblings follow, hides the part of MuZero that is genuinely new
   (backprop through time across the dynamics function), and makes the
   `alphazero-go` comparison a comparison of two codebases.

## The algorithm as implemented

### Three functions

All three are residual convnets in the AlphaZero style (3×3 convs, batch
norm, ReLU, residual blocks of two convs). Hidden states are `F × n × n`
feature maps (spatial, as in the paper, so actions can be planes).

- **Representation h(o) → s⁰.** Input: `alphazero-go`'s planes — own /
  opponent stones for the last 8 positions plus a black-to-move and a
  white-to-move plane (2·8 + 2 = 18 planes). Paper: "In Go ... we encode
  the last 8 board states as in AlphaZero". Stem conv + BN + ReLU, then
  `repr_blocks` residual blocks, then hidden-state normalization.
  *Optional deviation (`pass_plane`, off by default, so the paper's input
  stays the default and old checkpoints keep working):* one extra input
  plane at index 2H + 2, all ones iff the move that led to the position was
  a pass (as in KataGo). In the AlphaZero encoding a pass shows only as two
  identical history boards, and measurements showed the network cannot learn
  from that that a pass right after the opponent's pass ends the game. The
  plane is symmetry invariant and is part of the architecture (the input
  has 2H + 3 planes).
- **Dynamics g(sᵏ⁻¹, aᵏ) → sᵏ (, rᵏ).** Input: hidden state concatenated
  with one action plane. Paper: "a normal action ... is encoded as an all
  zero plane, with a single one in the position of the played stone. A pass
  is encoded as an all zero plane." So the input is `F + 1` planes. Stem
  conv + BN + ReLU to `F`, then `dyn_blocks` residual blocks, then
  normalization. The paper is inconsistent about the reward for board
  games: its §3 loss and the pseudocode's `update_weights` include the
  reward term for every game, while App. G omits it ("in board games without
  intermediate rewards, we omit the reward prediction loss"). The reward
  head (`reward_head = true`: the value head's shape, 1×1 conv → BN → ReLU →
  FC → ReLU → FC → tanh, reading the normalized sᵏ; trained on the final
  outcome at the last move of a game the players ended, 0 elsewhere, and 0
  on every edge of a game the move cap truncated) is **on in both presets**. Without
  it, the game-ending move leads to an absorbing state with value target 0
  and carries no reward, so the search values it at about 0 whatever the
  score. Measured on 5×5 in `docs/results.md`: a winning pass got prior
  0.028 and no visits, and the no-reward-head generation 12 lost 1–39 to
  the reward-head generation 12. The struct default stays `false` (search
  uses r = 0), so a config without the key keeps the App. G behaviour.
- **Prediction f(sᵏ) → (pᵏ, vᵏ).** AlphaZero heads, no residual blocks of
  its own (paper: "one or two convolutional layers that ... reduce the
  number of planes" then a fully connected layer). Policy: 1×1 conv to 2
  planes → BN → ReLU → FC to n²+1 logits. Value: 1×1 conv to 1 plane → BN
  → ReLU → FC → ReLU → FC → tanh.
- **Batch norm statistics per unroll step** (my call; the paper and its
  pseudocode are silent on batch norm across the unroll). Every batch norm
  keeps one set of running statistics per call index — h one, g and the
  reward head K (the calls producing s¹..sᴷ), f K + 1 (s⁰..sᴷ) — while its
  learnable γ and β stay shared. Training step k normalizes with its own
  batch statistics and updates only set k; inference at search depth d
  uses set min(d, K). With one shared set, inference at every depth would
  normalize with an average over the unroll steps and compute activations
  training never produced (the final review measured k = 1 reward MSE
  0.032 at inference against 0.0046 with batch statistics). K
  (`unroll_steps`) is therefore an architecture key, stored in the
  checkpoint.
- **Hidden-state normalization** (App. G): after h and after every g,
  `s ← (s − min s) / (max s − min s)` over all elements of one sample's
  hidden state. Backward passes the gradient through min and max exactly
  (the argmin/argmax elements receive the correction terms); guard
  `max − min < 1e-5` by using 1e-5 as the denominator (my call).

Scaled sizes (paper: 16 blocks × 256 for h and g):

| preset | repr_blocks | dyn_blocks | filters | value_hidden |
|---|---|---|---|---|
| 5×5 | 3 | 3 | 32 | 32 |
| 9×9 | 6 | 6 | 64 | 64 |

The 9×9 sizes match `alphazero-go`'s 9×9 preset trunk (6 × 64), so a
search simulation costs about one AlphaZero evaluation.

### Search (App. B, pseudocode `run_mcts`)

- Root: h on the real observation, then f. The root's children are
  restricted to the real game's legal moves (positional superko included)
  and the prior renormalized over them. Below the root, every one of the
  n²+1 actions is available and nothing is masked ("MuZero only masks
  legal actions at the root"). No terminal detection inside the tree.
- Selection: pUCT `Q̄(s,a) + P(s,a)·√N(s)/(1+N(s,a))·(c1 + log((N(s)+c2+1)/c2))`,
  c1 = 1.25, c2 = 19652.
- Q is normalized with MinMaxStats initialized to the known bounds
  [−1, 1] (pseudocode: "establishes identical behaviour to AlphaZero in
  board games"). Unvisited children score Q̄ = 0 on the normalized scale,
  as in the pseudocode's `ucb_score` (equivalent to treating them as
  losses; AlphaZero-go uses a different first-play value — noted in the
  comparison doc).
- Two-player signs: values are stored from the perspective of the player
  to move at that node; the parent sees Q(s,a) = r + γ·(−v(child)) with
  γ = 1. (The pseudocode's sign handling is ambiguous — it uses
  `child.value()` without negation in `ucb_score`; this negamax form is
  the standard reading and the one `muzero-tictactoe` tested.)
- Expansion: one leaf per simulation: g on (parent hidden state, action),
  then f. Backup along the path with the sign flip each ply.
- Root noise in self-play: Dirichlet(α) with α = `dirichlet_scale / n²`
  (`dirichlet_scale = 10`, ≈ the paper's 0.03 at 19×19 and 0.12 at 9×9),
  fraction 0.25.
- Move choice: sample ∝ visit counts for the first `temperature_moves`
  moves, then argmax (pseudocode: temperature 1 for the first 30 moves,
  then 0). Presets: 5×5 → 8, 9×9 → 7 (`alphazero-go`'s preset values).
- Simulations (paper 800): 5×5 preset 200, 9×9 preset 200 (both match
  `alphazero-go`'s presets).
- **Batched lockstep, not virtual loss** (my call): each worker thread
  plays G games at once; every step it runs one simulation's selection in
  each of its games, evaluates all leaves in one batched g+f call, then
  backs each one up. Each game's search is exactly the sequential
  pseudocode search; batching only spans independent games. The search
  code is written against a small `Model` interface
  (`initialInference(batch)`, `recurrentInference(batch, depths)`: each
  row's depth in the tree picks the network's per-step batch norm
  statistics, and one lockstep batch mixes depths) so it can be tested
  with a fake model.

### Self-play and the game

- Rules from `alphazero-go`: Tromp-Taylor area scoring, komi (5×5: 24.5 as
  in `alphazero-go`'s 5×5 preset — measured in `docs/results.md` to fail
  from a cold start: Black wins only by owning all 25 points, never won a
  recorded self-play game, and learned to pass; the later 5×5 runs use
  `--set komi=0.5`; 9×9: 7.5), suicide illegal, positional
  superko, game ends on two consecutive passes or at `max_moves` (default
  2n²; at the cap the position is scored as it stands).
- A stored game: observations are recomputed from the move list (the
  boards are replayed), plus per move the root visit distribution over
  n²+1 actions and the root search value; the result for Black.
- No resignation (the paper does not use it for MuZero's self-play;
  `alphazero-go` does in its preset — a deliberate difference, recorded in
  the comparison doc).

### Targets (pseudocode `make_target`, board-game form)

For a sampled position t of a game of length T (T moves), unroll step
k = 0..K (K = `unroll_steps` = 5):

- value target z_{t+k} = final outcome from the perspective of the player
  to move at t+k (`td_steps = max_moves`, discount 1: pure Monte Carlo
  return, "For board games, we bootstrap directly to the end of the
  game"). Ties (possible only with integer komi) are 0.
- policy target = stored visit distribution at t+k.
- A game the players ended (two passes): at t+k = T, value target 0, no
  policy loss (the pseudocode's empty policy target; its loss is undefined,
  so the term is masked), and reward target = the outcome for the player
  who made the last move (the pseudocode's `last_reward`); for t+k > T
  (absorbing states) value 0, reward 0, no policy loss.
- A game the move cap stopped was truncated, not ended (my call: the
  pseudocode stops self-play at `max_moves` without saying how such a game
  is targeted): reward target 0 on every edge (its last move was an
  ordinary move), value target at t+k = T the score at the cap from the
  view of the player to move (the best estimate of that position's
  value), and for t+k > T no value, reward or policy target (all masked).
- actions a_{t+1..t+K} = the game's moves. The pseudocode's `sample_batch`
  takes `g.history[i:i+K]`, so it never unrolls past T: an unroll near the
  end is shorter, and its recurrent steps are scaled by 1/len(actions).
  This project always unrolls K steps and pads past T with uniformly
  random actions from all n²+1, as muzero-general does — my call, not a
  gap in the pseudocode.
- Sampling: games uniform from the replay window, positions uniform in the
  game (App. G: "For board games, states are sampled uniformly"). Uniform
  over positions overall is the closer match; implementation picks a
  position uniformly over all stored positions.

### Loss and training (App. G, pseudocode `update_weights`)

- loss = Σ_k scale_k · [ℓᵖ(πₖ, pₖ) + ℓᵛ(zₖ, vₖ) (+ ℓʳ if reward head)]
  + c‖θ‖², with ℓᵖ cross-entropy, ℓᵛ squared error, averaged over the
  batch. Each term has a per-sample mask (policy past the end; value and
  reward past the cap of a truncated game); the mean still divides by B.
- scale_0 = 1, scale_k = 1/K for k ≥ 1 (the pseudocode's weights: its
  1/len(actions) is 1/K whenever the unroll stays inside the game, and this
  project always unrolls K steps; the paper's text says 1/K for every head
  — noted as a discrepancy, pseudocode followed).
- The gradient flowing back into a hidden state through g is multiplied
  by 0.5. Placement follows the pseudocode, which applies
  `hidden_state = scale_gradient(hidden_state, 0.5)` after each
  `recurrent_inference`: sᵏ → g_{k+1} is scaled for k ≥ 1, while s⁰ (the
  representation's output) receives g₁'s gradient unscaled. The paper's
  text says "at the start of the dynamics function", i.e. every call —
  noted as a discrepancy, pseudocode followed.
- Backpropagation through time: forward stores every step's activations
  (one set per unroll step for g and f); backward runs k = K..0.
- Batch norm: training mode with each call's own batch statistics; step k
  updates only its own set of running statistics (see "Three functions"),
  which inference at depth k uses.
- Optimizer: SGD momentum 0.9, weight decay 1e-4 on conv/FC weights
  (gradient + c·w, matching TF's `l2_loss` = Σw²/2), copied from
  `alphazero-go`.
- Learning rate: the pseudocode's exponential decay,
  lr = lr_init · 0.1^(step / lr_decay_steps), per training step. Presets
  pick lr_init and decay to fit the run length (9×9: 0.02 decaying by 10×
  over the run's planned steps — my call; the paper's 0.01 is for batch
  2048).
- Batch size: 5×5 128, 9×9 256 (paper 2048).
- Optional symmetry augmentation (`symmetry_augment`, default **off**:
  the paper does not mention it). When on, one random symmetry is applied
  to the whole unroll — observation, action planes and policy targets.

### The training loop

Synchronous generations, as in `alphazero-go` (the paper runs actors and a
learner asynchronously, refreshing the actors' network every 1000 steps;
synchronous generations are the scaled-down equivalent and make the
comparison exact): each generation plays `games_per_generation` self-play
games with the latest network, adds them to a replay window of the most
recent `replay_games` games, then runs `train_steps` SGD steps. Presets
(starting points, tuned by measurement):

| | 5×5 | 9×9 |
|---|---|---|
| generations | 60 | 150 |
| games_per_generation | 200 | 200 |
| simulations | 200 | 200 |
| train_steps | 200 | 400 |
| batch_size | 128 | 256 |
| replay_games | 2000 | 2500 |

Run directory `runs/<name>/`: `config.cfg` (the resolved config),
`log.csv` (per generation: self-play time, mean game length, per-unroll-step
losses k = 0..K for policy and value, lr, arena results), checkpoints
`checkpoints/gen_NNNN.mzn`, `state.bin` (replay window + optimizer + RNG
for resume), a couple of SGF games per generation.

Config: a `key = value` preset file plus `--set key=value` overrides; the
architecture keys (`unroll_steps` among them, for the per-step batch norm
statistics) cannot change on resume.

Added later, off by default: league play (`league_size` > 0), where part
of each generation's games are played against a league of earlier networks
and the random bot, designed in
[`2026-11-20-league-design.md`](2026-11-20-league-design.md).

### When the paper's version fails

The paper's version is always the default (where the paper contradicts
itself, as on the board-game reward loss, the presets take the reading that
measured better; see Deviations, 10). If it fails in a way that
stops learning (e.g. batch norm saturating at generation 0, as
`alphazero-go` saw with deep uncalibrated networks; the unrolled dynamics
diverging; the value collapsing), a stabilising change may be added as an
**opt-in switch**, listed under Deviations, and `docs/results.md` reports
the paper version's failure and the switch's effect side by side. A
failed paper-version run, measured and explained, is a valid result.
Candidate switch already known: `bn_calibrate` (copied from
`alphazero-go`: set every BN layer's running statistics from one training-
mode pass at generation 0; here one K-step pass, which sets every unroll
step's statistics), default off.

## Evaluation

- **Arena** (in-project): a generation plays earlier generations and fixed
  bots at temperature 0 after a few sampled opening moves, no noise. Bots:
  uniform-random legal player and a uniform-prior / zero-value MCTS bot
  (searching with the real rules — a pure-search yardstick). Every
  `arena_every` generations, gen g plays gen g−`arena_gap` and the bots,
  `arena_games` games each, colors alternating; results go to `log.csv`.
- **GTP engine** `mzgo_gtp`: same command set as `alphazero-go`'s `gtp`
  (protocol_version, name, version, known_command, list_commands, quit,
  boardsize, clear_board, komi, play, genmove, undo, showboard,
  final_score), MCTS at temperature 0, no noise.
- **Head-to-head with `alphazero-go`**: `tools/gtp_match.py` copied from
  `alphazero-go`, run between `mzgo_gtp` and `alphazero-go`'s `gtp`, at
  matched settings. A matched `alphazero-go` 9×9 run (same presets where
  the two share keys, resignation off) is part of the results; existing
  `alphazero-go` checkpoints are used where their settings match. GNU Go
  is not installed on this Mac; if it is on another machine it can be
  used the same way, otherwise skipped.

## Diagnostics (the learning part)

Tools that only make sense because the model is learned:

1. **`latent_probe`** (from `muzero-tictactoe`): roll g forward k = 1..K'
   steps along real games; compare f on the imagined state with f on a
   fresh h of the real position reached — value error and policy total-
   variation distance per k.
2. **`rules_probe`**: how much policy mass f puts on moves that are
   illegal in the real position (occupied points, suicide, superko), at
   k = 0 (fresh h) and after k imagined steps. Plus: after an imagined
   capture, does the policy treat the captured points as playable? And
   after two imagined passes (game over), what does the value say?
3. **`board_probe`**: a linear readout trained offline (no gradient into
   the network) from hidden states to stones (black / white / empty per
   point), fitted on fresh h(o) states, then applied to imagined states
   after k steps. Reports accuracy per k, and separately on points whose
   state changed by capture. This is the closest MuZero has to "what does
   it think the board is" — without a decoder in training.
4. **Search statistics** per generation in the log: mean / max tree depth,
   fraction of simulations whose path continues past a point where the
   real game would have ended (two passes) — the tic-tac-toe finding,
   measured for Go.
5. **`analyze`**: for a position (SGF + move number, or a move list), print
   the root search: visits, priors, Q per move, the principal variation in
   imagined space, and the value along it.

## Testing

`assert`-based test executables per component under CTest, as in the
siblings, plus an integration smoke test (a 5×5 run of 2 generations, then
resume, then arena and GTP).

- Copied components keep their `alphazero-go` tests.
- Network: finite-difference gradient checks for each function and for the
  full unrolled loss (K = 2 on 5×5, small filters), including the min-max
  normalization, the 0.5 gradient scale and the masked absorbing-state
  policy terms; inference vs training-mode consistency; save/load
  round-trip.
- Search: with a fake `Model`, hand-checked pUCT selection, MinMaxStats,
  sign of backup, root masking (illegal moves get zero visits at the root
  and nonzero below it is allowed), Dirichlet mixing, lockstep batching
  giving the same result as sequential search.
- Targets: hand-built games with known outcomes — value signs alternate,
  absorbing states, random padding, policy mask.
- Determinism: same seed → identical log and checkpoint across thread
  counts (5×5, 1 generation).
- Learning sanity: a 5×5 run reaches a clear win rate against the random
  bot within a fixed small budget (threshold set from a measurement, not
  guessed).

## Documentation

- `README.md`: build, usage, presets, results summary, authorship.
- `docs/algorithm-explained.md`: walkthrough of MuZero on Go tied to
  `file:line` citations (the siblings have a citation checker; copy it).
- `docs/muzero-vs-alphazero.md`: every place the two Go projects differ.
- `docs/results.md`: everything measured, including failures.

## Deviations from the paper (complete list)

1. Sizes: blocks/filters, batch size, simulations, games, steps, replay
   window scaled down; 5×5 / 9×9 instead of 19×19.
2. Synchronous generations instead of asynchronous actors/learner.
3. Batched lockstep self-play across games (search per game unchanged).
4. Policy-loss masking past the end of the game (the pseudocode's empty
   policy target), and K-step unrolls padded with random actions past the
   end (the pseudocode instead stops the unroll at the end of the game).
5. Loss weights follow the pseudocode (scale 1 at k = 0) rather than the
   paper's text (1/K for all).
   Likewise the 0.5 hidden-gradient scale follows the pseudocode's
   placement (not applied between s⁰ and g₁).
6. Min-max denominator floor 1e-5.
7. Learning-rate values chosen for the smaller batch.
8. Dirichlet α scaled with board area; temperature move count scaled.
9. Any opt-in stabilising switch (default off), e.g. `bn_calibrate`; see
   "When the paper's version fails".
10. The reward loss for board games: App. G omits it, §3 and the pseudocode
    keep it. The presets follow §3 and the pseudocode (`reward_head = true`),
    because App. G's reading was measured to fail (`docs/results.md`). It is
    the one switch on by default in the presets.
11. Batch norm running statistics per unroll step (the paper is silent on
    batch norm across the unroll); see "Three functions".
12. Games stopped by the move cap are truncated, not ended (no reward, the
    score as the value at the cap, no targets past it); the pseudocode does
    not say how a game cut at `max_moves` is targeted.
13. Optional training-only rule (`no_pass_before`, default 0 = off): in
    self-play and league games a searching player may not pass before move
    N unless pass is its only legal move. On 9x9, self-play collapsed into
    games that end with pass-pass after a few moves, and league games
    against the random bot saw the network pass most of its moves; those
    passes were its own moves, so they became policy targets and reinforced
    passing. Search below the root, arena, GTP and match are unchanged.

## Left to the builder

File layout, class names, the exact config key spelling beyond those
named here, log column order, checkpoint binary format (versioned, little
-endian via the copied serializer), thread count defaults, how the
lockstep driver groups games, and the bot/arena details beyond the above.

## Authorship

Mark Castelluccio, designed and implemented with Claude Code (AI-assisted).
