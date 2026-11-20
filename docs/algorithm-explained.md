# MuZero for Go, from Scratch

How three networks learn a model of Go that nobody gave them, and then plan
inside it — walked through against the actual C++ in this repository, in the
order the code runs. This is the sequel to two projects:
`alphazero-go`, which plays the same game with the same rules, encoding and
layers but searches over real boards, and `muzero-tictactoe`, which built
MuZero's machinery on a game too small for the learned model to have much
to learn. Where an idea was already explained there (PUCT, the sign flip,
the visit-count policy target, the Monte Carlo return), this document recaps
it in a few lines and spends its words on what Go changes.

**Language:** C++17, no dependencies (BLAS optional) · **Networks:**
representation `h` 18 planes → `F` channels, dynamics `g` `F` + 1 planes →
`F` channels, prediction `f` `F` channels → {n² + 1 logits, 1 value}; 3 + 3
residual blocks × 32 filters on 5×5, 6 + 6 × 64 on 9×9 · **Results:**
measured in [`docs/results.md`](results.md) (the 5×5 run, then 9×9); this
document explains the algorithm and makes no claim about strength.

Every code excerpt below is quoted from the repository, abridged where
marked, and captioned with the file and the line where the quoted function
or type is defined. `python3 tools/check_citations.py` checks those line
numbers against the source.

## Contents

- [00 — What is new here](#00--what-is-new-here)
- [01 — Three functions on a Go board](#01--three-functions-on-a-go-board)
- [02 — An action is a plane](#02--an-action-is-a-plane)
- [03 — Keeping the hidden state in range](#03--keeping-the-hidden-state-in-range)
- [04 — The rules enter once](#04--the-rules-enter-once)
- [05 — Signs, MinMaxStats and the unvisited child](#05--signs-minmaxstats-and-the-unvisited-child)
- [06 — What a game teaches each unroll step](#06--what-a-game-teaches-each-unroll-step)
- [07 — Learning through K copies of g](#07--learning-through-k-copies-of-g)
- [08 — Batch norm across unroll steps](#08--batch-norm-across-unroll-steps)
- [09 — Self-play in lockstep](#09--self-play-in-lockstep)
- [10 — The generation loop, and resuming it](#10--the-generation-loop-and-resuming-it)
- [11 — Looking inside the model](#11--looking-inside-the-model)
- [12 — Concept map](#12--concept-map)

---

## 00 — What is new here

**Concept: planning with a learned model**

A reinforcement learning agent that plans needs a *model*: something that
answers "if I play this move here, what happens?". AlphaZero is handed a
perfect one. `alphazero-go`'s search calls `Board::play` on a copy of the
position at every node, asks the real rules which moves are legal there,
and when two passes end the game it scores the board exactly. Nothing below
the search root is a guess except the network's prior and value.

MuZero learns the model too. It reads the real board once per move, at the
root, and from then on every node of its search tree is an *imagined* state
that a network produced from its parent and a move. It is never told what
a capture is, that a stone cannot be placed on another stone, that a ko
cannot be retaken at once, or that two passes end the game. It is only
trained so that, along the moves that were really played, the predictions
made from its imagined states match what the real game later showed: the
search's move preferences and the final result.

`muzero-tictactoe` showed all of the machinery and also its limits: on a
9-point board with no captures, the learned model has almost nothing to
learn, and the search wandered far past the end of games that cannot last
more than nine moves. Go has real dynamics. A move can remove a group of
stones from anywhere on the board; a point that was occupied becomes
playable again; whether a move is legal can depend on the position before
the last one; the game ends by agreement (two passes) rather than by a
pattern on the board. Those are exactly the things the learned model has to
pick up, and the diagnostics in chapter 11 measure whether it did.

What is new relative to the two siblings, chapter by chapter:

| | `alphazero-go` | `muzero-tictactoe` | here |
|---|---|---|---|
| state in the tree | a real `Board` | a 32-float latent vector | an `F × n × n` hidden state, spatial like the board |
| how a move enters the model | — (the rules play it) | a one-hot vector | one extra input plane (ch. 02) |
| legality below the root | real rules at every node | none | none, on a game where illegal moves are common (ch. 04) |
| end of game in the tree | exact, scored by the rules | none | none; two passes do not stop the search (ch. 04, 06) |
| unvisited child in pUCT | Q = 0 (a draw), or exact for a game-ending pass | Q̄ = 0 (a loss) | Q̄ = 0 (a loss) (ch. 05) |
| value target | final result | n-step return, by default reaching the end (Monte Carlo) | final result, with absorbing states past the end (ch. 06) |
| training step | one position | a 5-step unroll of MLPs | a 5-step unroll of residual convnets, BPTT by hand (ch. 07) |
| batch norm | one call per layer per step | — (no batch norm) | `g` called K times per step, `f` K + 1 times (ch. 08) |
| batching the search | virtual loss inside one tree | one game at a time | lockstep across games (ch. 09) |

The code that is *not* new was copied from `alphazero-go` into namespace
`mzgo` and is identical in behaviour: the rules (`Board`, `Game` with
positional superko and Tromp-Taylor scoring), the input encoding, the
symmetries, the layers with their hand-written backward passes, the naive
and BLAS backends, the thread pool, serialization, SGD with momentum, SGF
and the GTP engine's command set. That keeps every difference in the
comparison ([`muzero-vs-alphazero.md`](muzero-vs-alphazero.md)) a
difference of algorithm, not of codebase.

---

## 01 — Three functions on a Go board

**Concept: a latent dynamics model, and what the network sees**

MuZero splits AlphaZero's one network into three functions, trained
together:

```
 real position (18 planes, n × n)
        │
        ▼
 h  representation      stem 3×3 conv → repr_blocks residual blocks → min-max
        │
        ▼
 s⁰ (F × n × n) ──▶ f ──▶ policy logits (n² + 1), value
        │
        │   a¹ (one plane)
        ▼   ▼
 g  dynamics            stem 3×3 conv on F + 1 planes → dyn_blocks blocks → min-max
        │
        ▼
 s¹ (F × n × n) ──▶ f ──▶ policy, value
        │   a²
        ▼   ▼
        g ──▶ s² ──▶ f ...
```

- **`h(o) → s⁰`** reads exactly what `alphazero-go`'s network reads. The
  paper says so for Go: "we encode the last 8 board states as in
  AlphaZero". Eighteen planes: own and opponent stones for the current
  position and the seven before it, then a black-to-move and a
  white-to-move plane.
- **`g(sᵏ⁻¹, aᵏ) → sᵏ`** replaces `Board::play`. Its input is the previous
  hidden state with one more plane for the move (chapter 02). With the
  reward head it also predicts a reward. The presets turn the head on; the
  built-in default leaves it off (chapter 06).
- **`f(sᵏ) → (pᵏ, vᵏ)`** is AlphaZero's two heads with no trunk of its own:
  a policy over the n² points and pass, and a value in [−1, 1] for the side
  to move at that state.

```cpp
MuZeroNetwork::MuZeroNetwork(const NetConfig& cfg, uint64_t seed)
    : cfg_(cfg),
      repr_(cfg.planes(), cfg.filters, cfg.reprBlocks),
      dyn_(cfg.filters + 1, cfg.filters, cfg.dynBlocks),
      pred_(cfg.filters, cfg.boardSize, cfg.valueHidden),
      reward_(cfg.rewardHead ? ValueHead(cfg.filters, cfg.area(), cfg.valueHidden) : ValueHead()) {
```
*src/network.cpp:81 — `MuZeroNetwork::MuZeroNetwork`*

`h` and `g` are both a `Trunk`: the AlphaZero trunk, a 3×3 stem convolution
with batch norm and ReLU, then residual blocks of two convolutions with a
skip connection. Only the stem's input width differs: 18 planes for `h`,
`F + 1` for `g`.

```cpp
    // Stem: conv3x3 -> BN -> ReLU.
    stemConv.forward(be, x, B, n, col, convOut, pool);
    bnForward(stemBn, convOut, M, c.stemOut.data(), c.stemXhat.data(), c.stemInvStd.data(), mode, pool);
    reluForward(c.stemOut.data(), size, pool);
    // Residual blocks: conv -> BN -> ReLU -> conv -> BN -> + input -> ReLU.
    ...
```
*src/blocks.cpp:156 — `Trunk::forward`, abridged*

**The hidden state is spatial.** In `muzero-tictactoe` a latent was a flat
vector of 32 floats. Here it is `F` feature maps of n × n, the same shape
as a trunk activation in `alphazero-go`: 64 × 9 × 9 = 5,184 floats per state
on 9×9. That is the paper's choice for Go, and it matters. A convolution
computes the same function at every point, so `g` can learn "a stone placed
here, next to an opponent group with one liberty, removes that group" once
and apply it anywhere on the board, the way the rules do. A flat vector
would have to learn every location separately.

There is still **no decoder**. Nothing in the network maps a hidden state
back to stones, and nothing in the loss asks the hidden state to look like
a board. If a channel of `s³` happens to light up where black stones are,
that is because it helped predict the policy and value, not because it was
asked to. Chapter 11's board probe asks how far that went.

The search talks to the network through a two-call interface, so its tests
can use a fake model:

```cpp
class ModelSession {               // one per worker thread; not thread-safe
public:
    virtual ~ModelSession() = default;
    virtual void initialInference(const float* observations /*[B][planes][area]*/, int B, ModelOutput& out) = 0;
    virtual void recurrentInference(const float* hidden /*[B][hiddenSize]*/, const int* actions, int B,
                                    ModelOutput& out) = 0;
};
```
*include/mzgo/model.hpp:27 — `ModelSession`*

`initialInference` is `h` then `f`, once per move. `recurrentInference` is
`g` then `f`, once per simulation. Everything below the root goes through
the second call, with no board anywhere in its arguments.

| preset | `repr_blocks` | `dyn_blocks` | `filters` | `value_hidden` | parameters (`h` / `g` / `f`) |
|---|---|---|---|---|---|
| 5×5 | 3 | 3 | 32 | 32 | 60,928 / 65,248 / 2,293 |
| 9×9 | 6 | 6 | 64 | 64 | 454,400 / 481,472 / 18,877 |

(The paper used 16 blocks × 256 channels for `h` and `g` on 19×19.) `h` plus
`f` has exactly the layers of `alphazero-go`'s network of the same size;
`g` is a second trunk of the same depth. So one simulation, which calls `g`
and `f`, costs about as much as one `alphazero-go` evaluation, which calls
its trunk and heads.

---

## 02 — An action is a plane

**Concept: action encoding for a spatial dynamics model**

`g` needs the move as an input. The paper's Go encoding: "a normal action
... is encoded as an all zero plane, with a single one in the position of
the played stone. A pass is encoded as an all zero plane." So the move is
one more channel next to the `F` channels of the hidden state:

```cpp
void writeActionPlane(const int* actions, int stride, int B, int A, float* plane) {
    std::fill(plane, plane + static_cast<size_t>(B) * A, 0.0f);
    for (int b = 0; b < B; ++b) {
        const int a = actions[static_cast<size_t>(b) * stride];
        assert(a >= 0 && a <= A);
        if (a < A) plane[static_cast<size_t>(b) * A + a] = 1.0f;
    }
}
```
*src/network.cpp:39 — `writeActionPlane`*

`A` is the number of points, so the pass action, `A`, writes nothing. Two
consequences are worth seeing.

First, the plane puts the move *where it happens*. `g`'s first 3×3
convolution sees the 1 together with the hidden state's channels at that
point and its eight neighbours. Whether the move captures depends on the
neighbouring groups' liberties, which can be far away, so one layer cannot
decide it alone; the residual blocks give the dynamics enough receptive
field (the stem and six blocks are thirteen 3×3 convolutions, which see
13 points in every direction, more than a 9×9 board) to propagate "this group has just lost its last
liberty" across the board.

Second, a pass is distinguishable from every other move only by the
absence of a 1. That is enough: an all-zero plane is a perfectly good
input, and `g` can learn that "nothing on the action plane" means "same
stones, other player to move". What it cannot see is *how many* passes in
a row there have been, unless it carries that in the hidden state itself.
That becomes important in chapter 04.

Inside the network every layer works channel-major, `[C][B][n·n]`, while
the search stores hidden states sample by sample. In training the action
plane is written straight into channel `F` of the tensor that already holds
`sᵏ⁻¹`, so `g`'s input is assembled without a copy:

```cpp
    for (int k = 1; k <= K; ++k) {
        float* in = ws.state_[k - 1].data();     // s_{k-1} in channels 0..F-1
        writeActionPlane(batch.actions.data() + (k - 1), K, B, A, in + F * M);
        dyn_.forward(be, in, B, n, ws.dynCaches_[k - 1], mode, pool);
```
*src/network.cpp:140 — `MuZeroNetwork::forward`, abridged*

With `symmetry_augment = true` (off by default; the paper does not mention
augmentation), one of the eight board symmetries is applied to the whole
unroll: the observation, every action and every policy target, so the
action planes line up with the rotated board.

---

## 03 — Keeping the hidden state in range

**Concept: normalizing a recurrent state**

`g` is applied to its own output again and again: up to K = 5 times in a
row in training, and in play as many times as the search tree is deep. A function composed with itself can drift: if each call scales its
input by 1.1, five calls scale it by 1.6 and twenty by 6.7, and the layers
downstream, trained on states of one scale, see states of another. The
paper's fix (Appendix G) is to rescale every hidden state to [0, 1], the
range of the action plane it will be concatenated with:

```cpp
void minMaxForward(const float* x, int n, float* y, MinMaxCache& c) {
    ...
    c.min = x[lo];
    const float range = x[hi] - x[lo];
    c.floored = range < kMinMaxFloor;
    c.range = c.floored ? kMinMaxFloor : range;
    for (int i = 0; i < n; ++i) y[i] = (x[i] - c.min) / c.range;
}
```
*src/blocks.cpp:52 — `minMaxForward`, abridged*

The minimum and maximum are taken over the whole hidden state of one
sample (all `F × n × n` values), not per channel. It runs after `h` and
after every `g`, in training and in search alike. The 1e-5 floor on the
denominator (a constant hidden state would otherwise divide by zero) is
this project's choice; the paper does not say.

Because min and max are themselves functions of the input, the gradient is
not just `dy / range`. Moving the minimum element moves every output, and
so does moving the maximum. The backward pass adds those two corrections on
the argmin and argmax elements:

```cpp
    // y_i = (x_i - m) / r with m = min x and r = max x - m. Differentiating:
    //   dy_i/dx_i = 1/r (every element, directly)
    //   dy_i/dm   = (y_i - 1)/r   -> lands on the argmin element
    //   dy_i/dmax = -y_i / r      -> lands on the argmax element
    ...
    for (int i = 0; i < n; ++i) dx[i] = static_cast<float>(dy[i] / r);
    if (c.floored) {
        dx[c.argmin] = static_cast<float>(dx[c.argmin] - sumDy / r);
    } else {
        dx[c.argmin] = static_cast<float>(dx[c.argmin] + (sumDyY - sumDy) / r);
        dx[c.argmax] = static_cast<float>(dx[c.argmax] - sumDyY / r);
    }
```
*src/blocks.cpp:68 — `minMaxBackward`, abridged*

A finite-difference check covers this, including the floored case
(`test_minmax_gradient`, `test_minmax_constant_input` in
`tests/test_blocks.cpp`). Getting it wrong would not crash anything: the
network would still train, on a slightly wrong gradient, and the only
symptom would be a worse model. That is why every backward pass in this
project has a gradient check.

> **vs AlphaZero-Go.** No equivalent: AlphaZero's network is never applied
> to its own output, so there is nothing to drift.

---

## 04 — The rules enter once

**Concept: root-only legality, and what the model must learn instead**

The paper: "MuZero only masks legal actions at the root". The search is
handed the real game's legal moves exactly once, when it starts:

```cpp
    // The only place the rules enter: children for the legal moves only,
    // prior renormalized over them (uniform if the model gave them all 0).
    softmax(logits, dims_.actions(), probs_.data());
    double sum = 0.0;
    for (int a : legal) sum += probs_[a];
    ...
    if (cfg_.addNoise) {
        const std::vector<double> eta = dirichlet(rng, cfg_.dirichletAlpha, static_cast<int>(legal.size()));
        ...
    }
    nodes_[0].firstChild = 1;
    nodes_[0].numChildren = static_cast<int>(legal.size());
    for (size_t i = 0; i < legal.size(); ++i) nodes_.push_back(Node{legal[i], priors[i]});
```
*src/mcts.cpp:36 — `Search::beginRoot`, abridged*

`legal` comes from `Game::legalActions()`: empty points that are not
suicide and do not repeat an earlier whole-board position (positional
superko), plus pass. The Dirichlet noise of self-play is mixed over those
moves only. Every node below the root gets a child for every one of the
n² + 1 actions, with the prior straight from `f`:

```cpp
    // Below the root: every action, prior straight from the model, unmasked.
    softmax(logits, dims_.actions(), probs_.data());
    const int first = static_cast<int>(nodes_.size());
    for (int a = 0; a < dims_.actions(); ++a) nodes_.push_back(Node{a, probs_[a]});
```
*src/mcts.cpp:113 — `Search::expandLeaf`, abridged*

A test pins both halves: a fake model that loves an illegal move gets zero
visits for it at the root, and the same move is still chosen below the root.

```cpp
    for (int a = 0; a < kActions; ++a) {
        if (a != 0 && a != 5 && a != kPass) assert(r.visits[a] == 0 && r.policy[a] == 0.0f);
        total += r.visits[a];
    }
    assert(total == 50 && s.rootEdges().size() == 3);
    // Below the root nothing is masked: the favourite is chosen there.
```
*tests/test_mcts.cpp:160 — `test_root_masking`, abridged*

On tic-tac-toe this was a curiosity. On Go it is the central difficulty,
because every one of the rules below shows up in ordinary play:

- **Illegal moves.** Midway through a 9×9 game a large share of the points
  hold a stone. Below the root, nothing stops the search from "playing" on one of
  them. What `g` does with such a move is undefined: it never saw one along
  a real game (chapter 06 explains the one place random actions do appear,
  after the end). The protection is indirect. `f`'s policy targets are root
  visit distributions, which are zero on illegal moves because the root
  never has those children, so `f` learns to put little prior on occupied
  points — on states computed by `h` from a real board, and, if `g` is good,
  on imagined states too. And an unvisited child scores as a loss
  (chapter 05), so a move with a tiny prior is rarely tried at all.
- **Captures.** When a move removes a group, the points it occupied become
  empty and playable. Nothing tells `g` this happened; it has to infer from
  its own hidden state that the group had one liberty and that the move
  filled it. A model that only learned "a move adds a stone" would keep the
  captured stones in its imagination and steer the search away from the
  freed points.
- **Ko.** Retaking a ko immediately would repeat the previous position.
  `h` can see that: the observation holds the last eight positions, so a
  ko capture that just happened is visible in the history planes. Below the
  root there are no history planes, only the hidden state, so `g` has to
  carry "a ko was just taken here" forward itself. Positional superko,
  which looks at the whole game, would need far more memory than that; it
  is rare in practice.
- **Passes past the end of the game.** Two consecutive passes end a real
  game. In the tree they end nothing: there is no terminal detection below
  the root, so after two imagined passes the search keeps expanding, and
  `f` is asked for a value of a state that does not exist in the real game.
  What the model is taught about those states is the subject of chapter
  06's absorbing states.

The search measures the last point directly. For every simulation it
checks whether the imagined path contains two passes in a row, counting
the real move before the root when that was a pass:

```cpp
    bool prevPass = previousMoveWasPass_, pastEnd = false;
    for (size_t i = 1; i < path_.size(); ++i) {
        const bool isPass = nodes_[path_[i]].action == dims_.area();
        pastEnd = pastEnd || (isPass && prevPass);
        prevPass = isPass;
    }
    pastEnd_ += pastEnd;
```
*src/mcts.cpp:113 — `Search::expandLeaf`, abridged*

Averaged over a generation's self-play it is the `past_end_fraction` column
of `log.csv`, next to `mean_depth` and `max_depth`. That is the
tic-tac-toe finding ("the search goes on past the end of the game")
measured for Go.

> **vs AlphaZero-Go.** `alphazero-go` builds each child by playing the move
> on a copy of the parent's real board, expands only legal moves, and stops
> a descent at a node whose game is over (two passes or the move cap),
> backing up the exact Tromp-Taylor result. None of these four problems
> exist there.

Why accept all this? Because it is the point. A planner that only needs
`initialInference` and `recurrentInference` works the same whether the
"rules" are Go's or an Atari emulator's that nobody can query. Go is a
setting where we *do* have the rules, so we can check how close the
learned model came to them — which is what chapter 11 does.

---

## 05 — Signs, MinMaxStats and the unvisited child

**Concept: pUCT over a learned model, two-player backup**

Selection is the pseudocode's `ucb_score`:

  score(a) = Q̄(s, a) + P(s, a) · √N(s) / (1 + N(s, a)) · (c1 + log((N(s) + c2 + 1) / c2))

with c1 = 1.25 and c2 = 19652. For a few hundred simulations the log term
is about 0.01, so the exploration weight is effectively the constant 1.25.

```cpp
int Search::selectChild(const Node& parent) const {
    const float np = static_cast<float>(parent.visits);
    const float pbC = std::log((np + cfg_.pbCBase + 1.0f) / cfg_.pbCBase) + cfg_.pbCInit;
    const float sqrtN = std::sqrt(np);
    ...
        // Unvisited: Q̄ = 0 on the normalized scale (as bad as a loss), so
        // only the prior term can draw the search there.
        const float q = c.visits > 0 ? minmax_.normalize(edgeQ(c)) : 0.0f;
        const float score = q + c.prior * sqrtN / (1.0f + static_cast<float>(c.visits)) * pbC;
        // Ties (every score is 0 on the root's first simulation): larger
        // prior, then lower action (children are in ascending action order).
```
*src/mcts.cpp:75 — `Search::selectChild`, abridged*

Three pieces of this need explaining.

**Signs.** Every node stores values from the view of the player to move
*at that node*. The parent's player wants the opposite, so an edge's Q is
the reward on the edge plus the negated child value:

```cpp
    float mean(const Node& n) const { return n.visits ? n.valueSum / static_cast<float>(n.visits) : 0.0f; }
    float edgeQ(const Node& child) const { return child.reward + cfg_.discount * -mean(child); }
```
*include/mzgo/mcts.hpp:81 — `Search`, abridged*

Backup walks the path from the leaf up, flipping the sign once per ply:

```cpp
    float v = value;
    for (int i = static_cast<int>(path_.size()) - 1; i >= 0; --i) {
        Node& n = nodes_[path_[i]];
        n.valueSum += v;
        n.visits += 1;
        if (i > 0) minmax_.update(edgeQ(n));
        v = n.reward + cfg_.discount * -v;
    }
```
*src/mcts.cpp:113 — `Search::expandLeaf`, abridged*

The pseudocode's own sign handling is ambiguous (its `ucb_score` uses
`child.value()` without negation); this negamax reading is the standard
one and the one `muzero-tictactoe` tested. `test_backup_signs` in
`tests/test_mcts.cpp` checks it by hand.

**Discount 1.** The discount γ = 1, so Q = r − v(child). In a board game
the only payoff is the result at the end, and a game of 60 moves is worth
exactly as much as a game of 80; discounting would make the value of a
position depend on how far it is from the end, which the network cannot
see. With γ = 1 and no reward head, Q is just −v(child): AlphaZero's
negamax.

**MinMaxStats and known bounds.** MuZero was designed for Atari too, where
rewards have any scale, so it maps Q into [0, 1] with the minimum and
maximum seen in the tree before adding it to the prior term. For board
games the pseudocode starts the range at the known bounds [−1, 1], which
it says "establishes identical behaviour to AlphaZero in board games":

```cpp
void MinMaxStats::update(float q) {
    lo_ = std::min(lo_, q);
    hi_ = std::max(hi_, q);
}
float MinMaxStats::normalize(float q) const { return hi_ > lo_ ? (q - lo_) / (hi_ - lo_) : q; }
```
*src/mcts.cpp:12 — `MinMaxStats::update`, with `normalize` below it*

Without a reward head every value comes out of a `tanh` and Q stays inside
[−1, 1], so the range never widens and the normalization is the fixed map
Q̄ = (Q + 1) / 2. (With the reward head, which the presets turn on, r − v
can reach ±2, and the range grows as such values appear. A real game's
return never goes above 1, so a Q above 1 means the model's rewards and
values disagree with the absorbing-state targets; `docs/results.md` finds
such values in the 5×5 reward-head run.)

**The unvisited child.** An edge nobody has visited has no Q. The
pseudocode scores it 0 on the *normalized* scale — and Q̄ = 0 means Q = −1:
an unvisited move is treated as a loss. A child is tried only when its
prior term alone beats the visited siblings' Q̄ plus their (shrinking)
prior terms. So the search follows the prior closely and goes deep along
it rather than wide.

That matters more here than in AlphaZero. Below the root, the prior is the
only thing keeping the search off occupied points and other illegal moves
(chapter 04). With a pessimistic first-play value, a move whose prior is
small stays unvisited; with an optimistic or neutral one it would be tried
sooner, and its value would come from a state the model was never trained
on.

> **vs AlphaZero-Go.** `alphazero-go`'s `selectEdge` scores an unvisited
> edge at Q = 0 on the raw [−1, 1] scale, a draw, except that a pass which
> would end the game gets its exact Tromp-Taylor result before it is ever
> visited. Its exploration term is a fixed `c_puct = 1.5` on raw Q. MuZero's
> 1.25 on Q̄ = (Q + 1)/2 is the same as `c_puct = 2.5` on raw Q, so for
> visited children this search is also more prior-driven, by a factor of
> about 1.7.

One small detail: the root's value from `initialInference` is not backed
up (as in the pseudocode), so on the first simulation N(root) = 0, every
score is 0, and the tie goes to the largest prior. The first simulation
always follows the prior's favourite.

---

## 06 — What a game teaches each unroll step

**Concept: Monte Carlo targets, absorbing states**

A finished self-play game is stored as its moves, the stones of every
position, the root visit distribution at every move and the result. From
it, training cuts samples: a starting position t, the next K = 5 moves, and
a target for each of the K + 1 states of the unroll:

```cpp
    for (int k = 0; k < K; ++k) {
        const int a = (t + k < T) ? g.moves[t + k] : rng.below(A);
        batch.actions[static_cast<size_t>(b) * K + k] = transformAction(sym, a, n);
    }

    for (int k = 0; k <= K; ++k) {
        const int p = t + k;
        ...
        if (p < T) {
            const float* src = g.policies.data() + static_cast<size_t>(p) * A;
            for (int a = 0; a < A; ++a) pol[transformAction(sym, a, n)] = src[a];
            batch.policyMask[slot] = 1.0f;
            batch.valueTargets[slot] = g.outcomeFor(p);
        } else {
            batch.policyMask[slot] = 0.0f;
            batch.valueTargets[slot] = 0.0f;
        }
        // The move that ended the game (made at T-1) earns its mover the outcome.
        batch.rewardTargets[slot] = (k >= 1 && p == T) ? g.outcomeFor(T - 1) : 0.0f;
    }
```
*src/game_record.cpp:40 — `fillSample`, abridged*

**Policy target: the search, not the move.** At every real position the
target is the root's visit distribution, MuZero's and AlphaZero's
"policy improvement operator": search with the current network finds a
better policy than the network alone, and the network is trained towards
it. Because the root only has legal children, these targets put zero mass
on every illegal move — the only lesson about legality the model ever gets.

**Value target: the result, from the mover's view.** `outcomeFor(p)` is the
final result for the player to move at position p, so the targets alternate
in sign along the unroll. In reinforcement-learning terms this is a
**Monte Carlo** target: the actual return of the episode, rather than a
**bootstrapped** one that would plug the search's own value estimate in
after n steps. The pseudocode's board-game configuration sets
`td_steps = max_moves`, which makes its n-step return reach past the end of
every game ("For board games, we bootstrap directly to the end of the
game"); the stored root values are never used as targets. A Monte Carlo
target is unbiased but noisy (one game decides it), which is fine when
there are hundreds of games per generation and the outcome is ±1. The root
values are still recorded in each game, so a bootstrapped or Reanalyze
target could be added later.

**Absorbing states.** The unroll runs a fixed K steps, but a sample that
starts near the end of a game runs out of moves. For t + k ≥ T the state
lies past the end of the game, and the pseudocode treats it as *absorbing*:
the game has ended, nothing more can happen, all future rewards are 0, so
the value target is 0. The pseudocode's policy target there is empty, which
leaves its loss undefined; this implementation masks the term
(`policyMask = 0`). And the actions fed to `g` past the end are drawn
uniformly from all n² + 1 actions — the pseudocode leaves this open, and
random padding is the common reading (muzero-general does the same).
`test_short_game_padding` in `tests/test_game_record.cpp` builds such a
sample by hand.

**What the absorbing target teaches on Go.** Follow it through. The state
after the game's final pass is position T: an absorbing state, value
target 0. So the model is trained to say "0" right after the game-ending
pass, whatever the score. Without a reward head the search sees that pass
as Q = 0 − v(after) ≈ 0: ending the game looks like a draw to both sides,
even when one of them is far ahead. The information is not lost from the
model — the value at the position *before* the final pass is trained on
the true result — but the edge that ends the game carries no payoff. The
same holds for a game stopped by the move cap: its last state is absorbing
too.

That is what the reward head fixes. With `reward_head = true`, the reward
target on the edge into position T is the outcome for the player who made
that last move, and Q = r − v(after) ≈ r: ending a won game is worth +1
again. The paper reads both ways. Its Appendix G omits the reward loss for
board games ("in board games without intermediate rewards, we omit the
reward prediction loss"), while its §3 loss and its pseudocode's
`update_weights` keep the reward term for every game. The first version of
this project followed Appendix G. It showed exactly the problem above
([`docs/results.md`](results.md)). On 5×5 with komi 0.5, after `C3 pass`,
Black's pass would end the game and win by 24.5, yet it got a prior of
0.028 and no visits. The run without the head had fallen far behind: its
generation 12 lost 1–39 to the reward-head run's generation 12. Both
presets now set `reward_head = true`. The built-in default stays off, so
older configs keep their meaning.

The head is not the whole answer. With it, the 5×5 model beats its own
earlier generations but loses to a random player. In the positions where
it goes wrong, the search rates a move that keeps the game going at about
+1.4, above anything a real result can be. It rates the pass that would end
a won game at about +0.4. So the model plays on inside its own territory
until it fills its own last eye. Where those values above 1 come from is
not yet measured (`docs/results.md`, "What is not yet known").

**Sampling.** A training batch picks positions uniformly over every
position in the replay window, the paper's "states are sampled uniformly"
for board games, so a long game contributes more positions than a short
one:

```cpp
    for (int b = 0; b < B; ++b) {
        const long pick = rng.below(static_cast<int>(total));
        // First game whose cumulative count exceeds pick.
        const size_t gi = static_cast<size_t>(
            std::upper_bound(prefix_.begin(), prefix_.end(), pick) - prefix_.begin());
        ...
        fillSample(games_[gi], t, K, history, sym, rng, out, b);
    }
```
*src/replay_buffer.cpp:32 — `ReplayBuffer::sampleBatch`, abridged*

> **vs AlphaZero-Go.** `alphazero-go` trains on single positions: one
> observation, its visit distribution, its result. MuZero trains on the
> position plus the next five moves and five more sets of targets, because
> what it learns is the *transition*, and a transition can only be checked
> by what comes after it.

---

## 07 — Learning through K copies of g

**Concept: the unrolled loss, backpropagation through time**

One training step runs `h` on the starting observations, then `g` K times
along the recorded moves, and `f` on every state:

```cpp
// s0 = norm(h(obs)); (p0, v0) = f(s0); for k = 1..K:
// s_k = norm(g(s_{k-1}, plane(a_k))), r_k = reward(s_k), (p_k, v_k) = f(s_k).
void MuZeroNetwork::forward(TrainWorkspace& ws, const TrainBatch& batch, BnMode mode) {
    ...
    repr_.forward(be, ws.obs_.data(), B, n, ws.reprCache_, mode, pool);
    normalizeHidden(ws.reprCache_.output(), B, F, A, ws.hidden_[0].data(), ws.minMax_[0].data(),
                    ws.state_[0].data(), pool);
    pred_.forward(be, ws.state_[0].data(), B, n, ws.heads_[0], mode, pool);

    for (int k = 1; k <= K; ++k) {
        ...
        dyn_.forward(be, in, B, n, ws.dynCaches_[k - 1], mode, pool);
        normalizeHidden(ws.dynCaches_[k - 1].output(), B, F, A, ws.hidden_[k].data(), ws.minMax_[k].data(),
                        ws.state_[k].data(), pool);
        ...
        pred_.forward(be, ws.state_[k].data(), B, n, ws.heads_[k], mode, pool);
    }
}
```
*src/network.cpp:140 — `MuZeroNetwork::forward`, abridged*

The same `g` weights run K times, so they cannot hold the activations their
backward pass needs. Every call gets its own cache (`dynCaches_[k - 1]`,
`heads_[k]`); the weights and their gradient accumulators stay in the
layer, and the K calls' gradients simply add up there.

**The loss.** Per unroll step: cross-entropy between the policy target and
`softmax(f)`'s policy (masked past the end), squared error between the
value target and the value (and the reward with the head). The steps are
weighted:

```cpp
    for (int k = 0; k <= K; ++k) {
        const double scale = k == 0 ? 1.0 : 1.0 / K;
        ...
                const double w = scale * batch.policyMask[row(b)] / B;
                float* d = ws.dLogits_[k].data() + b * actions;
                for (int a = 0; a < actions; ++a)
                    d[a] = static_cast<float>(w * (std::exp(l[a] - logZ[b]) * sumPi - target[a]));
                ws.dValues_[k][b] = static_cast<float>(scale * 2.0 * (values[b] - batch.valueTargets[row(b)]) / B);
```
*src/network.cpp:174 — `MuZeroNetwork::loss`, abridged*

Step 0 has weight 1 and every later step 1/K. That is the pseudocode's
`update_weights`, which appends `(1.0, ...)` for the initial inference and
`(1.0 / len(actions), ...)` for each recurrent one. The paper's text says
every head's loss is scaled by 1/K; where the two disagree this project
follows the pseudocode, and the spec lists it as a deviation. The effect:
the real position's prediction, the one search actually uses at the root,
counts as much as all K imagined ones together. Plus weight decay, applied
by the optimizer as `c · w` on conv and linear weights (TensorFlow's
`l2_loss`, as in `alphazero-go`).

**Backpropagation through time.** The backward pass runs k = K down to 0.
The gradient arriving at `sᵏ` has two sources: `f`'s input gradient at step
k, and — for k < K — the gradient that `g`'s call at step k + 1 sends back
into its hidden-state input. From `sᵏ` it goes through the min-max
normalization and into whichever trunk produced it: `g` at step k, or `h`
for k = 0.

```cpp
    for (int k = K; k >= 0; --k) {
        const float* s = ws.state_[k].data();
        pred_.backward(be, s, ws.dLogits_[k].data(), ws.dValues_[k].data(), B, n, ws.heads_[k], dHidden, pool);
        ...
        if (k < K) addInto(dHidden, dNext, size, pool);
        // Through the min-max normalization, sample by sample.
        ...
        if (k >= 1) {
            dyn_.backward(be, ws.state_[k - 1].data(), dRaw, B, n, ws.dynCaches_[k - 1], ws.dIn_.data(), pool);
            // scale_gradient(s_{k-1}, 0.5) when s_{k-1} came from g (k >= 2);
            // s0 (k = 1) gets the full gradient.
            const float scale = k >= 2 ? hiddenGradScale : 1.0f;
            ...
                for (size_t j = begin; j < end; ++j) dNext[j] = scale * dIn[j];
        } else {
            repr_.backward(be, ws.obs_.data(), dRaw, B, n, ws.reprCache_, nullptr, pool);
        }
    }
```
*src/network.cpp:243 — `MuZeroNetwork::backward`, abridged*

`g`'s input gradient has `F + 1` channels; the last one belongs to the
action plane, which is data, not a learned quantity, and is dropped.

**The half gradient, and where it goes.** The paper (Appendix G): "we also
scale the gradient at the start of the dynamics function by 1/2. This
ensures that the total gradient applied to the dynamics function stays
constant." The pseudocode implements it as
`hidden_state = scale_gradient(hidden_state, 0.5)` *after* each
`recurrent_inference`, so the edge `sᵏ → gₖ₊₁` is scaled for k ≥ 1 and the
edge `s⁰ → g₁` is not: the representation receives `g₁`'s gradient in full.
The paper's wording ("at the start of the dynamics function", every call)
would scale that edge too. Again the code follows the pseudocode, and the
spec lists the discrepancy.

Why does a factor of 1/2 keep anything constant? Ignore for a moment what
each `g` does to the gradient and count terms. Without the scale, the
gradient reaching `sᵏ` carries the losses of every step from k to K, so the
early calls of `g` see the sum of up to K steps' errors, and the weight
gradient of `g`, summed over its K calls, grows roughly with K² terms,
each weighted 1/K: linearly in K. With the scale, the loss at step j
reaches the call at step k attenuated by 2^−(j−k), so each call sees its
own step's error plus a geometric tail worth at most as much again; summed
over K calls with weight 1/K each, the total stays bounded no matter how
far the unroll goes. That is the counting argument behind "stays
constant".

The second reason is stability. Backpropagation through a recurrence
multiplies the gradient by each step's Jacobian in turn, and a product of
K Jacobians of a 13-layer residual network can grow or shrink quickly. The
half damps the long chains: `g` is trained mostly by the near-term
consequences of each transition, and the distant ones only nudge it. It
also keeps the dynamics from dominating the representation's gradient,
which is the only path by which the real board shapes the hidden state.

The scaled gradient is no longer the gradient of any loss, so the gradient
checks run with `hiddenGradScale = 1` (the exact gradient of the unrolled
loss, `test_unroll_gradient`), and a separate test checks that the scale
does what it says: because backpropagation is linear in the scale, the
gradient with 0.5 must be the average of the gradients with 0 and 1, and
`f`'s gradient must not change at all.

```cpp
        if (name.compare(0, 5, "pred.") == 0) {
            assert(g0[t] == gHalf[t] && g1[t] == gHalf[t]);   // bit for bit
            continue;
        }
        ...
        for (size_t i = 0; i < params[t].size; ++i) {
            const double mean = (static_cast<double>(g0[t][i]) + g1[t][i]) / 2;
            assert(std::fabs(gHalf[t][i] - mean) <= 1e-5 * scale);
```
*tests/test_network.cpp:259 — `test_gradient_scale_half`, abridged*

**What a step costs.** A sample's forward pass is `h + K·g + (K+1)·f` and
the backward pass the same again: with K = 5, six trunk passes and six
head passes where `alphazero-go` does one of each, so roughly 5–6 times the
work per sample. Measured with `mzgo_bench` for the 9×9 preset (6 + 6
blocks, 64 filters, BLAS): 453 ms per training step at batch 256, K = 5, on
6 threads, on a Mac Studio (M5 Max) while another training job used 12 of
its 18 cores. The training step's threads split every layer's work in
fixed chunks and the batch sums run in one order, so the bits do not
depend on the thread count (`test_pool_bit_identical`).

---

## 08 — Batch norm across unroll steps

**Concept: batch statistics in a recurrent network, and a calibration trap**

Every trunk and head uses batch norm. In training it normalizes with the
current batch's mean and variance and updates exponential running
averages (momentum 0.1); in search, which sees one position at a time, it
uses the running averages. That is ordinary. What is new is that `g` is
called K times per training step and `f` K + 1 times, each call with a
different input distribution:

- In a training step, `g`'s call at step k normalizes with the statistics
  of its own batch, the states `sᵏ⁻¹` of all B samples. Each call is
  self-consistent.
- Each call also updates the running statistics, so `g`'s running averages
  get K updates per step and `f`'s K + 1, in unroll order. The result is a
  blend of every step's statistics, the later steps weighted a little more
  (the last call's update is the most recent; with K = 5 and momentum 0.1
  the relative weights run from 0.9⁴ ≈ 0.66 for `g`'s first call, or
  0.9⁵ ≈ 0.59 for `f`'s, to 1 for the last).
- In search, `g` is applied at every depth with that one blend, and `f` is
  applied to the root's `s⁰` and to deep imagined states with the same
  blend.

If imagined states drift in distribution as k grows, the network in search
mode does not compute quite what it computed in training. Min-max
normalization helps — every state is in [0, 1] — but it fixes only the
range, not the distribution inside it.

Every forward runs in one of four modes (`BnMode` in
`include/mzgo/blocks.hpp`): `Infer` for search, `Train` for a training step,
`Probe` for evaluating a loss without side effects, and `Calibrate`, which
*sets* the running statistics instead of averaging into them:

```cpp
    switch (mode) {
        case BnMode::Infer: bn.forwardInfer(x, M, y, pool); break;
        case BnMode::Train: bn.forwardTrain(x, M, y, xhat, invStd, true, pool); break;
        case BnMode::Probe: bn.forwardTrain(x, M, y, xhat, invStd, false, pool); break;
        case BnMode::Calibrate: bn.forwardTrain(x, M, y, xhat, invStd, true, pool, 1.0f); break;
    }
```
*src/blocks.cpp:16 — `bnForward`*

**The calibration trap.** `alphazero-go` found that a freshly initialized
deep network, whose running statistics are mean 0 and variance 1, computes
badly scaled activations in inference mode, and its first self-play games
collapse; it sets generation 0's statistics from one training-mode pass
(`bn_calibrate`, on by default there). This project has the same switch,
**off by default**, because the paper does not do it (chapter 10's "when
the paper's version fails" rule). Turning it on reveals the trap: a
Calibrate pass *sets* each layer's statistics to those of its input, so a
layer called several times keeps only its last call's. One K-step pass
would leave `g` calibrated to step K's input and `f` to `sᴷ` — not the
states search starts from. So calibration runs two passes:

```cpp
// A Calibrate pass SETS each layer's statistics to those of its input in
// that pass, and a layer reused in the unroll (g K times, f K + 1 times)
// keeps only its last call's statistics. One K-step pass would leave g with
// step K's statistics and f with s^K's. So two passes: first K = 1, which
// sets g (and the reward head) from its first call, on s0; then K = 0,
// which sets h and f from the real position's s0 and leaves g as it is.
void Trainer::calibrateBatchNorm() {
    ...
    for (int K : {1, 0}) {
        TrainWorkspace ws(netCfg, cfg_.batchSize, K, *backend_, pool_.get());
        net_->calibrateBatchNorm(ws, truncateUnroll(batch, K, netCfg));
    }
}
```
*src/trainer.cpp:250 — `Trainer::calibrateBatchNorm`, abridged*

`test_inference_matches_probe` in `tests/test_network.cpp` checks the
consequence exactly: after calibrating on a batch, inference on that batch
reproduces the training-mode forward, but only for the calls whose
statistics the layer kept (after K = 1, the recurrent inference; after
K = 0, the initial one).

> **vs AlphaZero-Go.** One forward per step, one set of statistics per
> layer: no blend and no trap. Its `bn_calibrate` is a single pass and on
> by default.

---

## 09 — Self-play in lockstep

**Concept: batching independent searches**

Every move of self-play is a full search: one `initialInference` and
`simulations` recurrent inferences, each a batch of one. A matrix-multiply
backend is wasted on batches of one. `alphazero-go` batches inside one tree
with *virtual loss*: it runs eight descents before evaluating any of them,
and marks each in-flight path as a temporary loss so the next descent goes
elsewhere. That changes the search — the eight descents do not see each
other's results, and the virtual losses bend the selection while they are
in flight.

This project batches *across games* instead. A worker thread plays a group
of `selfplay_group` = 8 games. Each step, every game whose search is still
running selects one leaf; all those leaves go to the network in one batched
`recurrentInference`; each answer goes back to its own tree:

```cpp
        // 2. One simulation in every unfinished search: one batched recurrent inference.
        ...
        for (size_t k = 0; k < live.size(); ++k) {
            if (live[k]->search.finished()) continue;
            const LeafRequest req = live[k]->search.selectLeaf();
            batch.push_back(static_cast<int>(k));
            actions.push_back(req.action);
            parents.insert(parents.end(), req.parentHidden, req.parentHidden + hs);
        }
        if (!batch.empty()) {
            session.recurrentInference(parents.data(), actions.data(), static_cast<int>(batch.size()), out);
            for (size_t b = 0; b < batch.size(); ++b)
                live[batch[b]]->search.expandLeaf(out.hidden.data() + b * hs, out.logits.data() + b * dims.actions(),
                                                  out.values[b], out.rewards[b]);
        }
```
*src/selfplay.cpp:32 — `playGroup`, abridged*

Games starting a new move share one batched `initialInference` the same
way, and a game that ends leaves the group, so the batch shrinks towards
the end of a group. Each game's tree is exactly the sequential pseudocode
search: selection, one evaluation, backup, in order. The search API is
split into `beginRoot` / `selectLeaf` / `expandLeaf` precisely so a driver
can interleave many trees this way. `test_lockstep_equals_sequential`
plays the same games in groups of 4 and of 1 against a fake model and
checks the records are identical.

MuZero needs no tree to be shared between moves either. `beginRoot` clears
the tree, because the new root must be `h` of the real position, not the
imagined state the previous search had for it.

**Exploration.** Root Dirichlet noise with α = `dirichlet_scale` / n² (10 / 81
≈ 0.12 on 9×9, the paper's 0.03 scaled from 361 points), mixed in at 25%
over the legal moves; and the move sampled in proportion to visits for
the first `temperature_moves` moves (7 on 9×9, 8 on 5×5), the most visited
afterwards. No resignation: the paper does not use it for MuZero, so every
game is played to two passes or the move cap of 2n² moves.

**Determinism.** Game i takes all its randomness, in a fixed order, from
its own `Rng(deriveSeed(seed, i))`, and statistics are reduced in game
order, so the records do not depend on how many worker threads play them
(`test_selfplay_workers_independent`). They do depend on the group size:
a batched matrix multiply can round differently from a single one.

---

## 10 — The generation loop, and resuming it

**Concept: synchronous generations; reproducibility**

The paper runs actors and a learner asynchronously, refreshing the actors'
network every 1000 steps. Scaled down, that becomes synchronous
generations, as in `alphazero-go`, which also makes the comparison exact.
One generation:

```cpp
void Trainer::runGeneration() {
    ...
    // 1. Self-play with the current network (only read while it plays).
        records = playSelfPlay(model, sp, cfg_.gamesPerGeneration, deriveSeed(cfg_.seed, g, 1), workers, &stats);
    ...
    // 2. Into the replay window.
    for (GameRecord& rec : records) replay_.add(std::move(rec));
    // 3. Training. Step s's batch comes from its own seed, so neither the
    // number of threads nor a resume changes which positions it sees.
    for (int s = 1; s <= cfg_.trainSteps; ++s) {
        Rng rng(deriveSeed(cfg_.seed, static_cast<uint64_t>(g), 2, static_cast<uint64_t>(s)));
        replay_.sampleBatch(cfg_.batchSize, K, cfg_.history, cfg_.symmetryAugment, rng, batch);
        const float lr = exponentialLr(cfg_.lrInit, cfg_.lrDecayRate, cfg_.lrDecaySteps, globalStep_);
        const UnrollLoss loss = net_->trainStep(*workspace_, batch);
        ...
        optimizer_.step(params, lr, pool_.get());
        ++globalStep_;
```
*src/trainer.cpp:296 — `Trainer::runGeneration`, abridged*

Then the checkpoint `gen_GGGG.mzn`, `state.bin`, a couple of SGF games,
the arena (every `arena_every` generations: generation g against
g − `arena_gap`, a rollout-MCTS bot that searches with the real rules, and
the random bot), and one row of `log.csv`. The learning rate is the
pseudocode's exponential decay, lr = lr_init · 0.1^(step / lr_decay_steps),
per SGD step: 0.02 falling tenfold over the 9×9 preset's 60,000 planned
steps (the paper's 0.01 was for batch 2048, against 256 here).

The `log.csv` row records the loss per unroll step, `policy_k0` …
`policy_k5` and `value_k0` … `value_k5`. Those columns are the first
diagnostic: `value_k0` is the real position's prediction, the later ones
the imagined positions'. A model that learns the dynamics has later-step
losses that fall along with step 0's; a gap that widens says the
imagination is not keeping up.

**Resume determinism.** Every random choice of generation g comes from a
seed derived from (seed, g, phase, step), never from a generator carried
across generations. So `state.bin` needs only the generation, the global
step, the network, the optimizer's velocity and the replay window, and a
run stopped after any generation and resumed writes exactly the bytes an
uninterrupted run would have written. The test does it:

```cpp
    Trainer::startNew(tinyConfig(), straight).run();
    ...
    Trainer::startNew(one, split).run();
    Trainer second = Trainer::resume(split, {"generations=2"});
    ...
    assert(readFile(checkpointPath(straight, 2)) == readFile(checkpointPath(split, 2)));
    assert(readFile(straight + "/state.bin") == readFile(split + "/state.bin"));
```
*tests/test_trainer.cpp:124 — `test_resume_matches_uninterrupted`, abridged*

A resume may override any key except the architecture keys (they fix the
weights' shapes) and `unroll_steps` (log.csv has one column per step).
A non-finite loss, or non-finite weights after a generation's last step,
stops the run before anything of that generation is written, so a blow-up
never reaches `state.bin`.

**When the paper's version fails.** Every default is the paper's. If it
fails in a way that stops learning, a stabilising change may be added as
an opt-in switch (default off), and `docs/results.md` reports the paper
version's failure and the switch's effect side by side. `bn_calibrate`
(chapter 08) and `reward_head` (chapter 06) are the two switches that
exist. `reward_head` is the exception to "default off". The paper
contradicts itself on it (chapter 06), Appendix G's reading was measured
to fail, and the presets turn it on.

---

## 11 — Looking inside the model

**Concept: measuring a model that has no decoder**

A win rate says whether the whole system plays well; it cannot say whether
the model learned Go or found some other way to predict good moves. Go has
the advantage over Atari that the real rules are available, so the learned
model can be checked against them. All three probes use the same
construction, on real games: start at position t, take `s⁰ = h(position t)`,
apply `g` along the moves the game really played, and compare `sᵏ` with
the real position t + k.

```cpp
        session->initialInference(obs.data(), B, out);
        ...
        for (int k = 0;; ++k) {
            ...
            for (int j = 0; j < static_cast<int>(active.size()); ++j) {
                const Start& s = *active[j];
                visit(s, k, out, j);
                if (s.last > k) {
                    ...
                    actions.push_back(games[s.game].moves[s.t + k]);
                }
            }
            if (next.empty()) break;
            session->recurrentInference(hidden.data(), actions.data(), static_cast<int>(next.size()), out);
```
*src/probes.cpp:42 — `forEachImagined`, abridged*

The games come from the rollout bot (classic UCT with random playouts on
the real rules) playing itself, not from self-play: legal, varied Go that
the network did not choose. Each probe runs `--k 5` imagined steps by
default; asking for more than the training unroll (`--k 10` with K = 5)
measures how the model extrapolates past what it was trained on.

**`mzgo_latent_probe`: does the imagination agree with a fresh look?** For
each k, the mean |v(sᵏ) − v(h(position t+k))| and the policy
total-variation distance ½ Σₐ |p(sᵏ)ₐ − p(h(position t+k))ₐ|.

```cpp
        softmax(out.logits.data() + static_cast<size_t>(row) * A, A, probs.data());
        const float* real = fresh.probs[s.game].data() + static_cast<size_t>(p) * A;
        double l1 = 0;
        for (int a = 0; a < A; ++a) l1 += std::fabs(static_cast<double>(probs[a]) - real[a]);
        ...
        r.valueError += std::fabs(static_cast<double>(out.values[row]) - fresh.values[s.game][p]);
        r.policyTv += 0.5 * l1;
```
*src/probes.cpp:149 — `latentProbe`, abridged*

*How to read it:* k = 0 is 0 by construction (0 up to rounding). Growth with k
is the drift of the imagination away from what the network would say on
the real position. The total-variation distance is the fraction of policy
mass that would have to move; 0.5 means half the prior is on different
moves. Note what this does *not* measure: the fresh `h` is the network's
own opinion, not the truth, so a small error means the imagination is
*consistent* with the network's view of real positions — not that either is
right.

**`mzgo_rules_probe`: has the model learned what is legal?** For each k,
how much of `softmax(f(sᵏ))` falls on moves that are illegal in the real
position t + k, split by kind:

```cpp
            for (int a = 0; a < A; ++a) {
                const bool legal = game.isLegal(a);
                row[a] = legal ? 0 : kIllegal;
                if (a < r.area() && board.at(a) != Color::Empty) row[a] |= kOccupied;
                else if (!legal && board.isLegal(a)) row[a] |= kSuperko;
            }
```
*src/probes.cpp:174 — `rulesProbe`, abridged*

*How to read it:* the `illegal` column at k = 0 shows whether `f` learned
legality from the visit-count targets at all (it sees a real board there);
the values at k ≥ 1 show whether `g` keeps track of where the stones are.
`occupied` is almost all of `illegal` (suicide and superko are rare).
`superko` is the ko lesson of chapter 04. `captured` is different: it is
the mass on points that held a stone at t and were freed by a capture
within the k steps — legal again, so mass there is *good*, a sign the model
knows the stones are gone. It is 0 at k = 0 by definition and has no fixed
baseline, so compare it across checkpoints and with the board probe's
changed points. Mass on illegal moves below the root is only harmful if the
search follows it, so read this together with the next one.

**`mzgo_board_probe`: what board does the model think it is?** A linear
readout — the same small softmax regression at every point, from the `F`
hidden channels at that point to {empty, own stone, opponent stone} — is
fitted on fresh `h` states of the probe games (no gradient into the
network), then applied to imagined states:

```cpp
        for (int i = 0; i < area; ++i) {
            const bool right = probeArgmax(probe, F, area, h, i) == stoneClass(r, p, i);
            br.points += 1;
            br.accuracy += right;
            if (r.stones[static_cast<size_t>(s.t) * area + i] != r.stones[static_cast<size_t>(p) * area + i]) {
                br.changedPoints += 1;
                br.changedAccuracy += right;
            }
        }
```
*src/probes.cpp:318 — `boardProbe`, abridged*

*How to read it:* at k = 0 the accuracy says how linearly the stones can
be read out of `h`'s state at all; compare it with the trivial baseline of
always answering "empty" (the fraction of empty points). The interesting
column is `changed` accuracy at k ≥ 1: the points whose content differs
between t and t + k — new stones and captured ones — which a model that
remembered the starting board without updating it would get wrong. One
caution: the readout was fitted on `h`'s states. If `g` encodes stones in
different channels than `h` does, the readout misses them even though the
information is there. Low accuracy on imagined states means "not in the
form `h` uses", not necessarily "not known". (Own and opponent swap every
ply, so a model that forgot whose move it is fails here too.)

**Search statistics in `log.csv`.** Per generation: `mean_depth` and
`max_depth` of each simulation's new leaf, `past_end_fraction` (chapter
04), `capped_games` (games ended by the move cap rather than two passes,
chapter 06), `mean_length`. A deep search is not good or bad in itself;
depth past the real end of the game is effort spent on states that do not
exist.

**`mzgo_analyze`: one search, in full.** For a move list, the root search's
visits, priors and Q for the top moves, the depth statistics, and the
principal variation — the most visited child, repeatedly — with the value
along it. Below the first move the PV exists only in the model's
imagination; it can contain moves on occupied points or a pass after the
game would have ended, and reading one is the most direct way to see what
the search believes. To see what the model makes of the end of a game,
give it a move list that ends with a pass and look at whether, and at
what value, the PV passes back.

> **vs AlphaZero-Go.** None of these probes has a meaning there: its search
> stands on real boards at every node, so there is no imagination to
> check. Its diagnostics are about strength (Elo ladders, references,
> GNU Go).

---

## 12 — Concept map

| Concept | Where |
|---|---|
| Observation planes (own / opponent, 8 positions, side to move) | `encodePlanes` — `src/encode.cpp:15` |
| The three functions | `MuZeroNetwork::MuZeroNetwork` — `src/network.cpp:81` |
| The model interface search sees | `ModelSession` — `include/mzgo/model.hpp:27` |
| Action as a plane, pass all zero | `writeActionPlane` — `src/network.cpp:39` |
| Hidden-state min-max normalization | `minMaxForward`, `minMaxBackward` — `src/blocks.cpp:52, 68` |
| Representation then prediction | `MuZeroNetwork::initialInference` — `src/network.cpp:329` |
| Dynamics then prediction | `MuZeroNetwork::recurrentInference` — `src/network.cpp:344` |
| Root-only legality, Dirichlet noise | `Search::beginRoot` — `src/mcts.cpp:36` |
| pUCT, unvisited child scored as a loss | `Search::selectChild` — `src/mcts.cpp:75` |
| Q normalization with known bounds | `MinMaxStats::update`, `MinMaxStats::normalize` — `src/mcts.cpp:12, 17` |
| Unmasked expansion, signed backup, past-end count | `Search::expandLeaf` — `src/mcts.cpp:113` |
| Targets, absorbing states, random padding | `fillSample` — `src/game_record.cpp:40` |
| Uniform position sampling | `ReplayBuffer::sampleBatch` — `src/replay_buffer.cpp:32` |
| The unroll forward | `MuZeroNetwork::forward` — `src/network.cpp:140` |
| Loss weights 1 and 1/K | `MuZeroNetwork::loss` — `src/network.cpp:174` |
| BPTT and the 0.5 hidden-gradient scale | `MuZeroNetwork::backward` — `src/network.cpp:243` |
| Batch norm modes | `bnForward` — `src/blocks.cpp:16` |
| Calibrating batch norm across the unroll | `Trainer::calibrateBatchNorm` — `src/trainer.cpp:250` |
| Lockstep self-play | `playGroup`, `playSelfPlay` — `src/selfplay.cpp:32, 113` |
| The generation loop | `Trainer::runGeneration` — `src/trainer.cpp:296` |
| Resume | `Trainer::resume` — `src/trainer.cpp:167` |
| Imagined rollouts along real games | `forEachImagined` — `src/probes.cpp:42` |
| Latent, rules and board probes | `latentProbe`, `rulesProbe`, `boardProbe` — `src/probes.cpp:149, 174, 318` |
| One search, printed | `analyzePosition` — `src/probes.cpp:347` |

---

Written against the source in this repository. Build it and run the 5×5
preset with:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMZ_USE_BLAS=ON
cmake --build build -j
./build/mzgo_train --config configs/5x5.cfg --run runs/five
./build/mzgo_rules_probe --checkpoint runs/five/checkpoints/gen_0060.mzn
```

Mark Castelluccio, designed and implemented with Claude Code (AI-assisted).
