# League play: design

Mark Castelluccio, designed and written with Claude Code (AI-assisted).

## Why

On 9×9, self-play keeps beating its own earlier generations (0.5–0.95) but
loses to the random-move bot for 90+ generations (`docs/results.md`,
`9x9-s1` and `9x9-pp-s1`). The random bot never passes and drops stones
everywhere, including inside "our" territory, which leads to positions
self-play never visits. A league adds opponents that do not play like the
current network, so those positions enter training.

## The design (Mark's)

- A league of `league_size` opponent slots. Training starts with every slot
  held by the **random-move bot** (`RandomPlayer`). This is not the untrained
  network: by generation 1 the model already beat that 0.75, so the slots
  would be emptied at once.
- Each generation, a fraction of the self-play games is played by the current
  network against league members instead of against itself.
- When the network beats a member, that network replaces the member in its
  slot.

## Decisions (Claude's calls, agreed in conversation unless marked)

1. **Off by default.** `league_size = 0` keeps today's behavior, bit for bit.
2. **Game split.** With `league_size > 0`, L = round(games_per_generation ×
   `league_fraction`) of the generation's games are league games (default
   fraction 0.5); the rest are ordinary self-play. League games 2k and 2k + 1
   play member k mod league_size, the learner taking Black in the first and
   White in the second, so every member meets the learner in both colours
   whatever league_size is (an earlier "game j plays member j mod
   league_size" gave each member one colour only when league_size is even).
   The fraction counts games, not positions: league games against the random
   bot run long, so they can be most of the generation's new positions.
3. **How a member moves.** A random-bot member plays `RandomPlayer`. A network
   member plays MuZero search with the same settings as self-play (noise and
   temperature on), so it is as varied an opponent as the learner.
4. **What is trained from a league game.** Every position keeps its value
   target (the outcome, from the view of the side to move), its reward
   targets and its actions for the dynamics. Only the learner's own positions
   get a policy target; the opponent's positions have policy mask 0 (copying
   a random bot's choices would be wrong, and AlphaStar likewise trains only
   the main agent's actions).
5. **"Beats" (Claude's call).** Decided from the generation's own league
   games, so no extra games are played. A member is beaten when the learner
   scored at least `league_replace_score` (default 0.55; a draw counts 0.5)
   over at least `league_min_games` (default 20) games against it. With the
   presets (200 games, fraction 0.5, 3 slots) the members get 34, 34 and 32
   games. Settings under which some member gets fewer than
   `league_min_games` games a generation are rejected, since that member
   could never be replaced.
   In conversation this was "0.55 over 40 games" from separate matches; using
   the self-play games instead saves those matches.
6. **One replacement per generation.** If several members are beaten, only the
   one with the highest score is replaced (ties: the lowest slot). This keeps
   the league from filling with copies of one network in a single step.
7. **Who replaces it.** The network that played the games: generation g − 1's
   network during generation g, i.e. `checkpoints/gen_{g-1}.mzn`.
8. **Held-out test.** The arena is unchanged. The rollout bot is never a league
   member, so `arena_rollout` is the held-out measure; `arena_random` is no
   longer held out once the league uses the random bot.
9. **Persistence.** The league (one entry per slot: random bot or a generation
   number) is part of `state.bin`, so resume continues it exactly.
   `league_size` cannot change on resume.

## Non-goals

- No AlphaStar-style exploiter agents or prioritized opponent sampling.
- No rollout-bot members (it stays held out).
- No change to arena, GTP, probes or match tools.

## What is left to the builder

Internal structure of the two-model lockstep in self-play; file formats'
byte layouts (versioned, old versions still load); log column names beyond
those in the plan.
