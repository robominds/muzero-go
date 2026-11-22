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
   scored at least `league_replace_score` (default 0.6, raised from 0.55 by
   Mark; a draw counts 0.5) over at least `league_min_games` (default 20)
   games against it. With the presets (200 games, fraction 0.5, 3 slots) the
   members get 34, 34 and 32 games, so beating one takes 20.4, 20.4 and
   19.2 points (0.55 took 18.7, 18.7 and 17.6). Settings under which some
   member gets fewer than `league_min_games` games a generation are
   rejected, since that member could never be replaced.
   In conversation this was "0.55 over 40 games" from separate matches; using
   the self-play games instead saves those matches. A random-bot member must
   be beaten twice in a row (decision 10).
6. **One replacement per generation.** If several members are beaten, only the
   one with the highest score is replaced (ties: the lowest slot). This keeps
   the league from filling with copies of one network in a single step.
7. **Who replaces it.** The network that played the games: generation g − 1's
   network during generation g, i.e. `checkpoints/gen_{g-1}.mzn`.
8. **Held-out test.** The arena is unchanged. The rollout bot is never a league
   member, so `arena_rollout` is the held-out measure; `arena_random` is no
   longer held out once the league uses the random bot.
9. **Persistence.** The league (one entry per slot: random bot or a generation
   number, and whether it was beaten in the last generation) is part of
   `state.bin`, so resume continues it exactly. `league_size` cannot change
   on resume.
10. **Random members need two consecutive wins (Mark's choice).** A beaten
   network member is replaced at once, but a random-bot member only when it
   was also beaten in the previous generation. The random bot is the only
   opponent that never comes back once replaced (a replaced network is only
   one of many similar ones, and later networks join anyway), so one noisy
   win should not remove it. Each slot keeps a flag, "beaten last
   generation": every generation it becomes whether the slot was beaten,
   and false for the slot just replaced. A random member beaten but not
   replaced (its first strike, or another slot chosen) keeps the flag for
   the next generation; a generation without a win clears it. Decision 6
   still allows one replacement per generation, among the eligible slots.

11. **Option `league_value` (default true).** A 5×5 test showed the league
   making the model weaker. One hypothesis: league games train the value
   toward outcomes against the random bot ("what happens if the opponent
   plays randomly"), which biases the search. With `league_value = false`
   league games still train the policy (the learner's moves) and the
   dynamics/reward, but every value mask of their samples is 0. The flag is
   stored per game in the replay buffer (format version 2) and set when the
   league records enter the window, so it may change on resume and only
   affects games added afterwards. Off by default in the sense that true
   keeps decision 4 unchanged.

12. **Option `league_network_wins` (default 2, Mark's choice).** Network
   members now also need two consecutive wins by default, for the same
   noise rejection as decision 10: one lucky generation should not retire a
   member. With `league_network_wins = 1` a beaten network member is
   replaced at once (decision 10's original network rule, together with
   `league_reset_on_replace = false`, decision 13); random members
   always need two. The same "beaten last generation" flag serves both. It
   may change on resume. Since the default changed, an old league run
   resumed with a new build uses 2 unless its config sets 1.

13. **Option `league_reset_on_replace` (default true, Mark's choice).**
   After a replacement the league has a new member, stronger or equal, and
   the learner's next generation should prove itself afresh against
   everyone, so all strikes clear (every slot's "beaten last generation"
   flag becomes false, not only the replaced slot's). Without a replacement
   the flags are whether each slot was beaten, as before. With
   `league_reset_on_replace = false`, other slots keep their strike, as in
   decisions 10 and 12. It may change on resume; an old league run resumed
   with a new build uses true unless its config sets false.

## Non-goals

- No AlphaStar-style exploiter agents or prioritized opponent sampling.
- No rollout-bot members (it stays held out).
- No change to arena, GTP, probes or match tools.

## What is left to the builder

Internal structure of the two-model lockstep in self-play; file formats'
byte layouts (versioned, old versions still load); log column names beyond
those in the plan.
