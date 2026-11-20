#!/usr/bin/env bash
# End-to-end smoke test: a tiny 5x5 run (2 generations, resumed to 3), every
# probe and analyze on generation 3, a scripted GTP session, a gtp_match.py
# game pair and an mzgo_match. Tiny settings, at most 2 threads, under a minute.
# Usage: integration_smoke.sh BUILD_DIR [SOURCE_DIR]
set -euo pipefail

BUILD=$(cd "${1:?usage: integration_smoke.sh BUILD_DIR [SOURCE_DIR]}" && pwd)
SRC=${2:-$(cd "$(dirname "$0")/.." && pwd)}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

RUN="$TMP/run"
CKPT="$RUN/checkpoints/gen_0003.mzn"
TINY=(--set workers=2 --set train_threads=2 --set generations=2
      --set repr_blocks=1 --set dyn_blocks=1 --set filters=8 --set value_hidden=8
      --set games_per_generation=4 --set simulations=8 --set train_steps=4 --set batch_size=8
      --set arena_games=2 --set arena_simulations=8 --set rollout_bot_simulations=8)

fail() { echo "FAIL: $*" >&2; exit 1; }
expect() {  # expect NAME PATTERN FILE
    grep -q -- "$2" "$3" || { cat "$3" >&2; fail "$1: no '$2' in output"; }
}

echo "== train: new run, 2 generations"
"$BUILD/mzgo_train" --config "$SRC/configs/5x5.cfg" --run "$RUN" "${TINY[@]}" >"$TMP/train1.txt" 2>&1 \
    || { cat "$TMP/train1.txt" >&2; fail "train"; }
[ -f "$RUN/checkpoints/gen_0002.mzn" ] || fail "no gen 2 checkpoint"

echo "== train: resume to 3"
"$BUILD/mzgo_train" --resume "$RUN" --set workers=2 --set train_threads=2 --set generations=3 \
    >"$TMP/train2.txt" 2>&1 || { cat "$TMP/train2.txt" >&2; fail "resume"; }
[ -f "$CKPT" ] || fail "no gen 3 checkpoint"
[ "$(wc -l <"$RUN/log.csv")" -eq 4 ] || fail "log.csv should have a header and 3 rows"

echo "== probes and analyze on gen 3"
for p in latent_probe rules_probe board_probe; do
    extra=()
    [ "$p" = board_probe ] && extra=(--epochs 2)
    "$BUILD/mzgo_$p" --checkpoint "$CKPT" --games 4 --k 2 --workers 2 ${extra[@]+"${extra[@]}"} >"$TMP/$p.txt" 2>&1 \
        || { cat "$TMP/$p.txt" >&2; fail "$p"; }
done
expect latent_probe "mean policy TV distance" "$TMP/latent_probe.txt"
expect rules_probe "illegal" "$TMP/rules_probe.txt"
expect board_probe "accuracy" "$TMP/board_probe.txt"
"$BUILD/mzgo_analyze" --checkpoint "$CKPT" --moves "C3 B3" --simulations 8 --top 3 >"$TMP/analyze.txt" 2>&1 \
    || { cat "$TMP/analyze.txt" >&2; fail "analyze"; }
expect analyze "principal variation" "$TMP/analyze.txt"

echo "== gtp session"
printf 'protocol_version\nboardsize 5\ngenmove b\nplay w C3\nfinal_score\nquit\n' \
    | "$BUILD/mzgo_gtp" --checkpoint "$CKPT" --simulations 8 >"$TMP/gtp.txt" 2>"$TMP/gtp.err" \
    || { cat "$TMP/gtp.txt" "$TMP/gtp.err" >&2; fail "gtp"; }
# Only stdout carries GTP responses; stderr is the engine's log.
gtp_fail() { cat "$TMP/gtp.txt" >&2; echo "--- stderr" >&2; cat "$TMP/gtp.err" >&2; fail "$1"; }
[ "$(grep -c '^=' "$TMP/gtp.txt")" -eq 6 ] || gtp_fail "gtp: expected 6 '=' responses"
! grep -q '^?' "$TMP/gtp.txt" || gtp_fail "gtp: error response"
grep -q '^= 2$' "$TMP/gtp.txt" || gtp_fail "gtp: protocol_version should answer '= 2'"
grep -Eq '^= ([A-HJ-T][1-5]|pass|PASS)$' "$TMP/gtp.txt" || gtp_fail "gtp: genmove b should answer a vertex or pass"
grep -Eq '^= ([BW]\+[0-9.]+|0)$' "$TMP/gtp.txt" || gtp_fail "gtp: final_score should answer a score"

echo "== gtp_match.py: 2 games"
ENGINE="$BUILD/mzgo_gtp --checkpoint $CKPT --simulations 8"
python3 "$SRC/tools/gtp_match.py" --a "$ENGINE" --b "$ENGINE --seed 2" --games 2 --size 5 --komi 24.5 \
    --out "$TMP/gtp_match" >"$TMP/gtp_match.txt" 2>&1 || { cat "$TMP/gtp_match.txt" >&2; fail "gtp_match.py"; }

expect gtp_match "Elo A - B" "$TMP/gtp_match.txt"
[ "$(grep -c . "$TMP/gtp_match/results.csv")" -ge 3 ] || fail "gtp_match: results.csv should hold 2 games"

echo "== mzgo_match: random vs rollout:8"
"$BUILD/mzgo_match" --a random --b rollout:8 --games 2 --board-size 5 --komi 24.5 --workers 2 \
    >"$TMP/match.txt" 2>&1 || { cat "$TMP/match.txt" >&2; fail "match"; }

expect match "wins" "$TMP/match.txt"
# "A random: X wins / B rollout:8: Y wins / Z draws, ..." -- X + Y + Z must be the 2 games played.
total=$(sed -n 's/^A .*: \([0-9]*\) wins \/ B .*: \([0-9]*\) wins \/ \([0-9]*\) draws.*/\1 \2 \3/p' "$TMP/match.txt" | awk '{print $1+$2+$3}')
[ "${total:-}" = 2 ] || { cat "$TMP/match.txt" >&2; fail "match: wins + draws should sum to 2"; }

echo "PASS: integration_smoke"
