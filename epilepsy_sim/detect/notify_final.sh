#!/usr/bin/env bash
# Ring the phone (ntfy) when detect/general.py finishes completely (full cap included),
# or if it crashes / the process disappears.
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="${2:-$DIR/run_general.log}"
TOPIC="$(cat "$DIR/ntfy_topic.txt")"
PID="${1:-}"

call() {
  curl -s -m 30 -H "Title: $1" -H "Priority: urgent" -H "Tags: $3" -d "$2" "https://ntfy.sh/$TOPIC" >/dev/null || true
}

while true; do
  if grep -q "saved .*general.png" "$LOG" 2>/dev/null; then
    g=$(grep "\[general\] Full cap" "$LOG" | sed 's/.*normalised: //; s/  (.*//')
    f=$(grep "\[finetune k=2\] Full cap" "$LOG" | sed 's/.*finetuned //; s/ | (.*//')
    call "All seizure models DONE" "Full cap new user: $g | fine-tuned: $f" "tada"
    echo "notified: done"; exit 0
  fi
  if grep -q "Traceback" "$LOG" 2>/dev/null; then
    call "Seizure model run CRASHED" "$(grep -v -i warn "$LOG" | tail -1 | cut -c1-180)" "rotating_light"
    echo "notified: crash"; exit 1
  fi
  if [ -n "$PID" ] && ! tasklist //FI "PID eq $PID" 2>/dev/null | grep -q "$PID"; then
    sleep 5
    grep -q "saved .*general.png" "$LOG" 2>/dev/null && continue
    call "Seizure model run STOPPED" "Process ended before finishing the full-cap step (maybe low memory)." "warning"
    echo "notified: stopped"; exit 2
  fi
  sleep 30
done
