#!/usr/bin/env bash
# "Call" the phone via ntfy when the ear-hook (both ears) step of detect/general.py finishes.
# Also fires if the run crashes or the process disappears, so silence never means "still running".
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$DIR/run_general.log"
TOPIC="$(cat "$DIR/ntfy_topic.txt")"
PID="${1:-}"

call() {  # $1 = title, $2 = message, $3 = tags
  curl -s -m 30 -H "Title: $1" -H "Priority: urgent" -H "Tags: $3" -d "$2" "https://ntfy.sh/$TOPIC" >/dev/null || true
}

while true; do
  if grep -q "\[finetune k=2\] Ear-hook both ears" "$LOG" 2>/dev/null; then
    g=$(grep "\[general\] Ear-hook both ears" "$LOG" | sed 's/.*normalised: //; s/  (.*//')
    f=$(grep "\[finetune k=2\] Ear-hook both ears" "$LOG" | sed 's/.*normalised: //; s/ (.*//')
    call "Ear-hook both ears: DONE" "New user: $g | After 2 own seizures: $f" "ear,white_check_mark"
    echo "notified: done"
    exit 0
  fi
  if grep -q "Traceback" "$LOG" 2>/dev/null; then
    call "Seizure model run CRASHED" "$(grep -v -i warn "$LOG" | tail -1 | cut -c1-180)" "rotating_light"
    echo "notified: crash"
    exit 1
  fi
  if [ -n "$PID" ] && ! tasklist //FI "PID eq $PID" 2>/dev/null | grep -q "$PID"; then
    call "Seizure model run STOPPED" "Process ended before the ear-hook step finished (maybe low memory)." "warning"
    echo "notified: stopped"
    exit 2
  fi
  sleep 30
done
