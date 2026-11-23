#!/bin/bash
# Builds the tracking-view recording for $SEGMENT under $DATA/tview/$SEGMENT.
# Three arms share one clip: casa tracks CasA's detections, persist tracks the
# persistent donor emulator, and indep tracks an emulator with independent
# noise. Each arm is tracked, exported, scored by Waymo's evaluator, marked,
# and recorded with its topics under /casa, /persist or /indep. A base
# recording of the CasA log carries /points and /tf. merge_mcap.py joins the
# four recordings into tview.mcap, which opens with viz/tview_layout.json.
# Reads SEGMENT, DATA, PYTHON, DETECTIONS and SEED from the environment; the
# official evaluator runs in Docker.
set -euo pipefail

CASA_LOG="$DATA/staged/$SEGMENT.mcap"
TVIEW_DIR="$DATA/tview/$SEGMENT"
PARQUET_ROOT="$DATA/parquet"

if [ ! -f "$CASA_LOG" ]; then
  echo "tview.sh: no staged CasA log at $CASA_LOG, run make stage" >&2
  exit 1
fi
mkdir -p "$TVIEW_DIR"

# The staged arms get their own names: each arm's recording below is written
# to $TVIEW_DIR/ARM.mcap, which must not be the file the tracker reads.
"$PYTHON" stage/stage_segment.py --parquet-root "$PARQUET_ROOT" \
  --segment "$SEGMENT" --emulate "$SEED" --emulator donor \
  --out "$TVIEW_DIR/staged-persist.mcap"
"$PYTHON" stage/stage_segment.py --parquet-root "$PARQUET_ROOT" \
  --segment "$SEGMENT" --emulate "$SEED" --emulator independent \
  --out "$TVIEW_DIR/staged-indep.mcap"

for arm in casa persist indep; do
  case "$arm" in
    casa) log="$CASA_LOG" ;;
    *) log="$TVIEW_DIR/staged-$arm.mcap" ;;
  esac
  tracks="$TVIEW_DIR/$arm.csv"
  details="$TVIEW_DIR/$arm.tsv"
  marks="$TVIEW_DIR/${arm}_marks.csv"

  ./build/tracker --segment "$log" --export "$tracks"
  PYTHON="$PYTHON" PARQUET_ROOT="$PARQUET_ROOT" \
    eval/run_official.sh "$SEGMENT" "$tracks" "$details"
  "$PYTHON" eval/marks.py --details "$details" \
    --parquet-root "$PARQUET_ROOT" --segment "$SEGMENT" \
    --tracks "$tracks" --out "$marks"
  ./build/tracker --segment "$log" --record "$TVIEW_DIR/$arm.mcap" \
    --topic-prefix "/$arm" --marks "$marks"
done

./build/tracker --segment "$CASA_LOG" --record "$TVIEW_DIR/base.mcap"

"$PYTHON" eval/merge_mcap.py --out "$TVIEW_DIR/tview.mcap" \
  "$TVIEW_DIR/base.mcap" "$TVIEW_DIR/casa.mcap" \
  "$TVIEW_DIR/persist.mcap" "$TVIEW_DIR/indep.mcap"

echo "tview: $TVIEW_DIR/tview.mcap"
echo "layout: viz/tview_layout.json"
