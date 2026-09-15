#!/usr/bin/env bash
set -euo pipefail

S3_ROOT="s3://argoverse/datasets/av2/motion-forecasting"
DATA_ROOT="${AV2_DATA_ROOT:-/root/datasets/argoverse2/motion_forecasting_10k}"
TRAIN_COUNT="${AV2_TRAIN_COUNT:-11500}"
VAL_COUNT="${AV2_VAL_COUNT:-2300}"

if ! command -v s5cmd >/dev/null 2>&1; then
  echo "s5cmd is required" >&2
  exit 1
fi

mkdir -p "$DATA_ROOT/train" "$DATA_ROOT/val"
work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT

download_split() {
  local split="$1"
  local count="$2"
  local ids_file="$work_dir/${split}_ids.txt"
  local commands_file="$work_dir/${split}_commands.txt"

  echo "==> listing AV2 $split scenarios"
  s5cmd --no-sign-request ls "$S3_ROOT/$split/*" \
    | awk '{print $NF}' \
    | cut -d/ -f1 \
    | sort -u \
    | awk -v count="$count" 'NR <= count' > "$ids_file"

  : > "$commands_file"
  while IFS= read -r scenario_id; do
    destination="$DATA_ROOT/$split/$scenario_id"
    parquet="$destination/scenario_${scenario_id}.parquet"
    if [[ -f "$parquet" ]]; then
      continue
    fi
    mkdir -p "$destination"
    printf 'cp "%s/%s/%s/scenario_%s.parquet" "%s/"\n' \
      "$S3_ROOT" "$split" "$scenario_id" "$scenario_id" "$destination" \
      >> "$commands_file"
  done < "$ids_file"

  missing="$(wc -l < "$commands_file")"
  echo "==> $split: downloading $missing missing parquet files"
  if [[ "$missing" -gt 0 ]]; then
    s5cmd --no-sign-request run "$commands_file"
  fi
  found="$(find "$DATA_ROOT/$split" -name 'scenario_*.parquet' -type f | wc -l)"
  echo "==> $split: $found parquet files available"
}

# Download extra files because some scenarios are skipped when focal history is incomplete.
download_split train "$TRAIN_COUNT"
download_split val "$VAL_COUNT"

du -sh "$DATA_ROOT"
echo "==> AV2 motion subset download complete"
