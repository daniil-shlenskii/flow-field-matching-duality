#!/bin/bash
set -euo pipefail

compute_fid() {
  local network_path="$1"
  local ref_path="$2"
  local outdir="$3"
  local max_images="$4"
  local nproc="$5"
  local gen_batch="$6"
  local gen_steps="$7"
  local gen_sampler="$8"
  local gen_pfgmpp="$9"
  local aug_dim="${10}"

  if [[ "$max_images" -lt 2 ]]; then
    echo "max_images must be >= 2 (got $max_images)" >&2
    return 2
  fi

  local last_seed=$((SEED_START + max_images - 1))
  local seeds="${SEED_START}-${last_seed}"

  if [[ -e "$outdir" ]]; then
    echo "Output already exists; choose a fresh outdir_base: $outdir" >&2
    return 2
  fi

  torchrun --standalone --nproc_per_node="$nproc" generate.py \
    --outdir="$outdir" --seeds="$seeds" --subdirs \
    --network="$network_path" --sampler="$gen_sampler" --steps="$gen_steps" --batch="$gen_batch" --pfgmpp="$gen_pfgmpp" --aug_dim="$aug_dim" >&2

  torchrun --standalone --nproc_per_node="$nproc" fid.py calc \
    --images="$outdir" --ref="$ref_path" --num="$max_images" \
  | awk '/^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$/ {v=$0} END{if (v == "") exit 1; print v}'
}

csv_escape() {
  local s="$1"
  s=${s//\"/\"\"}
  printf '"%s"' "$s"
}

extract_kimg() {
  local pkl="$1"
  local base
  base="$(basename "$pkl")"

  local kimg=""
  kimg="$(echo "$base" | sed -n 's/.*network-snapshot-\([0-9]\+\)\.pkl/\1/p')"

  if [[ -z "$kimg" ]]; then
    kimg="$(echo "$base" | grep -oE '[0-9]+' | tail -n 1 || true)"
  fi

  echo "$kimg"
}

usage() {
  echo "Usage: $0 <ckpt_dir> <ref_npz> [csv_path] [max_images] [min_kimg] [step_kimg] [outdir_base] [sampler] [pfgmpp] [aug_dim]"
  echo "Env overrides: NPROC (default 2), GEN_BATCH (default 512), GEN_STEPS (default 50), AUG_DIM (default 128), SEED_START (default 0)"
}

ckpt_dir="${1:-}"
ref="${2:-}"
csv_path="${3:-fid_results.csv}"
max_images="${4:-50000}"
min_kimg="${5:-0}"
step_kimg="${6:-0}"
outdir_base="${7:-${OUTDIR_BASE:-fid-tmp}}"
gen_sampler="${8:-${GEN_SAMPLER:-fm}}"
gen_pfgmpp="${9:-${GEN_PFGMPP:-0}}"
aug_dim="${10:-${AUG_DIM:-128}}"

if [[ -z "$ckpt_dir" || -z "$ref" ]]; then
  usage
  exit 2
fi

NPROC="${NPROC:-2}"
GEN_BATCH="${GEN_BATCH:-512}"
GEN_STEPS="${GEN_STEPS:-50}"
SEED_START="${SEED_START:-0}"
if [[ ! "$SEED_START" =~ ^[0-9]+$ ]]; then
  echo "SEED_START must be a nonnegative integer" >&2
  exit 2
fi

mkdir -p "$(dirname "$csv_path" 2>/dev/null || echo ".")"

if [[ ! -f "$csv_path" ]]; then
  echo "fid,checkpoint,kimg" > "$csv_path"
fi

pkls=()
while IFS= read -r pkl; do
  pkls+=("$pkl")
done < <(find "$ckpt_dir" -type f -name "network-snapshot-*.pkl" | sort)

if [[ "${#pkls[@]}" -eq 0 ]]; then
  echo "No .pkl files found under: $ckpt_dir" >&2
  exit 1
fi

last_processed_kimg=-1

for pkl in "${pkls[@]}"; do
  kimg="$(extract_kimg "$pkl")"
  
  if [[ -n "$kimg" && "$kimg" =~ ^[0-9]+$ && $((10#$kimg)) -lt $min_kimg ]]; then
    echo "SKIP: kimg=${kimg} < min_kimg=${min_kimg} pkl=$pkl"
    continue
  fi
  
  if [[ -n "$kimg" && "$kimg" =~ ^[0-9]+$ && $last_processed_kimg -ge 0 ]]; then
    local_kimg=$((10#$kimg))
    if (( local_kimg - last_processed_kimg < step_kimg )); then
      echo "SKIP: kimg=${kimg} too close to last (${last_processed_kimg}), step_kimg=${step_kimg} pkl=$pkl"
      continue
    fi
  fi
  
  outdir="${outdir_base}/kimg_${kimg:-unknown}"

  fid="$(compute_fid "$pkl" "$ref" "$outdir" "$max_images" "$NPROC" "$GEN_BATCH" "$GEN_STEPS" "$gen_sampler" "$gen_pfgmpp" "$aug_dim")"

  echo "${fid},$(csv_escape "$pkl"),$(csv_escape "$kimg")" >> "$csv_path"
  echo "OK: kimg=${kimg:-unknown} fid=$fid pkl=$pkl"
  
  if [[ -n "$kimg" && "$kimg" =~ ^[0-9]+$ ]]; then
    last_processed_kimg=$((10#$kimg))
  fi
done

echo "Wrote: $csv_path"
