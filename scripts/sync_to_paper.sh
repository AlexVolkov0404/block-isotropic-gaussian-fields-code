#!/usr/bin/env bash
# Copy the manuscript's figures from this repo's figs/ into the paper's figs/.
#
# Usage:
#   scripts/sync_to_paper.sh [--apply] [PAPER_DIR]
#
# Without --apply this is a dry run: it only reports which figures differ.
# The paper folder is taken from the PAPER_DIR argument, then the PAPER_DIR
# environment variable, then ../paper next to this repo.
#
# The figures are saved without a creation date (bigf.plotting.save_figure),
# so "differs" means that the plot itself changed. Run this from a clean,
# merged main so that the reported commit identifies the figures.
set -euo pipefail

# Figures included by the manuscript (cas-sc-template.tex).
FIGURES="
variance_envelope.pdf
probability_bounds_combined.pdf
radius_dimension.pdf
calibration_p95.pdf
"

apply=0
paper_arg=""
for arg in "$@"; do
  case "$arg" in
    --apply) apply=1 ;;
    -h|--help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "unknown option: $arg" >&2; exit 2 ;;
    *) paper_arg="$arg" ;;
  esac
done

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
paper_dir="${paper_arg:-${PAPER_DIR:-$repo_root/../paper}}"
if [ ! -d "$paper_dir/figs" ]; then
  echo "error: '$paper_dir/figs' not found. Pass the paper folder or set PAPER_DIR." >&2
  exit 1
fi
paper_dir="$(cd "$paper_dir" && pwd)"

commit="$(git -C "$repo_root" rev-parse --short HEAD)"
note=""
if [ -n "$(git -C "$repo_root" status --porcelain -- figs)" ]; then
  note=" + uncommitted changes in figs/"
fi

echo "code:  $repo_root (code@$commit$note)"
echo "paper: $paper_dir"
echo

changed=0
for f in $FIGURES; do
  src="$repo_root/figs/$f"
  dst="$paper_dir/figs/$f"
  if [ ! -f "$src" ]; then
    echo "  missing in code   $f"
    continue
  fi
  if [ -f "$dst" ] && cmp -s "$src" "$dst"; then
    echo "  same              $f"
    continue
  fi
  if [ -f "$dst" ]; then state="differs"; else state="new"; fi
  changed=$((changed + 1))
  if [ "$apply" -eq 1 ]; then
    cp "$src" "$dst"
    printf '  %-18s%s\n' "copied ($state)" "$f"
  else
    printf '  %-18s%s\n' "$state" "$f"
  fi
done

echo
if [ "$changed" -eq 0 ]; then
  echo "Nothing to sync."
elif [ "$apply" -eq 0 ]; then
  echo "Dry run: $changed file(s) would be copied. Re-run with --apply to copy."
else
  echo "Copied $changed file(s). Review and commit in the paper repo, e.g.:"
  echo "  git -C \"$paper_dir\" add figs"
  echo "  git -C \"$paper_dir\" commit -m \"figs: update from code@$commit\""
fi
