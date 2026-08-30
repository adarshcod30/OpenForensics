#!/usr/bin/env bash
# Render docs/diagrams/*.mmd to PNG.
#
# The diagrams are pre-rendered rather than left as ```mermaid fences because
# GitHub's Mermaid measures label text with a different font than it draws
# with, so every label loses its last character -- "NORMALISE" renders as
# "NORMALIS", "7x7x2048" as "7x7x204". mermaid-cli measures correctly.
#
# Sources stay in .mmd so the diagrams remain editable and diffable; run this
# after changing one.
set -euo pipefail
cd "$(dirname "$0")/.."
for f in docs/diagrams/*.mmd; do
  out="${f%.mmd}.png"
  npx --yes @mermaid-js/mermaid-cli@11 -i "$f" -o "$out" -b "#0d1117" -s 2 --quiet
  echo "  $out"
done
