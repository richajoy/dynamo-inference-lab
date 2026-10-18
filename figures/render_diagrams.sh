#!/usr/bin/env bash
# Render every figures/diagrams/*.mmd to a static SVG next to it.
# Static images render everywhere, including the GitHub mobile app, which does not draw Mermaid blocks.
set -euo pipefail
cd "$(dirname "$0")/diagrams"
MMDC=(npx -y @mermaid-js/mermaid-cli@12.0.0)
for src in *.mmd; do
  "${MMDC[@]}" -q -i "$src" -o "${src%.mmd}.svg" -c mermaid-config.json -b white
  echo "rendered ${src%.mmd}.svg"
done
