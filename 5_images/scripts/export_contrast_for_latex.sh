#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

png_path="${1:-$repo_root/5_images/images/Contrast.png}"
svg_path="${2:-$repo_root/5_images/svg/Contrast.svg}"
pdf_path="${3:-$repo_root/5_images/images/Contrast.pdf}"

if [[ ! -f "$png_path" ]]; then
  echo "Missing PNG source: $png_path" >&2
  exit 1
fi

if command -v inkscape >/dev/null 2>&1; then
  inkscape_bin="$(command -v inkscape)"
elif [[ -x /Applications/Inkscape.app/Contents/MacOS/inkscape ]]; then
  inkscape_bin="/Applications/Inkscape.app/Contents/MacOS/inkscape"
else
  echo "Inkscape is required to export the PDF." >&2
  echo "Install it or add it to PATH, then rerun this script." >&2
  exit 1
fi

dimensions="$(
  PNG_PATH="$png_path" python3 - <<'PY'
import os
from PIL import Image

with Image.open(os.environ["PNG_PATH"]) as image:
    print(f"{image.width} {image.height}")
PY
)"

width="${dimensions%% *}"
height="${dimensions##* }"

mkdir -p "$(dirname "$svg_path")" "$(dirname "$pdf_path")"

href="$(
  SVG_PATH="$svg_path" PNG_PATH="$png_path" python3 - <<'PY'
import os

svg_dir = os.path.dirname(os.path.abspath(os.environ["SVG_PATH"]))
png_path = os.path.abspath(os.environ["PNG_PATH"])
print(os.path.relpath(png_path, svg_dir))
PY
)"

cat > "$svg_path" <<SVG
<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" viewBox="0 0 $width $height">
  <image href="$href" x="0" y="0" width="$width" height="$height" preserveAspectRatio="none"/>
</svg>
SVG

"$inkscape_bin" \
  --export-type=pdf \
  --export-filename="$pdf_path" \
  "$svg_path"

echo "Wrote SVG: $svg_path"
echo "Wrote PDF: $pdf_path"
