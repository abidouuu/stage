#!/usr/bin/env bash
# Compresse les .pdf et .eps de plus de 50 MB (limite Overleaf) en réduisant
# progressivement la résolution des IMAGES RASTERISÉES éventuellement
# incluses dans le fichier. Le contenu vectoriel (courbes, texte, axes) n'est
# jamais dégradé : seule la résolution des rasters (dpi) baisse par paliers
# jusqu'à passer sous la limite.
#
# Usage : ./compress_for_overleaf.sh [dossier]   (par défaut : data_new)

set -euo pipefail

DIR="${1:-data_new}"
MAXSIZE=$((50 * 1000 * 1000))   # 50 MB décimaux (limite Overleaf)
DPI_START=300
DPI_MIN=72
DPI_STEP=50

need_gs() {
    command -v gs >/dev/null 2>&1 || { echo "ghostscript (gs) n'est pas installé." >&2; exit 1; }
}

compress_one() {
    local f="$1"
    local ext="${f##*.}"
    local device
    if [ "$ext" = "pdf" ]; then
        device="pdfwrite"
    else
        device="eps2write"
    fi

    local dpi="$DPI_START"
    local tmp
    tmp="$(mktemp --suffix=".$ext")"

    while [ "$dpi" -ge "$DPI_MIN" ]; do
        gs -sDEVICE="$device" -dCompatibilityLevel=1.5 \
           -dPDFSETTINGS=/prepress \
           -dDownsampleColorImages=true -dColorImageResolution="$dpi" \
           -dDownsampleGrayImages=true  -dGrayImageResolution="$dpi" \
           -dDownsampleMonoImages=true  -dMonoImageResolution="$dpi" \
           -dNOPAUSE -dBATCH -dQUIET \
           -sOutputFile="$tmp" "$f" >/dev/null 2>&1

        local size
        size=$(stat -c%s "$tmp")

        if [ "$size" -le "$MAXSIZE" ]; then
            mv "$tmp" "$f"
            printf "OK   %-60s -> %6d MB (dpi=%d)\n" "$f" "$((size/1000000))" "$dpi"
            return 0
        fi
        dpi=$((dpi - DPI_STEP))
    done

    # Dernier essai gardé même s'il dépasse encore la limite, pour info.
    mv "$tmp" "$f"
    local size
    size=$(stat -c%s "$f")
    printf "WARN %-60s -> %6d MB, encore au-dessus de 50MB malgre dpi=%d\n" "$f" "$((size/1000000))" "$DPI_MIN" >&2
    return 1
}

need_gs

echo "Recherche des .pdf et .eps > 50MB dans '$DIR'..."
mapfile -d '' -t files < <(find "$DIR" \( -name "*.pdf" -o -name "*.eps" \) -size +50M -print0)

if [ "${#files[@]}" -eq 0 ]; then
    echo "Aucun fichier au-dessus de 50MB."
    exit 0
fi

fail=0
for f in "${files[@]}"; do
    compress_one "$f" || fail=1
done

exit "$fail"
