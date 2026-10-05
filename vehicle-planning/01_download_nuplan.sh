#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
ensure_sysdeps

URLS_FILE="${1:-$HERE/urls.txt}"
KEEP_ZIPS="${KEEP_ZIPS:-0}"
ZIP_DIR="$NUPLAN_DATA_ROOT/_zips"
DONE_DIR="$NUPLAN_DATA_ROOT/_done"
mkdir -p "$ZIP_DIR" "$DONE_DIR" "$SPLITS_DIR"/{mini,trainval,test}

echo "[disk] $(df -h "$NUPLAN_DATA_ROOT" | tail -1)"

MAPS_URL="https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip"
if [ ! -f "$DONE_DIR/maps" ]; then
    wget -c -q --show-progress -O "$ZIP_DIR/nuplan-maps-v1.1.zip" "$MAPS_URL"
    rm -rf "$ZIP_DIR/_maps_tmp" && mkdir -p "$ZIP_DIR/_maps_tmp"
    unzip -q -o "$ZIP_DIR/nuplan-maps-v1.1.zip" -d "$ZIP_DIR/_maps_tmp"
    src="$(dirname "$(find "$ZIP_DIR/_maps_tmp" -name 'nuplan-maps-v1.0.json' | head -1)")"
    [ -n "$src" ] || { echo "[err] maps zip 안에서 nuplan-maps-v1.0.json 못 찾음"; exit 1; }
    rm -rf "$NUPLAN_MAPS_ROOT" && mv "$src" "$NUPLAN_MAPS_ROOT"
    rm -rf "$ZIP_DIR/_maps_tmp"
    [ "$KEEP_ZIPS" = "1" ] || rm -f "$ZIP_DIR/nuplan-maps-v1.1.zip"
    touch "$DONE_DIR/maps"
fi
echo "[maps] OK -> $NUPLAN_MAPS_ROOT"

if [ ! -f "$URLS_FILE" ]; then
    echo "[warn] $URLS_FILE 없음 -> maps만 받음. urls.txt.example 참고해서 만들어."
else
    grep -vE '^\s*(#|$)' "$URLS_FILE" | while read -r url; do
        fname="$(basename "${url%%\?*}")"
        lower="$(echo "$fname" | tr 'A-Z' 'a-z')"
        if echo "$lower" | grep -qE 'sensor|camera|lidar'; then
            echo "[skip] 센서 데이터: $fname"; continue
        fi
        if   echo "$lower" | grep -q mini; then split=mini
        elif echo "$lower" | grep -q test; then split=test
        else split=trainval; fi

        if [ -f "$DONE_DIR/$fname" ]; then echo "[done] $fname"; continue; fi
        echo "[get] $fname -> splits/$split"
        wget -c -q --show-progress -O "$ZIP_DIR/$fname" "$url"

        tmp="$ZIP_DIR/_tmp_${fname%.zip}"
        rm -rf "$tmp" && mkdir -p "$tmp"
        unzip -q -o "$ZIP_DIR/$fname" -d "$tmp"
        n=$(find "$tmp" -name '*.db' | wc -l)
        [ "$n" -gt 0 ] || { echo "[err] $fname 안에 .db가 없음"; exit 1; }
        find "$tmp" -name '*.db' -exec mv -f {} "$SPLITS_DIR/$split/" \;
        rm -rf "$tmp"
        [ "$KEEP_ZIPS" = "1" ] || rm -f "$ZIP_DIR/$fname"
        touch "$DONE_DIR/$fname"
        echo "[ok] $fname: $n db -> splits/$split"
    done
fi

ln -sfn "$SPLITS_DIR/trainval" "$NUPLAN_DATA_ROOT/nuplan-v1.1/trainval"
ln -sfn "$SPLITS_DIR/test"     "$NUPLAN_DATA_ROOT/nuplan-v1.1/test"

for s in mini trainval test; do
    echo "[db] $s: $(find "$SPLITS_DIR/$s" -maxdepth 1 -name '*.db' | wc -l) files"
done
echo "[disk] $(df -h "$NUPLAN_DATA_ROOT" | tail -1)"
