#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MAIN="${MAIN:-main}"
OUTPUT_PDF="${OUTPUT_PDF:-Mariem_ElWedani_PFE.pdf}"
BUILD_DIR="$ROOT/build"
PDF_OUT="$ROOT/$OUTPUT_PDF"

clean_aux() {
  rm -rf "$BUILD_DIR"
  rm -f "$ROOT/${MAIN}-blx.bib"
  rm -f "$ROOT/tpl/"*.aux

  local f
  for f in aux log out toc lof lot bbl blg run.xml fls fdb_latexmk idx ind ilg maf; do
    rm -f "$ROOT/${MAIN}.${f}"
  done
  rm -f "$ROOT/${MAIN}.mtc" "$ROOT/${MAIN}.mtc"[0-9]
  rm -f "$ROOT/${MAIN}.synctex.gz"
}

usage() {
  echo "Usage: $0 [build|clean]" >&2
  echo "  build (default) — compile ${MAIN}.tex to ${OUTPUT_PDF} and remove auxiliary files" >&2
  echo "  clean           — remove auxiliary files and ${OUTPUT_PDF}" >&2
}

build() {
  if ! command -v latexmk >/dev/null 2>&1; then
    echo "latexmk not found. Install a TeX distribution (e.g. MacTeX)." >&2
    exit 1
  fi

  mkdir -p "$BUILD_DIR"

  latexmk -pdf -interaction=nonstopmode \
    -outdir="$BUILD_DIR" \
    -auxdir="$BUILD_DIR" \
    "${MAIN}.tex"

  cp "$BUILD_DIR/${MAIN}.pdf" "$PDF_OUT"
  clean_aux

  echo "Built: $PDF_OUT"
}

case "${1:-build}" in
  build)
    build
    ;;
  clean)
    clean_aux
    rm -f "$PDF_OUT"
    rm -f "$ROOT/main.pdf"
    echo "Cleaned auxiliary files and removed ${OUTPUT_PDF}"
    ;;
  -h | --help | help)
    usage
    ;;
  *)
    echo "Unknown command: $1" >&2
    usage
    exit 1
    ;;
esac
