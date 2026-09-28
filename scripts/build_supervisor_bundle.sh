#!/usr/bin/env bash
# Build encadrant delivery zip (report + notebooks + code + metrics + Colab logs).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 scripts/sanitize_notebooks_for_delivery.py

STUDENT="Mariem_ElWedani"
BUNDLE_NAME="${STUDENT}_PFE_livrable_encadrant"
DESKTOP="${HOME}/Desktop"
STAGING="$DESKTOP/$BUNDLE_NAME"
ZIP_PATH="$DESKTOP/${BUNDLE_NAME}.zip"

rm -rf "$STAGING"
mkdir -p "$STAGING"/{rapport,notebooks,code,resultats/agregats,journaux_colab}

if [[ ! -f "$ROOT/Mariem_ElWedani_PFE.pdf" ]]; then
  echo "Missing PDF. Run ./build.sh first." >&2
  exit 1
fi
cp "$ROOT/Mariem_ElWedani_PFE.pdf" "$STAGING/rapport/"

cp "$ROOT/notebooks/"*.ipynb "$STAGING/notebooks/"
cp "$ROOT/notebooks/README.md" "$STAGING/notebooks/README.md"

cp -R "$ROOT/src/hakemer" "$STAGING/code/"
cp "$ROOT/requirements-train.txt" "$STAGING/code/"
cp "$ROOT/train.sh" "$STAGING/code/"
for sh in run_eval_supplements.sh run_step0_cardinality.sh run_per_class_from_metrics.sh; do
  [[ -f "$ROOT/$sh" ]] && cp "$ROOT/$sh" "$STAGING/code/"
done

cp "$ROOT/reference/artifacts/"*_campaign.json "$STAGING/resultats/agregats/" 2>/dev/null || true
cp "$ROOT/reference/artifacts/"*.json "$STAGING/resultats/agregats/" 2>/dev/null || true
# Drop nested dirs if glob copied nothing useful
find "$ROOT/reference/artifacts" -maxdepth 1 -name '*campaign.json' -exec cp {} "$STAGING/resultats/agregats/" \;

while IFS= read -r -d '' nb; do
  rel="${nb#"$ROOT/reference/training_records/"}"
  mkdir -p "$STAGING/journaux_colab/$(dirname "$rel")"
  cp "$nb" "$STAGING/journaux_colab/$rel"
done < <(find "$ROOT/reference/training_records" -path '*/colab/*.ipynb' -print0)

find "$ROOT/reference/training_records" -name manifest.json -print0 | while IFS= read -r -d '' m; do
  rel="${m#"$ROOT/reference/training_records/"}"
  mkdir -p "$STAGING/journaux_colab/$(dirname "$rel")"
  cp "$m" "$STAGING/journaux_colab/$rel"
done

cat > "$STAGING/LISEZMOI.txt" <<'EOF'
Livrable PFE — Détection des émotions multiples (HAKE-MER / GoEmotions)
=======================================================================

Étudiante : Mariem El Wedani
Encadrant académique : Sahbi Bahroun
Année universitaire : 2025–2026

Contenu de l’archive
--------------------
1. rapport/          Rapport final (PDF).
2. notebooks/        Notebooks Google Colab pour reproduire les campagnes
                     (GPU, protocole GoEmotions : batch 16, LR 5e-5, 4 époques,
                     graines 42, 123, 456).
3. code/             Implémentation Python (module hakemer), requirements et
                     scripts d’entraînement / évaluation.
4. resultats/agregats/  Fichiers JSON de synthèse par campagne (métriques test
                     et validation agrégées sur trois graines).
5. journaux_colab/   Notebooks Colab exécutés et manifestes de traçabilité
                     (preuve des runs archivés).

Reproduction sous Colab
-----------------------
Ouvrir un notebook dans notebooks/, choisir un runtime GPU, exécuter les cellules
dans l’ordre. Le code est cloné depuis le dépôt Git du projet (variable d’environnement
GITHUB_REPO si besoin). Une copie du code source est incluse dans code/ pour lecture
ou exécution locale.

Terminologie
------------
Step 0 : baseline PLM flat (DistilBERT ou RoBERTa).
M1 : encodage hiérarchique phrase par phrase.
M2 : attention croisée par émotion et encodeur inter-émotions.
M3 / M4 : prior lexical (NRC ou SenticNet) et portes dynamiques (hypothèses H3, H4).

Contact
-------
Pour toute question sur le protocole ou les résultats, contacter l’étudiante via
les coordonnées ISI / SIIVA habituelles.
EOF

rm -f "$ZIP_PATH"
( cd "$DESKTOP" && zip -r -q "${BUNDLE_NAME}.zip" "$BUNDLE_NAME" )
echo "Created: $ZIP_PATH"
echo "Folder:  $STAGING"
du -sh "$ZIP_PATH" "$STAGING"
