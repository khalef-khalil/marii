# Training records (reproducibility)

Each experimental step that produces metrics in `reference/artifacts/` should have a matching record here.

## What to store

1. **Colab executed notebook** (preferred proof of the run): after a session, use **File → Download → .ipynb** and save under the step folder, e.g. `step0_plm/colab/YYYYMMDD_distilbert_executed.ipynb`. Commit outputs intact (git hash cell, training logs, `show_campaign` output, zip step).
2. **Artifact bundle**: campaign JSON + per-seed `metrics.json` + optional zip (see `reference/artifacts/`).
3. **`manifest.json`** in the step folder: links notebooks, artifacts, protocol, and aggregate test metrics.

Template notebooks live in `notebooks/` (no execution outputs). Records live here.

## Steps

| Step | Folder | Template notebook |
|------|--------|-------------------|
| Step 0 PLM baseline | `step0_plm/` | `notebooks/baseline_plm_campaign.ipynb` |
| M1 hierarchical | `step_m1/` | `notebooks/m1_campaign.ipynb` |

Future ablation steps (M2–M4) should add a folder and manifest the same way.
