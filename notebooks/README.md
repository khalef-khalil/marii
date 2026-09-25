# Colab — baseline PLM (step 0)

Your GitHub repo is **private**, so the one-click Colab link from GitHub often shows *Notebook not found* until Colab is allowed to read that repo.

## Easiest way to start

1. Go to [colab.research.google.com](https://colab.research.google.com).
2. **File → Upload notebook** → choose `baseline_plm_campaign.ipynb` from your local `marii` folder (same project you use with Cursor).
3. **Runtime → Change runtime type → GPU**.
4. Left sidebar **key icon** → add secret **`GITHUB_TOKEN`** = a GitHub [personal access token](https://github.com/settings/tokens) with access to `khalef-khalil/marii` (read is enough).
5. **Runtime → Run all**.

## Alternative: open from GitHub inside Colab

**File → Open notebook → GitHub** → sign in → authorize Colab → select `marii` → `notebooks/baseline_plm_campaign.ipynb`.  
Still set **`GITHUB_TOKEN`** for the clone step if the repo stays private.

## Optional: public repo

If you make the repo public, the badge link works and you may not need a token for `git clone`. Only do that if you are comfortable exposing the project.

```text
https://colab.research.google.com/github/khalef-khalil/marii/blob/main/notebooks/baseline_plm_campaign.ipynb
```

(Works after the repo is public and Colab can read GitHub.)
