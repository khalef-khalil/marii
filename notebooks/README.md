# Notebooks Colab — campagnes HAKE-MER (GoEmotions)

**PFE — Mariem El Wedani** · Encadrant académique : Sahbi Bahroun · 2025–2026

Ces notebooks reproduisent les expériences du chapitre 4 du rapport : ablations Step 0, M1 à M4, contrôles M2, extensions RoBERTa, analyse par classe et compléments d’évaluation.

## Prérequis

- Compte Google Colab, **runtime GPU** (T4 ou équivalent).
- Protocole commun : lot 16, taux d’apprentissage \(5\times10^{-5}\), **4 époques**, graines **42**, **123**, **456**, sélection au meilleur F1-macro validation.

## Ordre logique des campagnes

| Objectif | Notebook |
|----------|----------|
| Baseline flat (Step 0) | `baseline_plm_campaign.ipynb` |
| M1 seul (H1) | `m1_campaign.ipynb` |
| M1+M2 (H2) | `m1_m2_campaign.ipynb` |
| M3 NRC / SenticNet (H3) | `m1_m2_m3_nrc_campaign.ipynb`, `m1_m2_m3_senticnet_campaign.ipynb` |
| Prior lexical par étiquette | `m1_m2_m3_nrc_emotion_specific_campaign.ipynb`, `m1_m2_m3_senticnet_emotion_specific_campaign.ipynb` |
| M4 (H4) | `m1_m2_m3_m4_senticnet_campaign.ipynb`, `m1_m2_m3_nrc_m4_campaign.ipynb` |
| Contrôles M2 | `m1_m2_control_no_enc_campaign.ipynb`, `m1_m2_control_no_xattn_campaign.ipynb` |
| RoBERTa (même ladder) | `roberta_m1_m1_m2_campaign.ipynb`, `roberta_m3_m4_campaign.ipynb`, `roberta_m2_controls_campaign.ipynb`, `roberta_full_ladder_campaign.ipynb` |
| F1 par classe | `per_class_export_campaign.ipynb` |
| Analyse d’erreurs (+M1+M2) | `error_analysis_m1_m2_campaign.ipynb` |
| Cardinalité gold / ablation lexique | `step0_cardinality_campaign.ipynb`, `eval_supplements_campaign.ipynb` |

## Après une session Colab

1. Télécharger l’archive zip proposée en fin de notebook (métriques et checkpoints légers).
2. Télécharger le **notebook exécuté** (Fichier → Télécharger → `.ipynb`) pour conserver logs et hash Git.
3. Les agrégats JSON correspondants figurent aussi dans le livrable encadrant (`resultats/agregats/`).

## Dépôt Git

Les notebooks clônent le dépôt du projet au démarrage. En cas de dépôt privé, définir le secret Colab `GITHUB_REPO` (forme `utilisateur/nom-du-depot`) et éventuellement `GITHUB_TOKEN`.

Les journaux exécutés archivés pour ce PFE sont regroupés dans `reference/training_records/` du projet source (copie dans le zip encadrant : `journaux_colab/`).
