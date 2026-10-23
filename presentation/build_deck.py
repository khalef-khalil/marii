#!/usr/bin/env python3
"""Build the defense deck (native .pptx) for the HAKE-MER master thesis.

Numbers are read from archived campaign JSON and asserted against the report values.
Usage: python3 presentation/build_deck.py [output.pptx]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deck_lib import *  # noqa: F401,F403
from deck_lib import (GRAY, GREEN, INK, LGREEN, LIGHT, LINE, LNAVY, LORANGE, NAVY, ORANGE, WHITE, Arrow, Box, Cell,
                      P, Pic, Slide, Table, render_pptx, render_preview)

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "reference/artifacts"
FIG = Path(__file__).resolve().parent / "figs"
IMG = ROOT / "img"
NB = "\u00a0"  # no-break space
LILAC = (0xEC, 0xE6, 0xF6)
VIOLET = (0x6B, 0x4F, 0xA0)

LABEL = "Mariem El WEDANI  |  HAKE-MER  |  Soutenance de Master, ISI 2025 - 2026"

# ----------------------------------------------------------------------------- data


def fr(x, nd=3):
    return f"{x:.{nd}f}".replace(".", ",")


def stat(name):
    d = json.loads((ART / f"{name}_campaign.json").read_text(encoding="utf-8"))["test_aggregate"]["f1_macro"]
    return float(d["mean"]), float(d["std"])


def pm(name):
    m, s = stat(name)
    return f"{fr(m)}{NB}±{NB}{fr(s)}"


def mean(name):
    return stat(name)[0]


DB, RB = "distilbert_base_uncased", "roberta_base"
ROWS = {
    # key: (distil name, roberta name or None)
    "step0": (f"baseline_plm_{DB}", f"baseline_plm_{RB}"),
    "m1": (f"m1_{DB}", f"m1_{RB}"),
    "m12": (f"m1_m2_{DB}", f"m1_m2_{RB}"),
    "nrc": (f"m1_m2_m3_nrc_{DB}", f"m1_m2_m3_nrc_{RB}"),
    "sent": (f"m1_m2_m3_senticnet_{DB}", f"m1_m2_m3_senticnet_{RB}"),
    "m4s": (f"m1_m2_m3_senticnet_m4_{DB}", f"m1_m2_m3_senticnet_m4_{RB}"),
    "m4n": (f"m1_m2_m3_nrc_m4_{DB}", None),
    "noenc": (f"m1_m2_no_enc_{DB}", f"m1_m2_no_enc_{RB}"),
    "noxa": (f"m1_m2_no_xattn_{DB}", f"m1_m2_no_xattn_{RB}"),
}


def D(k):
    return pm(ROWS[k][0])


def R(k):
    return pm(ROWS[k][1]) if ROWS[k][1] else "non testé"


# integrity: values must match the report tables
EXPECT = {
    ("step0", 0): "0,486", ("step0", 1): "0,487", ("m1", 0): "0,457", ("m1", 1): "0,485",
    ("m12", 0): "0,505", ("m12", 1): "0,526", ("nrc", 0): "0,504", ("nrc", 1): "0,493",
    ("sent", 0): "0,504", ("sent", 1): "0,523", ("m4s", 0): "0,499", ("m4s", 1): "0,521",
    ("m4n", 0): "0,510", ("noenc", 0): "0,476", ("noenc", 1): "0,507", ("noxa", 0): "0,504", ("noxa", 1): "0,524",
}
for (k, i), v in EXPECT.items():
    assert fr(mean(ROWS[k][i])) == v, (k, i, fr(mean(ROWS[k][i])), v)

gain_db = mean(ROWS["m12"][0]) - mean(ROWS["step0"][0])
gain_rb = mean(ROWS["m12"][1]) - mean(ROWS["step0"][1])
gap_bb = mean(ROWS["m12"][1]) - mean(ROWS["m12"][0])
loss_enc_db = mean(ROWS["m12"][0]) - mean(ROWS["noenc"][0])
loss_enc_rb = mean(ROWS["m12"][1]) - mean(ROWS["noenc"][1])
assert 0.018 < gain_db < 0.021 and 0.037 < gain_rb < 0.041 and 0.019 < gap_bb < 0.023
assert 0.028 < loss_enc_db < 0.031 and 0.018 < loss_enc_rb < 0.021

TEST_CARD = (4590, 774, 61, 2)
assert sum(TEST_CARD) == 5427
single_share = TEST_CARD[0] / 5427 * 100  # 84.6

# ----------------------------------------------------------------------------- helpers


def card(x, y, w, h, title, body, accent=GREEN, fill=LIGHT, size=15, title_size=18):
    paras = [P(title, title_size, NAVY, True, after=6)]
    for b in body:
        paras.append(P(b, size, INK, bullet=True, after=5))
    return [Box(x, y, 0.07, h, fill=accent), Box(x + 0.07, y, w - 0.07, h, paras, fill=fill, margins=(0.18, 0.12, 0.15, 0.1))]


def fpic(path, x, y, w, caption):
    from PIL import Image
    with Image.open(path) as im:
        h = w * im.height / im.width
    out = [Pic(path, x, y, w=w), Box(x, y + h + 0.05, w, 0.4, [P(caption, 12, GRAY, italic=True, after=0)], margins=(0, 0, 0, 0))]
    if y + h + 0.45 > 6.95:
        print(f"WARN fig {Path(path).name} bottom {y + h + 0.45:.2f}")
    return out


slides: list[Slide] = []


def new(title=None, notes="", footer=True):
    s = Slide(title, notes, footer)
    slides.append(s)
    return s


# ----------------------------------------------------------------------------- 1. Title
s = new(footer=False, notes=(
    "[0:30] Bonjour à toutes et à tous. Je m'appelle Mariem El Wedani. Je vous présente mon mémoire de Master de "
    "recherche, encadré par M. Sahbi Bahroun : la détection d'émotions multiples dans les textes, par fusion "
    "d'attentions hiérarchiques et de connaissances sémantiques externes. Mon architecture s'appelle HAKE-MER. "
    "Je vais d'abord poser le problème, puis présenter l'architecture, le protocole expérimental, les résultats, "
    "et enfin les limites et les perspectives."))
s.add(Pic(IMG / "logo-isi.png", 0.6, 0.35, h=0.95), Pic(IMG / "Logo_UTM.png", 11.75, 0.35, h=0.95))
s.add(Box(2.9, 0.3, 7.5, 1.1, [
    P("République Tunisienne", 12, GRAY, True, align="c", after=1),
    P("Ministère de l'Enseignement Supérieur et de la Recherche Scientifique", 11, GRAY, align="c", after=1),
    P("Université de Tunis El Manar  |  Institut Supérieur d'Informatique", 11, GRAY, align="c", after=0)],
    anchor="m", margins=(0, 0, 0, 0)))
s.add(Box(0.6, 1.85, 12.1, 0.4, [P("MÉMOIRE DE MASTER DE RECHERCHE EN INFORMATIQUE", 15, GREEN, True, align="c", after=0)],
          anchor="m", margins=(0, 0, 0, 0)))
s.add(Box(0.6, 2.4, 12.1, 2.35, [
    P("Détection des Émotions Multiples dans les Textes par Fusion d'Attentions Hiérarchiques et Connaissances "
      "Sémantiques Externes", 30, WHITE, True, align="c", after=0)],
    fill=NAVY, anchor="m", margins=(0.5, 0.2, 0.5, 0.2)))
s.add(Box(0.6, 5.05, 12.1, 1.5, [
    P("Présenté par : **Mariem El WEDANI**", 20, INK, align="c", after=6),
    P("Encadrant académique : **Sahbi BAHROUN**, Maître de Conférences", 18, INK, align="c", after=6),
    P("Année universitaire 2025 - 2026", 16, GRAY, align="c", after=0)], anchor="m", margins=(0, 0, 0, 0)))

# ----------------------------------------------------------------------------- 2. Context and data
s = new("Contexte : un texte, plusieurs émotions", notes=(
    "[1:15] Le point de départ est simple : un même commentaire peut exprimer plusieurs émotions à la fois. "
    "La tâche est donc multi-étiquettes. Le jeu de données de référence est GoEmotions : des commentaires Reddit "
    "annotés avec 27 émotions plus neutral, soit 28 sorties. Le découpage officiel donne 43 410 exemples "
    "d'entraînement, 5 426 en validation et 5 427 en test. Le graphique montre le nombre d'étiquettes gold par "
    "commentaire sur le test : environ 85 % des commentaires n'ont qu'une seule étiquette, donc les cas "
    "réellement multi-émotions sont minoritaires. Le deuxième point, c'est le déséquilibre : admiration est environ "
    "trente fois plus fréquente que grief ; sur le test, grief n'a que 6 exemples contre 1 787 pour neutral."))
s.add(Box(0.6, 1.55, 6.3, 5.2, [
    P("**Tâche** : prédire l'ensemble des émotions exprimées dans un commentaire (multi-étiquettes)", 18, bullet=True, after=12),
    P("**GoEmotions** : commentaires Reddit annotés avec 27 émotions et neutral, soit K = 28", 18, bullet=True, after=12),
    P("**Découpage officiel** : 43" + NB + "410 entraînement, 5" + NB + "426 validation, 5" + NB + "427 test", 18, bullet=True, after=12),
    P("**Déséquilibre marqué** : sur le test, neutral compte 1" + NB + "787 exemples positifs, grief seulement 6", 18, bullet=True, after=12),
    P(f"**Peu de cas multi-émotions** : {fr(single_share, 1)}{NB}% des commentaires de test portent une seule étiquette", 18, bullet=True, after=0)],
    margins=(0, 0, 0, 0)))
s.add(*fpic(FIG / "goemotions_cardinality.png", 7.2, 1.7, 5.5, "Nombre d'étiquettes gold par commentaire (test, n = 5" + NB + "427)"))

# ----------------------------------------------------------------------------- 3. Problem
s = new("Problématique : trois limites des approches standards", notes=(
    "[1:00] Les classifieurs standards, un encodeur pré-entraîné avec une sortie sigmoïde par émotion, "
    "présentent trois limites. L1 : les étiquettes sont traitées comme indépendantes, alors que les émotions "
    "se co-occurrent ou s'excluent. L2 : tout le commentaire est résumé dans un seul vecteur, ce qui mélange les "
    "signaux portés par des phrases différentes. L3 : les émotions rares reçoivent très peu de signal supervisé. "
    "Ces trois limites mènent à la question de recherche, en bas : une architecture avec encodage hiérarchique, "
    "attention entre émotions et lexiques externes peut-elle mieux détecter les émotions multiples et rares "
    "qu'un modèle pré-entraîné affiné de façon standard ?"))
cw = 3.9
s.add(*card(0.6, 1.6, cw, 3.0, "L1 : étiquettes indépendantes",
            ["Une sigmoïde par émotion, sans modèle des co-occurrences", "Joie et tristesse sont rarement actives ensemble"], size=17))
s.add(*card(0.6 + cw + 0.2, 1.6, cw, 3.0, "L2 : structure du texte perdue",
            ["Un seul vecteur [CLS] pour tout le commentaire", "Les signaux de phrases différentes se mélangent"], size=17))
s.add(*card(0.6 + 2 * (cw + 0.2), 1.6, cw, 3.0, "L3 : émotions rares",
            ["Longue traîne : grief 6 exemples en test, neutral 1" + NB + "787", "Peu de signal supervisé pour les classes rares"], size=17))
s.add(Box(0.6, 4.95, 12.1, 1.75, [
    P("Question de recherche", 14, (0xB4, 0xE0, 0xC8), True, after=4),
    P("Une architecture avec encodage hiérarchique, attention entre émotions et lexiques externes peut-elle "
      "mieux détecter les émotions multiples et rares qu'un modèle pré-entraîné affiné de façon standard ?",
      20, WHITE, True, after=0)], fill=NAVY, anchor="m", margins=(0.35, 0.1, 0.35, 0.1)))

# ----------------------------------------------------------------------------- 4. Objectives and hypotheses
s = new("Objectifs et hypothèses", notes=(
    "[1:00] Quatre objectifs : concevoir une architecture modulaire, HAKE-MER, avec quatre composantes cumulables ; "
    "l'évaluer par ablation cumulative à budget d'entraînement constant, sur deux encodeurs ; comparer deux "
    "ressources externes, NRC et SenticNet ; et analyser les résultats au niveau de la classe, pas seulement en "
    "macro. À droite, les quatre hypothèses, une par module : H1, M1 améliore le flat ; H2, M2 améliore M1 ; "
    "H3, un prior lexical améliore M1+M2 ; H4, la pondération dynamique améliore le modèle précédent. "
    "Chacune est testable par une comparaison directe entre deux étapes de l'ablation."))
s.add(Box(0.6, 1.55, 4.9, 5.2, [
    P("Objectifs", 20, NAVY, True, after=10),
    P("**Concevoir** HAKE-MER, une architecture modulaire à quatre composantes cumulables", 18, bullet=True, after=12),
    P("**Évaluer** par ablation cumulative à budget d'affinage constant, sur deux encodeurs", 18, bullet=True, after=12),
    P("**Comparer** deux ressources externes : lexique NRC et SenticNet", 18, bullet=True, after=12),
    P("**Analyser** par classe, au-delà du F1-macro, et cibler les confusions", 18, bullet=True, after=0)],
    margins=(0, 0, 0, 0)))
s.add(Table(5.85, 1.6, [0.7, 1.75, 4.4], [
    ["Hyp.", "Comparaison", "Énoncé testé"],
    [Cell("H1", True), "Flat vs +M1", "M1 améliore le F1-macro par rapport au flat à 128 tokens"],
    [Cell("H2", True), "+M1 vs +M1+M2", "M2 améliore le F1-macro par rapport à M1 seul"],
    [Cell("H3", True), "+M1+M2 vs +M3", "Un prior lexical (NRC ou SenticNet) améliore M1+M2"],
    [Cell("H4", True), "+M3 vs +M4", "La pondération dynamique améliore le F1-macro"],
], size=15, min_row=0.85))

# ----------------------------------------------------------------------------- 5. State of the art
s = new("Positionnement dans la littérature", notes=(
    "[1:00] Trois lignes de travail se rejoignent. Un : la classification multi-étiquettes avec modèles "
    "pré-entraînés. GoEmotions publie un F1-macro d'environ 0,46 avec BERT-base ; Seq2Emo, environ 0,52, "
    "modélise explicitement les dépendances entre étiquettes. Deux : l'attention hiérarchique, avec le HAN de "
    "Yang et collaborateurs, qui sépare mots et phrases. Trois : les connaissances externes, comme le lexique "
    "d'associations mot-émotion NRC et la ressource conceptuelle SenticNet. HAKE-MER combine ces trois pistes "
    "dans une architecture où chaque brique peut être activée ou désactivée, ce qui permet de mesurer sa "
    "contribution isolément."))
cw = 3.9
s.add(*card(0.6, 1.6, cw, 3.55, "Multi-étiquettes avec PLM",
            ["GoEmotions : BERT-base, F1-macro ≈ 0,46 (Demszky et al., 2020)",
             "Seq2Emo : dépendances entre étiquettes, ≈ 0,52 (Ke et al., 2021)"], accent=ORANGE, size=17))
s.add(*card(0.6 + cw + 0.2, 1.6, cw, 3.55, "Attention hiérarchique",
            ["HAN : attention aux mots puis aux phrases (Yang et al., 2016)", "Répond à la perte de structure (L2)"], accent=GREEN, size=17))
s.add(*card(0.6 + 2 * (cw + 0.2), 1.6, cw, 3.55, "Connaissances externes",
            ["NRC : associations mot-émotion (Mohammad et Turney, 2013)",
             "SenticNet : ressource conceptuelle (Cambria et al., 2018)"], accent=(0x5B, 0x7B, 0xA6), size=17))
s.add(Box(0.6, 5.5, 12.1, 1.2, [
    P("**HAKE-MER** combine ces trois pistes dans une architecture modulaire : chaque module s'active par un "
      "drapeau, ce qui isole sa contribution par ablation.", 18, NAVY, after=0)],
    fill=LNAVY, anchor="m", margins=(0.3, 0.1, 0.3, 0.1)))

# ----------------------------------------------------------------------------- 6. Architecture
s = new("Architecture HAKE-MER : quatre modules activables", notes=(
    "[1:45] Voici l'architecture. Le texte est découpé en au plus quatre phrases de 32 tokens, encodées par le "
    "backbone partagé, DistilBERT ou RoBERTa. M1 agrège les mots en un vecteur par phrase : c'est l'encodage "
    "hiérarchique, qui répond à L2. M2 introduit une requête par émotion qui interroge les phrases par attention "
    "croisée, puis une auto-attention entre les 28 émotions : c'est elle qui répond à L1, la dépendance entre "
    "étiquettes. M4 pondère chaque émotion selon le contexte. En bas, la branche lexicale M3 : un vecteur "
    "document est construit à partir de NRC ou SenticNet, projeté, puis ajouté aux requêtes d'émotions via une "
    "porte g initialisée à zéro ; quand M3 est désactivé, g vaut zéro et rien ne change. Chaque module s'active "
    "par un drapeau : c'est ce qui rend l'ablation propre."))
BW, BH, BY = 1.47, 1.35, 2.0
tops = [
    ("Texte d'entrée", "4 phrases × 32 tokens", LIGHT, LINE, None),
    ("Backbone partagé", "DistilBERT ou RoBERTa", LIGHT, LINE, None),
    ("M1", "Encodage hiérarchique mot → phrase", LORANGE, ORANGE, None),
    ("M2", "Attention croisée émotion → phrases", LNAVY, (0x5B, 0x7B, 0xA6), None),
    ("M2", "Auto-attention inter-émotions", LNAVY, (0x5B, 0x7B, 0xA6), None),
    ("M4", "Pondération dynamique", LILAC, VIOLET, None),
    ("Sortie", "28 sigmoïdes", LIGHT, LINE, None),
]
s.add(Box(0.6, 1.5, 4.0, 0.35, [P("Tronc principal", 13, GRAY, True, italic=True, after=0)], margins=(0, 0, 0, 0)))
for i, (t, sub, f, ln, _) in enumerate(tops):
    x = 0.6 + i * 1.77
    s.add(Box(x, BY, BW, BH, [P(t, 15, NAVY, True, align="c", after=3), P(sub, 12, INK, align="c", after=0)],
              fill=f, line=ln, line_w=1.5, shape="round", anchor="m", margins=(0.06, 0.05, 0.06, 0.05)))
    if i < len(tops) - 1:
        s.add(Arrow(x + BW + 0.03, BY + BH / 2 - 0.12, 0.24, 0.24))
s.add(Box(0.6, 3.95, 5.0, 0.35, [P("Branche lexicale (M3)", 13, GRAY, True, italic=True, after=0)], margins=(0, 0, 0, 0)))
LY, LH = 4.4, 1.05
s.add(Box(0.6, LY, 1.75, LH, [P("Lexique affectif", 14, NAVY, True, align="c", after=2), P("NRC ou SenticNet", 12, INK, align="c", after=0)],
          fill=LGREEN, line=GREEN, line_w=1.5, shape="round", anchor="m"))
s.add(Arrow(2.4, LY + LH / 2 - 0.12, 0.3, 0.24))
s.add(Box(2.75, LY, 2.4, LH, [P("M3 : projection + tanh", 14, NAVY, True, align="c", after=2), P("porte g = 0 au départ", 12, INK, align="c", after=0)],
          fill=LGREEN, line=GREEN, line_w=1.5, shape="round", anchor="m"))
s.add(Arrow(5.2, LY + LH / 2 - 0.12, 0.3, 0.24))
s.add(Box(5.55, LY, 2.4, LH, [P("Requêtes d'émotions", 14, NAVY, True, align="c", after=2), P("Qk + g · prior", 12, INK, align="c", after=0)],
          fill=LNAVY, line=(0x5B, 0x7B, 0xA6), line_w=1.5, shape="round", anchor="m"))
# arrow up from queries to M2 (cross-attention box center x = 0.6 + 3*1.77 + 0.735 = 6.645)
s.add(Arrow(6.525, BY + BH + 0.05, 0.24, LY - (BY + BH) - 0.1, direction="up"))
s.add(Box(8.4, LY, 4.3, LH, [P("Quand M3 est désactivé, g = 0 : les requêtes restent inchangées et la pile se réduit à M1+M2.", 13, INK, after=0)],
          anchor="m", margins=(0.05, 0.05, 0.05, 0.05)))
s.add(Box(0.6, 5.8, 12.1, 0.95, [
    P("**M1** garde la structure par phrases (L2)  |  **M2** modélise les interactions entre émotions (L1)  |  "
      "**M3** injecte un lexique (L3)  |  **M4** pondère chaque émotion selon le contexte", 15, NAVY, after=0)],
    fill=LIGHT, anchor="m", margins=(0.3, 0.1, 0.3, 0.1)))

# ----------------------------------------------------------------------------- 7. Protocol
s = new("Protocole expérimental : ablation cumulative à budget constant", notes=(
    "[1:00] Le protocole est le même pour toutes les configurations. On part de Step 0, le modèle plat avec un "
    "seul vecteur sur 128 tokens, puis on ajoute les modules un à un : M1, puis M2, puis M3, puis M4. Le budget "
    "d'affinage est identique : quatre époques, taux d'apprentissage 5 fois 10 puissance moins 5, lot de 16, "
    "AdamW, warmup de 10 %. On sélectionne le meilleur checkpoint sur le F1-macro de validation, puis on "
    "évalue sur le test. Chaque configuration est lancée avec trois graines, 42, 123 et 456, et on rapporte la "
    "moyenne et l'écart-type. Les deux encodeurs, DistilBERT et RoBERTa, sont traités de la même façon. "
    "Le F1-macro est la métrique principale car il pondère toutes les émotions de la même façon, y compris les rares."))
s.add(Box(0.6, 1.55, 6.3, 5.2, [
    P("**Corpus** : GoEmotions, K = 28, découpage officiel", 19, bullet=True, after=14),
    P("**Encodeurs** : DistilBERT-base et RoBERTa-base", 19, bullet=True, after=14),
    P("**Ablation** : Step 0, +M1, +M1+M2, +M3, +M4, à budget d'affinage identique", 19, bullet=True, after=14),
    P("**Métrique principale** : F1-macro test, checkpoint choisi sur le F1-macro validation", 19, bullet=True, after=14),
    P("**Répétitions** : trois graines (42, 123, 456), moyenne ± écart-type", 19, bullet=True, after=0)],
    margins=(0, 0, 0, 0)))
s.add(Table(7.2, 1.6, [2.6, 2.9], [
    ["Paramètre commun", "Valeur"],
    ["Époques", "4"],
    ["Taille de lot", "16"],
    ["Taux d'apprentissage", "0,00005"],
    ["Optimiseur", "AdamW, décroissance 0,01"],
    ["Warmup", "10 % des pas"],
    ["Seuil de décision", "0,5"],
    ["Texte (Step 0)", "128 tokens"],
    ["Texte (M1 et plus)", "4 phrases × 32 tokens"],
], size=14, min_row=0.5))

# ----------------------------------------------------------------------------- 8. Results DistilBERT
s = new("Résultat 1 : M1+M2 est la seule étape qui progresse", notes=(
    "[1:30] Voici l'ablation sur DistilBERT, F1-macro test, moyenne et écart-type sur trois graines. Step 0, "
    "le modèle plat, fait 0,486. En ajoutant M1 seul, on descend à 0,457 : l'encodage hiérarchique sans "
    "interaction entre émotions ne bat pas le flat, donc H1 n'est pas soutenue. En ajoutant M2, on passe à "
    f"0,505, soit un gain de {fr(gain_db)} sur Step 0 : H2 est soutenue. Ensuite, ni le lexique NRC, ni "
    "SenticNet, ni la pondération M4 ne dépassent M1+M2 : 0,504, 0,504 et 0,499, dans la dispersion entre "
    "graines. L'axe est tronqué à 0,42 pour que les écarts soient lisibles ; ce détail est indiqué sur le graphique."))
s.add(*fpic(FIG / "ablation_distilbert.png", 0.6, 1.5, 7.7, "F1-macro test, DistilBERT-base, moyenne ± écart-type sur 3 graines. Axe vertical tronqué à 0,42."))
s.add(*card(8.6, 1.6, 4.1, 1.55, "M1 seul recule", [f"{fr(mean(ROWS['m1'][0]))} contre {fr(mean(ROWS['step0'][0]))} pour le flat"], accent=ORANGE, size=17))
s.add(*card(8.6, 3.3, 4.1, 1.55, "M1+M2 progresse", [f"{fr(mean(ROWS['m12'][0]))}, soit +{fr(gain_db)} sur Step 0"], accent=GREEN, size=17))
s.add(*card(8.6, 5.0, 4.1, 1.75, "M3 et M4 sans gain", ["0,499 à 0,504, aucun ne dépasse M1+M2"], accent=(0x5B, 0x7B, 0xA6), size=17))

# ----------------------------------------------------------------------------- 9. Results RoBERTa
s = new("Résultat 2 : le même profil sur RoBERTa-base", notes=(
    "[1:00] On refait la même échelle sur RoBERTa, un encodeur plus fort, pour vérifier que le résultat ne "
    "dépend pas du backbone. Le profil est le même : M1 seul n'apporte rien, 0,485 contre 0,487 pour le flat ; "
    f"M1+M2 atteint 0,526, soit +{fr(gain_rb)} sur le flat. Et l'écart entre les deux encodeurs sur M1+M2 est "
    f"d'environ {fr(gap_bb, 3)}, soit deux points. La conclusion tient donc sur les deux backbones : M2 aide, "
    "M1 seul non."))
s.add(*fpic(FIG / "backbone.png", 0.6, 1.55, 6.6, "F1-macro test, moyenne ± écart-type sur 3 graines."))
s.add(Table(7.5, 1.6, [2.0, 1.6, 1.6], [
    ["Configuration", "DistilBERT", "RoBERTa"],
    ["Step 0 (flat)", fr(mean(ROWS["step0"][0])), fr(mean(ROWS["step0"][1]))],
    ["+M1", fr(mean(ROWS["m1"][0])), fr(mean(ROWS["m1"][1]))],
    [Cell("+M1+M2", True, LGREEN), Cell(fr(mean(ROWS["m12"][0])), True, LGREEN), Cell(fr(mean(ROWS["m12"][1])), True, LGREEN)],
], size=15, min_row=0.55))
s.add(Box(7.5, 4.0, 5.2, 2.7, [
    P(f"RoBERTa dépasse DistilBERT d'environ {fr(gap_bb, 3)} sur M1+M2", 18, bullet=True, after=12),
    P("M1 seul n'améliore pas le flat sur l'un ni l'autre des encodeurs", 18, bullet=True, after=12),
    P("Le gain de M2 est observé sur les deux backbones", 16, bullet=True, after=0)], margins=(0, 0, 0, 0)))

# ----------------------------------------------------------------------------- 10. M2 controls
s = new("Pourquoi M2 aide : contrôles mécanistiques", notes=(
    "[1:00] Pour comprendre d'où vient le gain de M2, on a lancé deux contrôles. Premier contrôle : on retire "
    f"l'encodeur inter-émotions. Le F1-macro tombe à 0,476 sur DistilBERT, une perte de {fr(loss_enc_db)}, et à "
    f"0,507 sur RoBERTa, une perte de {fr(loss_enc_rb)}. Deuxième contrôle : on retire l'attention croisée "
    "phrase-émotion. Le score ne bouge pas de façon nette : 0,504 contre 0,505 sur DistilBERT, 0,524 contre 0,526 "
    "sur RoBERTa, et l'écart-type de 0,020 sur RoBERTa est plus grand que la différence. Donc c'est l'interaction "
    "entre émotions, la limite L1, qui porte l'essentiel du gain, davantage que l'attention aux phrases."))
s.add(Table(0.6, 1.65, [4.6, 3.3, 3.3], [
    ["Configuration", "DistilBERT", "RoBERTa"],
    ["M1 seul", D("m1"), R("m1")],
    [Cell("M1+M2 complet", True, LGREEN), Cell(D("m12"), True, LGREEN), Cell(R("m12"), True, LGREEN)],
    ["M1+M2 sans encodeur inter-émotions", Cell(D("noenc"), fill=LORANGE), Cell(R("noenc"), fill=LORANGE)],
    ["M1+M2 sans attention croisée aux phrases", D("noxa"), R("noxa")],
], size=16, min_row=0.65))
s.add(Box(0.6, 5.0, 11.2, 1.75, [
    P(f"Retirer l'encodeur inter-émotions coûte {fr(loss_enc_db)} (DistilBERT) et {fr(loss_enc_rb)} (RoBERTa)", 17, bullet=True, after=8),
    P("Retirer l'attention croisée ne change pas nettement la macro : l'écart reste dans l'écart-type", 17, bullet=True, after=8),
    P("Le gain de M2 vient surtout de l'interaction entre émotions (limite L1)", 17, bullet=True, after=0)],
    margins=(0, 0, 0, 0)))

# ----------------------------------------------------------------------------- 11. Per class
s = new("Au-delà de la macro : où M2 aide, où il nuit", notes=(
    "[1:15] Le F1-macro cache des comportements très différents selon les émotions. Sur ce graphique, Step 0 en "
    "gris, M1+M2 en vert, pour douze émotions triées par fréquence croissante. M2 aide surtout les émotions de "
    "fréquence intermédiaire : desire, annoyance, curiosity. Sur grief, le F1 passe de zéro à 0,26, et relief "
    "de 0,16 à 0,40, mais avec 6 et 11 exemples de test, ces chiffres ne sont pas statistiquement fiables. "
    "M2 nuit à nervousness et embarrassment, avec environ 0,09 de perte chacun. Un indicateur de synthèse : "
    "l'écart de F1 moyen entre émotions fréquentes et très rares passe de 0,365 pour le flat à 0,296 avec M1+M2, "
    "alors que M1 seul l'élargit à 0,405. Les gains de H2 ne sont donc pas un rappel uniforme sur la longue traîne."))
s.add(*fpic(FIG / "per_class.png", 0.6, 1.5, 7.0, "F1 test par émotion, DistilBERT, moyenne sur 3 graines, labels par effectif croissant."))
s.add(*card(8.6, 1.6, 4.1, 1.9, "Gains",
            ["Fréquence intermédiaire : desire, annoyance, curiosity", "grief et relief progressent, avec 6 et 11 exemples de test : fragile"], accent=GREEN, size=14))
s.add(*card(8.6, 3.65, 4.1, 1.4, "Pertes", ["nervousness et embarrassment reculent d'environ 0,09"], accent=ORANGE, size=14))
s.add(*card(8.6, 5.2, 4.1, 1.55, "Écart fréquent vs rare",
            ["0,365 (flat) → 0,296 (M1+M2)", "M1 seul l'élargit : 0,405"], accent=(0x5B, 0x7B, 0xA6), size=14))
s.add(Box(0.6, 6.1, 7.4, 0.75, [P("Les gains de M2 viennent des émotions de fréquence intermédiaire, pas d'un rappel uniforme sur la longue traîne.", 14, NAVY, True, after=0)],
          fill=LNAVY, anchor="m", margins=(0.25, 0.08, 0.25, 0.08)))

# ----------------------------------------------------------------------------- 12. Errors
s = new("Analyse des erreurs : neutral est le pivot des confusions", notes=(
    "[0:45] Pour finir l'analyse, on a compté les erreurs sur les checkpoints M1+M2, trois graines cumulées. "
    "Les dix motifs les plus fréquents impliquent tous neutral : approval manqué et prédit neutral, neutral "
    "prédit à tort alors que le gold est approval, annoyance manqué, et ainsi de suite. Les confusions dominantes "
    "ne sont donc pas entre émotions voisines comme anger et annoyance, mais entre une émotion faible, comme "
    "approval, disapproval ou annoyance, et l'absence d'émotion. C'est une limite de fond de GoEmotions, et une "
    "piste d'amélioration : mieux séparer neutral des émotions de faible intensité."))
s.add(*fpic(FIG / "errors.png", 0.6, 1.6, 8.1, "Dix motifs d'erreur les plus fréquents, M1+M2, DistilBERT, 3 graines cumulées (test)."))
s.add(*card(9.0, 1.6, 3.7, 2.4, "Constat", ["Les dix motifs impliquent tous neutral", "Approval, annoyance, disapproval vs neutral dominent"], accent=ORANGE, size=15))
s.add(*card(9.0, 4.2, 3.7, 2.4, "Lecture", ["Frontière floue entre émotion faible et absence d'émotion", "Piste : mieux modéliser neutral"], accent=GREEN, size=15))

# ----------------------------------------------------------------------------- 13. Hypotheses verdict
s = new("Bilan des hypothèses : une seule clairement soutenue", notes=(
    "[1:15] Voici le bilan. H1, M1 améliore le flat : non soutenue, sur les deux encodeurs. H2, M2 améliore M1 : "
    "soutenue, avec +0,048 sur DistilBERT et +0,041 sur RoBERTa. H3, un prior lexical : on reste prudent, car le "
    "prior document est neutre en macro, et sur RoBERTa NRC est même instable, avec un écart-type de 0,045. H4, "
    "la pondération dynamique : SenticNet recule ; NRC avec M4 donne 0,510 contre 0,504 sur DistilBERT, un gain "
    "faible, non testé sur RoBERTa ; on ne retient pas M4. La configuration retenue, sur validation, est M1+M2 "
    "pour les deux encodeurs."))
assert abs((mean(ROWS["m12"][0]) - mean(ROWS["m1"][0])) - 0.048) < 0.0015 and abs((mean(ROWS["m12"][1]) - mean(ROWS["m1"][1])) - 0.041) < 0.0015
s.add(Table(0.6, 1.6, [0.7, 2.7, 3.3, 3.1, 2.3], [
    ["Hyp.", "Comparaison", "DistilBERT", "RoBERTa", "Verdict"],
    [Cell("H1", True), "Step 0 vs +M1", f"{D('step0')} vs {D('m1')}", f"{R('step0')} vs {R('m1')}", Cell("Non soutenue", True, fill=LORANGE)],
    [Cell("H2", True), "+M1 vs +M1+M2", f"{D('m1')} vs {D('m12')}", f"{R('m1')} vs {R('m12')}", Cell("Soutenue", True, fill=LGREEN)],
    [Cell("H3", True), "+M1+M2 vs +M3 (NRC ; SenticNet)", f"{D('m12')} vs {D('nrc')} ; {D('sent')}", f"{R('m12')} vs {R('nrc')} ; {R('sent')}", Cell("Prudente : pas de gain stable", True, fill=LIGHT)],
    [Cell("H4", True), "+M3 vs +M4 (SenticNet)", f"{D('sent')} vs {D('m4s')}", f"{R('sent')} vs {R('m4s')}", Cell("Prudente : M4 non retenu", True, fill=LIGHT)],
], size=14, min_row=0.8))
s.add(Box(0.6, 5.95, 12.1, 0.85, [P("Configuration retenue sur validation : **M1+M2**, sur DistilBERT-base comme sur RoBERTa-base. "
                                    "NRC avec M4 (DistilBERT, 0,510 ± 0,003) donne un gain faible et n'est pas testé sur RoBERTa.", 16, NAVY, after=0)],
          fill=LNAVY, anchor="m", margins=(0.3, 0.08, 0.3, 0.08)))

# ----------------------------------------------------------------------------- 14. Discussion and limits
s = new("Discussion : positionnement et limites", notes=(
    "[1:00] À gauche, le positionnement. Nos scores M1+M2, 0,505 et 0,526, sont au-dessus des 0,46 de BERT-base "
    "publiés avec GoEmotions, et dans la zone des 0,52 de Seq2Emo. Une seule précaution : les encodeurs, les "
    "fenêtres de contexte et les budgets diffèrent, donc c'est un repère, pas une revendication d'état de l'art. "
    "L'argument principal reste interne, les ablations à budget constant. À droite, les limites, que l'on assume : "
    "un seul corpus, trois graines sans test statistique, quatre époques avec des hyperparamètres communs non "
    "réglés par configuration, un seuil unique de 0,5, et NRC avec M4 non testé sur RoBERTa."))
s.add(Box(0.6, 1.5, 5.9, 0.45, [P("Comparaison avec la littérature (F1-macro)", 18, NAVY, True, after=0)], margins=(0, 0, 0, 0)))
s.add(Table(0.6, 2.05, [3.3, 2.6], [
    ["Travail", "F1-macro"],
    ["BERT-base, GoEmotions (2020)", "≈ 0,46"],
    ["Seq2Emo (2021)", "≈ 0,52"],
    [Cell("HAKE-MER M1+M2, DistilBERT", True, LGREEN), Cell(fr(mean(ROWS["m12"][0])), True, LGREEN)],
    [Cell("HAKE-MER M1+M2, RoBERTa", True, LGREEN), Cell(fr(mean(ROWS["m12"][1])), True, LGREEN)],
], size=15, min_row=0.55))
s.add(Box(0.6, 5.1, 5.9, 1.65, [P("Encodeurs, fenêtres et budgets diffèrent : ces valeurs sont des repères. L'argument principal reste les "
                                  "comparaisons internes à budget constant.", 14, GRAY, italic=True, after=0)], margins=(0, 0, 0, 0)))
s.add(*card(6.9, 1.55, 5.8, 4.4, "Limites assumées", [
    "Un seul corpus (GoEmotions, Reddit)",
    "Trois graines, sans test statistique ni intervalle de confiance",
    "Quatre époques, hyperparamètres communs, non réglés par configuration",
    "Seuil de décision unique de 0,5",
    "Segmentation 4 × 32 tokens : troncature des textes longs",
    "NRC + M4 non testé sur RoBERTa"], accent=ORANGE, size=17, title_size=20))

# ----------------------------------------------------------------------------- 15. Conclusion
s = new("Contributions, conclusion et perspectives", notes=(
    "[1:15] Pour conclure. Trois contributions : une architecture modulaire, HAKE-MER, où chaque composante "
    "s'active par un drapeau ; un protocole d'ablation à budget constant sur deux encodeurs et trois graines ; "
    "et une lecture critique par classe et par confusions qui montre que la macro masque des comportements "
    "très différents. Le résultat : M1+M2 est la configuration à retenir, M1 seul non, M3 et M4 n'apportent pas "
    "de gain stable dans ce protocole. Pour les perspectives : traiter le déséquilibre avec des pertes "
    "repondérées, des seuils par émotion ; mieux fusionner le lexique, par émotion plutôt qu'avec une porte "
    "globale ; valider sur d'autres corpus, comme SemEval ; et comparer avec des modèles généralistes. Merci."))
cw = 3.9
s.add(*card(0.6, 1.6, cw, 5.1, "Contributions", [
    "HAKE-MER : architecture modulaire à quatre modules activables",
    "Ablation cumulative à budget constant, deux encodeurs, trois graines",
    "Analyse par classe, par cardinalité et par confusions"], accent=GREEN, size=18))
s.add(*card(0.6 + cw + 0.2, 1.6, cw, 5.1, "Conclusion", [
    "M1+M2 est la configuration retenue sur les deux encodeurs",
    "L'interaction entre émotions (M2) porte le gain",
    "M3 et M4 : pas de gain stable ici",
    "La macro masque des comportements très différents par classe"], accent=(0x5B, 0x7B, 0xA6), size=18))
s.add(*card(0.6 + 2 * (cw + 0.2), 1.6, cw, 5.1, "Perspectives", [
    "Pertes repondérées, seuils par émotion pour les classes rares",
    "Fusion lexicale par émotion plutôt que porte globale",
    "Validation externe : SemEval, autres corpus",
    "Comparaison avec des modèles généralistes"], accent=ORANGE, size=18))

# ----------------------------------------------------------------------------- 16. Questions
s = new(footer=False, notes=(
    "[Questions] Réponses préparées.\n"
    "Q1. Pourquoi M1 seul recule ? Le flat voit 128 tokens d'un bloc ; M1 découpe en 4 phrases de 32 tokens et "
    "agrège sans interaction entre émotions. La perte apparaît sur les deux encodeurs ; M2 la rattrape et la dépasse. "
    "Le mécanisme exact n'a pas été isolé davantage dans ce travail.\n"
    "Q2. Pourquoi si peu de gain avec les lexiques ? Un prior document global injecte la même information pour "
    "toutes les émotions ; NRC par étiquette donne un gain marginal sur DistilBERT. Les scores restent dans "
    "l'écart-type de M1+M2.\n"
    "Q3. Les résultats sont-ils significatifs ? Trois graines, écart-type de 0,002 à 0,008 sur la plupart des "
    "configurations ; pas de test statistique. Le gain de M2 est de 0,019 sur DistilBERT et 0,039 sur RoBERTa "
    "par rapport au flat, bien supérieur aux écarts-types, mais pas validé par un test formel.\n"
    "Q4. Pourquoi le F1-macro ? Il pondère toutes les émotions de la même façon, y compris les rares ; "
    "le F1-micro est dominé par neutral et les classes fréquentes.\n"
    "Q5. Comparaison avec l'état de l'art ? Repères seulement : encodeurs, fenêtres et budgets diffèrent.\n"
    "Q6. Pourquoi DistilBERT et RoBERTa ? DistilBERT est léger, RoBERTa plus fort : deux capacités "
    "pour vérifier que la conclusion ne dépend pas du backbone."))
s.add(Box(0.0, 0.0, 13.333, 7.5, fill=NAVY))
s.add(Box(0.6, 2.4, 12.1, 1.4, [P("Merci de votre attention", 44, WHITE, True, align="c", after=0)], anchor="m", margins=(0, 0, 0, 0)))
s.add(Box(0.6, 3.9, 12.1, 0.9, [P("Questions et discussion", 26, (0xB4, 0xE0, 0xC8), align="c", after=0)], anchor="m", margins=(0, 0, 0, 0)))
s.add(Box(0.6, 5.7, 12.1, 0.9, [P("Mariem El WEDANI  |  Encadrant : Sahbi BAHROUN  |  Institut Supérieur d'Informatique, 2025 - 2026", 14, (0xC9, 0xD1, 0xDC), align="c", after=0)],
          anchor="m", margins=(0, 0, 0, 0)))

# ----------------------------------------------------------------------------- A1. Full results
s = new("Annexe A : résultats complets de l'ablation", notes="Annexe pour les questions sur un chiffre précis. F1-macro test, moyenne ± écart-type, 3 graines.")
s.add(Table(0.6, 1.6, [5.0, 3.55, 3.55], [
    ["Configuration", "DistilBERT-base", "RoBERTa-base"],
    ["Step 0 (flat)", D("step0"), R("step0")],
    ["+M1", D("m1"), R("m1")],
    [Cell("+M1+M2", True, LGREEN), Cell(D("m12"), True, LGREEN), Cell(R("m12"), True, LGREEN)],
    ["+M1+M2+M3 (NRC)", D("nrc"), R("nrc")],
    ["+M1+M2+M3 (SenticNet)", D("sent"), R("sent")],
    ["+M1+M2+M3+M4 (SenticNet)", D("m4s"), R("m4s")],
    ["+M1+M2+M3+M4 (NRC)", D("m4n"), R("m4n")],
], size=15, min_row=0.55))
s.add(Box(0.6, 6.2, 12.1, 0.6, [P("F1-macro test, moyenne ± écart-type sur trois graines (42, 123, 456), protocole commun du chapitre 4.", 12, GRAY, italic=True, after=0)],
          margins=(0, 0, 0, 0)))

# ----------------------------------------------------------------------------- A2. Recommendations
s = new("Annexe B : recommandations d'usage", notes="Annexe : traduction du bilan en choix de déploiement.")
s.add(*card(0.6, 1.6, 5.95, 2.4, "Configuration par défaut", [
    "M1+M2 sur RoBERTa-base si les 2 points de macro justifient le coût",
    "M1+M2 sur DistilBERT-base pour un modèle plus léger"], accent=GREEN, size=19))
s.add(*card(6.75, 1.6, 5.95, 2.4, "Quand simplifier", [
    "Step 0 plat si la simplicité de déploiement prime",
    "Pas de M3 ni M4 si la macro prime"], accent=(0x5B, 0x7B, 0xA6), size=19))
s.add(*card(0.6, 4.2, 12.1, 2.5, "Coût et risques", [
    "M1+M2 encode jusqu'à quatre phrases par commentaire : coût modéré, réglages supplémentaires",
    "Changer le nombre de phrases ou la limite de 32 tokens impose de refaire l'analyse de troncature"], accent=ORANGE, size=19))

# ----------------------------------------------------------------------------- build
for i, sl in enumerate(slides, 1):
    sl.number = i
    if sl.footer_label is None:
        sl.footer_label = LABEL

if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/deck/Mariem_ElWedani_Soutenance.pptx")
    out.parent.mkdir(parents=True, exist_ok=True)
    render_pptx(slides, out, LABEL, dict(title="HAKE-MER : soutenance de Master", author="Mariem El WEDANI"))
    problems = render_preview(slides, "/tmp/deck/preview", LABEL)
    print(f"{len(slides)} slides -> {out}")
    for p in problems:
        print("PROBLEM", p)
