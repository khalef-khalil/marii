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
MINT = (0xB4, 0xE0, 0xC8)
PALE = (0x9F, 0xB0, 0xC8)
DIM = (0x3A, 0x55, 0x7E)
SECTIONS = ["Contexte", "Méthode", "Résultats", "Conclusion"]

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


def divider(active, notes):
    """Full-screen section slide, same structure as the Nejib defense deck."""
    s = new(footer=False, notes=notes)
    s.add(Box(0, 0, 13.333, 7.5, fill=NAVY))
    s.add(Box(0.90, 0.55, 6.5, 2.05, [P(f"{active + 1:02d}", 100, MINT, True, after=0)], margins=(0, 0, 0, 0)))
    s.add(Box(0.95, 2.70, 11.5, 1.25, [P(SECTIONS[active], 60, WHITE, True, after=0)], anchor="m", margins=(0, 0, 0, 0)))
    s.add(Box(0.98, 4.20, 1.60, 0.07, fill=MINT))
    x0, bw, gap = 0.95, 2.75, 0.20
    for i, name in enumerate(SECTIONS):
        x = x0 + i * (bw + gap)
        on = i == active
        s.add(Box(x, 5.75, bw, 0.09, fill=MINT if on else DIM))
        s.add(Box(x, 5.95, bw, 0.42, [P(name, 16, WHITE if on else PALE, on, after=0)], margins=(0, 0, 0, 0)))
    numbox = Box(11.05, 6.85, 1.80, 0.48, [P("00", 24, MINT, True, align="r", after=0)], anchor="m", margins=(0, 0, 0, 0))
    s.add(numbox)
    s._numbox = numbox
    return s


# ----------------------------------------------------------------------------- 1. Title
s = new(footer=False, notes=(
    "Bonjour. Je suis Mariem El Wedani, et je vous présente mon mémoire de Master, encadré par M. Sahbi Bahroun. "
    "Le sujet, c'est la détection de plusieurs émotions dans un même texte. Le modèle que j'ai construit s'appelle HAKE-MER."))
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

# ----------------------------------------------------------------------------- Plan
s = new("Plan", notes=(
    "Je vais avancer en quatre temps. "
    "D'abord, pourquoi ce sujet : la dataset, les limites des méthodes classiques, et mes hypothèses. "
    "Ensuite, comment HAKE-MER est construit, et comment j'ai rendu les tests comparables. "
    "Puis les résultats, y compris là où le score ne monte pas. "
    "Et pour finir, ce que je retiens, et ce que je ferais ensuite."))
for i, (title, sub) in enumerate([
    ("Contexte", "Les données, les limites et les hypothèses"),
    ("Méthode", "L'architecture et le protocole de test"),
    ("Résultats", "Les comparaisons, les classes et les erreurs"),
    ("Conclusion", "Le bilan et les pistes pour la suite"),
]):
    y = 1.62 + i * 1.25
    s.add(Box(0.6, y, 0.95, 0.72, [P(f"{i + 1:02d}", 30, GREEN, True, after=0)], anchor="m", margins=(0, 0, 0, 0)))
    s.add(Box(1.65, y, 4.3, 0.72, [P(title, 30, NAVY, True, after=0)], anchor="m", margins=(0, 0, 0, 0)))
    s.add(Box(6.15, y, 6.55, 0.72, [P(sub, 18, GRAY, after=0)], anchor="m", margins=(0, 0, 0, 0)))
    s.add(Box(0.6, y + 0.95, 12.1, 0.015, fill=LINE))

# ----------------------------------------------------------------------------- 01 Contexte
divider(0, "On commence par le contexte. Je vous montre d'abord sur quoi j'ai travaillé, puis les trois limites qui motivent le modèle, et enfin les hypothèses.")

# ----------------------------------------------------------------------------- Context and data
s = new("Contexte : un texte, plusieurs émotions", notes=(
    "L'idée de départ est simple : un même commentaire peut porter plusieurs émotions en même temps. "
    "On a travaillé sur la dataset GoEmotions. Elle se compose de commentaires Reddit, annotés avec 27 émotions, "
    "plus neutral, donc 28 classes. "
    "Le découpage est celui des auteurs : plus de 40 000 exemples pour l'entraînement, "
    "et un peu plus de 5 000 pour la validation comme pour le test. Les chiffres exacts sont sur la diapo. "
    "Le graphique, à droite, change un peu l'image qu'on se fait de la tâche : "
    "autour de 85 % des commentaires de test n'ont qu'une seule étiquette. "
    "Et les classes sont très inégales. neutral en a près de 1 800, grief seulement 6. "
    "Ce déséquilibre va revenir dans les résultats."))
s.add(Box(0.6, 1.55, 6.3, 5.2, [
    P("**Tâche** : prédire l'ensemble des émotions exprimées dans un commentaire (multi-étiquettes)", 18, bullet=True, after=12),
    P("**GoEmotions** : commentaires Reddit annotés avec 27 émotions et neutral, soit K = 28", 18, bullet=True, after=12),
    P("**Découpage officiel** : 43" + NB + "410 entraînement, 5" + NB + "426 validation, 5" + NB + "427 test", 18, bullet=True, after=12),
    P("**Déséquilibre marqué** : sur le test, neutral compte 1" + NB + "787 exemples positifs, grief seulement 6", 18, bullet=True, after=12),
    P(f"**Peu de cas multi-émotions** : {fr(single_share, 1)}{NB}% des commentaires de test portent une seule étiquette", 18, bullet=True, after=0)],
    margins=(0, 0, 0, 0)))
s.add(*fpic(FIG / "goemotions_cardinality.png", 7.2, 1.7, 5.5, "Nombre d'étiquettes gold par commentaire (test, n = 5" + NB + "427)"))

# ----------------------------------------------------------------------------- 3. Problem
s = new("Trois limites des méthodes classiques", notes=(
    "Sur cette dataset, les méthodes classiques butent sur trois points. "
    "Première carte, émotions décidées seules : chaque émotion est décidée toute seule. Or la joie et la tristesse vont rarement ensemble. "
    "Deuxième carte, structure du texte perdue : tout le commentaire est mis dans un seul vecteur, et les phrases se mélangent. "
    "Troisième carte, émotions rares : grief n'a que 6 exemples au test, neutral près de 1 800. Avec si peu d'exemples, le modèle apprend mal. "
    "D'où la question en bas. Si on sépare les phrases, si on relie les émotions, et si on ajoute un lexique, est-ce qu'on fait mieux ?"))
cw = 3.9
s.add(*card(0.6, 1.6, cw, 3.0, "Émotions décidées seules",
            ["Une sortie par émotion, sans lien entre elles", "La joie et la tristesse vont rarement ensemble"], size=17))
s.add(*card(0.6 + cw + 0.2, 1.6, cw, 3.0, "Structure du texte perdue",
            ["Un seul vecteur pour tout le commentaire", "Les phrases différentes se mélangent"], size=17))
s.add(*card(0.6 + 2 * (cw + 0.2), 1.6, cw, 3.0, "Émotions rares",
            ["grief : 6 exemples au test, neutral : 1" + NB + "787", "Très peu d'exemples pour apprendre"], size=17))
s.add(Box(0.6, 4.95, 12.1, 1.75, [
    P("Question de recherche", 14, MINT, True, after=4),
    P("Une architecture qui sépare les phrases, relie les émotions et utilise un lexique peut-elle mieux "
      "détecter les émotions multiples et rares qu'un modèle classique ?",
      20, WHITE, True, after=0)], fill=NAVY, anchor="m", margins=(0.35, 0.1, 0.35, 0.1)))

# ----------------------------------------------------------------------------- 4. Objectives and hypotheses
s = new("Objectifs et hypothèses", notes=(
    "Pour répondre à cette question, j'ai fixé quatre objectifs, à gauche. "
    "Concevoir HAKE-MER, avec quatre modules qu'on ajoute un par un. "
    "Tester chaque ajout dans les mêmes conditions, sur deux modèles. "
    "Comparer NRC et SenticNet. "
    "Et ne pas s'arrêter au F1-macro : regarder aussi classe par classe. "
    "À droite, chaque module a son hypothèse. H1 dit que M1 bat le modèle plat. H2, que M2 bat M1 seul. "
    "H3, qu'un lexique améliore encore. H4, qu'un poids par émotion fait mieux. "
    "Ce sont ces quatre lignes que les résultats vont trancher."))
s.add(Box(0.6, 1.55, 4.9, 5.2, [
    P("Objectifs", 20, NAVY, True, after=10),
    P("**Concevoir** HAKE-MER : quatre modules qu'on ajoute un par un", 18, bullet=True, after=12),
    P("**Tester** chaque ajout avec le même entraînement, sur deux modèles", 18, bullet=True, after=12),
    P("**Comparer** deux lexiques externes : NRC et SenticNet", 18, bullet=True, after=12),
    P("**Lire** les résultats classe par classe, pas seulement le F1-macro", 18, bullet=True, after=0)],
    margins=(0, 0, 0, 0)))
s.add(Table(5.85, 1.6, [0.7, 1.75, 4.4], [
    ["Hyp.", "Comparaison", "Énoncé testé"],
    [Cell("H1", True), "Plat contre +M1", "M1 améliore le F1-macro par rapport au modèle plat"],
    [Cell("H2", True), "+M1 contre +M1+M2", "M2 améliore le F1-macro par rapport à M1 seul"],
    [Cell("H3", True), "+M1+M2 contre +M3", "Un lexique (NRC ou SenticNet) améliore M1+M2"],
    [Cell("H4", True), "+M3 contre +M4", "Un poids par émotion améliore le F1-macro"],
], size=15, min_row=0.85))

# ----------------------------------------------------------------------------- 5. State of the art
s = new("Où se place HAKE-MER", notes=(
    "Avant l'architecture, je situe le travail. Trois cartes. "
    "À gauche, ce qui existe déjà pour plusieurs étiquettes : BERT est autour de 0,46 sur GoEmotions, Seq2Emo autour de 0,52. "
    "Au centre, l'attention par niveaux : d'abord les mots, puis les phrases. C'est la réponse directe à la structure du texte perdue. "
    "À droite, deux lexiques. NRC relie un mot à une émotion. SenticNet, lui, décrit des concepts. "
    "En bas, le point du mémoire : HAKE-MER met ces trois idées ensemble, et je peux allumer chaque module seul, pour voir ce qu'il apporte vraiment."))
cw = 3.9
s.add(*card(0.6, 1.6, cw, 3.55, "Plusieurs étiquettes",
            ["GoEmotions : BERT-base, F1-macro ≈ 0,46 (Demszky et al., 2020)",
             "Seq2Emo : liens entre étiquettes, ≈ 0,52 (Ke et al., 2021)"], accent=ORANGE, size=17))
s.add(*card(0.6 + cw + 0.2, 1.6, cw, 3.55, "Attention par niveaux",
            ["HAN : d'abord les mots, puis les phrases (Yang et al., 2016)", "Répond à la structure du texte perdue"], accent=GREEN, size=17))
s.add(*card(0.6 + 2 * (cw + 0.2), 1.6, cw, 3.55, "Lexiques externes",
            ["NRC : lien entre un mot et une émotion (Mohammad et Turney, 2013)",
             "SenticNet : concepts et sentiments (Cambria et al., 2018)"], accent=(0x5B, 0x7B, 0xA6), size=17))
s.add(Box(0.6, 5.5, 12.1, 1.2, [
    P("**HAKE-MER** réunit ces trois idées. On peut activer chaque module seul, pour mesurer ce qu'il apporte.",
      18, NAVY, after=0)],
    fill=LNAVY, anchor="m", margins=(0.3, 0.1, 0.3, 0.1)))

# ----------------------------------------------------------------------------- 02 Méthode
divider(1, "Passons à la méthode. Je vous montre d'abord comment le modèle est fait, puis comment j'ai rendu les tests comparables.")

# ----------------------------------------------------------------------------- Architecture
s = new("Architecture HAKE-MER : quatre modules", notes=(
    "En haut, le chemin principal, de gauche à droite. "
    "Le commentaire est coupé en quatre phrases de 32 tokens, puis lu par DistilBERT ou RoBERTa. "
    "M1 résume les mots de chaque phrase. "
    "M2 travaille en deux temps : chaque émotion regarde les phrases, puis les 28 émotions se comparent entre elles. "
    "M4 donne un poids à chaque émotion, et on sort 28 réponses. "
    "En bas, la branche lexicale. M3 lit NRC ou SenticNet, et l'ajoute avec une porte g. Au départ, g vaut 0 : le lexique n'entre pas tout de suite. "
    "Si on éteint M3, on retombe simplement sur M1+M2. "
    "La barre du bas relie chaque module à une limite : M1 à la structure du texte perdue, M2 aux émotions décidées seules, M3 aux émotions rares."))
BW, BH, BY = 1.47, 1.35, 2.0
tops = [
    ("Texte d'entrée", "4 phrases × 32 tokens", LIGHT, LINE, None),
    ("Modèle partagé", "DistilBERT ou RoBERTa", LIGHT, LINE, None),
    ("M1", "Des mots vers la phrase", LORANGE, ORANGE, None),
    ("M2", "L'émotion regarde les phrases", LNAVY, (0x5B, 0x7B, 0xA6), None),
    ("M2", "Les émotions entre elles", LNAVY, (0x5B, 0x7B, 0xA6), None),
    ("M4", "Un poids par émotion", LILAC, VIOLET, None),
    ("Sortie", "28 réponses", LIGHT, LINE, None),
]
s.add(Box(0.6, 1.5, 4.0, 0.35, [P("Chemin principal", 13, GRAY, True, italic=True, after=0)], margins=(0, 0, 0, 0)))
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
s.add(Box(2.75, LY, 2.4, LH, [P("M3 : projection", 14, NAVY, True, align="c", after=2), P("porte g = 0 au départ", 12, INK, align="c", after=0)],
          fill=LGREEN, line=GREEN, line_w=1.5, shape="round", anchor="m"))
s.add(Arrow(5.2, LY + LH / 2 - 0.12, 0.3, 0.24))
s.add(Box(5.55, LY, 2.4, LH, [P("Requêtes d'émotions", 14, NAVY, True, align="c", after=2), P("Qk + g · prior", 12, INK, align="c", after=0)],
          fill=LNAVY, line=(0x5B, 0x7B, 0xA6), line_w=1.5, shape="round", anchor="m"))
# arrow up from queries to M2 (cross-attention box center x = 0.6 + 3*1.77 + 0.735 = 6.645)
s.add(Arrow(6.525, BY + BH + 0.05, 0.24, LY - (BY + BH) - 0.1, direction="up"))
s.add(Box(8.4, LY, 4.3, LH, [P("Si M3 est éteint, g = 0. Les requêtes ne changent pas : on revient à M1+M2.", 14, INK, after=0)],
          anchor="m", margins=(0.08, 0.06, 0.08, 0.06)))
s.add(Box(0.6, 5.8, 12.1, 0.95, [
    P("**M1** garde les phrases (structure du texte perdue).  **M2** relie les émotions (émotions décidées seules).  "
      "**M3** ajoute un lexique (émotions rares).  **M4** donne un poids à chaque émotion.", 16, NAVY, after=0)],
    fill=LIGHT, anchor="m", margins=(0.3, 0.1, 0.3, 0.1)))

# ----------------------------------------------------------------------------- 7. Protocol
s = new("Protocole : le même entraînement pour tous les tests", notes=(
    "Pour que les comparaisons soient justes, tous les tests ont le même entraînement. "
    "On a travaillé sur la dataset GoEmotions : 28 classes, avec le découpage des auteurs. "
    "Les deux modèles sont DistilBERT et RoBERTa. On part du modèle plat, Step 0, puis on ajoute les modules un par un. "
    "Le score que je regarde, c'est le F1-macro au test. Et je garde le modèle qui était le meilleur en validation. "
    "Chaque configuration est lancée trois fois, avec trois graines fixes. "
    "Le tableau, à droite, fixe le reste : quatre époques, des lots de 16, un taux très faible, un seuil à 0,5. "
    "Une seule différence de format : Step 0 lit le texte d'un bloc. À partir de M1, on le coupe en quatre phrases courtes."))
s.add(Box(0.6, 1.55, 6.3, 5.2, [
    P("**Dataset** : GoEmotions, 28 classes, découpage des auteurs", 19, bullet=True, after=14),
    P("**Modèles** : DistilBERT-base et RoBERTa-base", 19, bullet=True, after=14),
    P("**Tests** : Step 0, +M1, +M1+M2, +M3, +M4, même entraînement", 19, bullet=True, after=14),
    P("**Score principal** : F1-macro au test. On choisit le meilleur modèle sur la validation.", 19, bullet=True, after=14),
    P("**Trois essais** : graines 42, 123 et 456. On donne la moyenne et l'écart-type.", 19, bullet=True, after=0)],
    margins=(0, 0, 0, 0)))
s.add(Table(7.2, 1.6, [2.6, 2.9], [
    ["Paramètre commun", "Valeur"],
    ["Époques", "4"],
    ["Taille de lot", "16"],
    ["Taux d'apprentissage", "0,00005"],
    ["Optimiseur", "AdamW, décroissance 0,01"],
    ["Montée du taux", "10 % des pas"],
    ["Seuil de décision", "0,5"],
    ["Texte (Step 0)", "128 tokens"],
    ["Texte (M1 et plus)", "4 phrases × 32 tokens"],
], size=14, min_row=0.5))

# ----------------------------------------------------------------------------- 03 Résultats
divider(2, "Voici les résultats. Je commence par DistilBERT, puis je vérifie que RoBERTa raconte la même histoire.")

# ----------------------------------------------------------------------------- Results DistilBERT
s = new("Résultat 1 : M1+M2 est la seule étape qui progresse", notes=(
    "Ce graphique, c'est le F1-macro au test pour DistilBERT : la moyenne, et l'écart-type, sur trois essais. "
    "L'axe est coupé en bas, sinon les barres se ressemblent trop. "
    "Le point qui saute aux yeux : M1 seul recule par rapport au modèle plat. "
    "Puis M1+M2 remonte, avec un gain d'environ 0,02 sur Step 0. "
    "M3 et M4 restent au même niveau. Ils ne dépassent pas M1+M2. "
    "Les valeurs exactes sont sur les cartes. Sur DistilBERT, une seule étape fait vraiment progresser le score."))
s.add(*fpic(FIG / "ablation_distilbert.png", 0.6, 1.5, 7.7, "F1-macro au test, DistilBERT, moyenne ± écart-type sur 3 essais. L'axe commence à 0,42."))
s.add(*card(8.6, 1.6, 4.1, 1.55, "M1 seul recule", [f"{fr(mean(ROWS['m1'][0]))} contre {fr(mean(ROWS['step0'][0]))} pour le flat"], accent=ORANGE, size=17))
s.add(*card(8.6, 3.3, 4.1, 1.55, "M1+M2 progresse", [f"{fr(mean(ROWS['m12'][0]))}, soit +{fr(gain_db)} sur Step 0"], accent=GREEN, size=17))
s.add(*card(8.6, 5.0, 4.1, 1.75, "M3 et M4 sans gain", ["0,499 à 0,504, aucun ne dépasse M1+M2"], accent=(0x5B, 0x7B, 0xA6), size=17))

# ----------------------------------------------------------------------------- 9. Results RoBERTa
s = new("Résultat 2 : le même profil sur RoBERTa-base", notes=(
    "Est-ce que c'est seulement DistilBERT ? Le graphique met les deux modèles côte à côte : DistilBERT en clair, RoBERTa en foncé. "
    "Le profil est le même. M1 seul ne dépasse le modèle plat sur aucun des deux. "
    "M1+M2 monte des deux côtés. RoBERTa est un peu devant, d'environ 0,02, mais le gain de M2 apparaît sur les deux. "
    "Ce n'est donc pas un effet du modèle choisi. Les chiffres exacts sont dans le tableau."))
s.add(*fpic(FIG / "backbone.png", 0.6, 1.55, 6.6, "F1-macro test, moyenne ± écart-type sur 3 graines."))
s.add(Table(7.5, 1.6, [2.0, 1.6, 1.6], [
    ["Configuration", "DistilBERT", "RoBERTa"],
    ["Step 0 (flat)", fr(mean(ROWS["step0"][0])), fr(mean(ROWS["step0"][1]))],
    ["+M1", fr(mean(ROWS["m1"][0])), fr(mean(ROWS["m1"][1]))],
    [Cell("+M1+M2", True, LGREEN), Cell(fr(mean(ROWS["m12"][0])), True, LGREEN), Cell(fr(mean(ROWS["m12"][1])), True, LGREEN)],
], size=15, min_row=0.55))
s.add(Box(7.5, 4.0, 5.2, 2.7, [
    P(f"RoBERTa dépasse DistilBERT d'environ {fr(gap_bb)} sur M1+M2", 18, bullet=True, after=12),
    P("M1 seul ne dépasse le modèle plat sur aucun des deux", 18, bullet=True, after=12),
    P("Le gain de M2 apparaît sur les deux modèles", 18, bullet=True, after=0)], margins=(0, 0, 0, 0)))

# ----------------------------------------------------------------------------- 10. M2 controls
s = new("Pourquoi M2 aide : deux contrôles", notes=(
    "M2 aide, mais pourquoi ? Le tableau isole deux pièces. "
    "La ligne verte, c'est M1+M2 complet. "
    "Si on enlève le lien entre les émotions, la ligne orange, on perd clairement : "
    "près de 0,03 sur DistilBERT, environ 0,02 sur RoBERTa. "
    "Si on enlève seulement le regard vers les phrases, les scores restent proches : l'écart tient dans l'écart-type. "
    "Le gain vient donc surtout du lien entre les émotions. Cela répond aux émotions décidées seules, pas à la structure du texte perdue."))
s.add(Table(0.6, 1.65, [4.6, 3.3, 3.3], [
    ["Configuration", "DistilBERT", "RoBERTa"],
    ["M1 seul", D("m1"), R("m1")],
    [Cell("M1+M2 complet", True, LGREEN), Cell(D("m12"), True, LGREEN), Cell(R("m12"), True, LGREEN)],
    ["M1+M2 sans lien entre émotions", Cell(D("noenc"), fill=LORANGE), Cell(R("noenc"), fill=LORANGE)],
    ["M1+M2 sans regard vers les phrases", D("noxa"), R("noxa")],
], size=16, min_row=0.65))
s.add(Box(0.6, 5.0, 11.2, 1.75, [
    P(f"Sans le lien entre émotions : perte de {fr(loss_enc_db)} (DistilBERT) et {fr(loss_enc_rb)} (RoBERTa)", 17, bullet=True, after=8),
    P("Sans le regard vers les phrases, le F1-macro change peu : l'écart reste dans l'écart-type", 17, bullet=True, after=8),
    P("Le gain de M2 répond aux émotions décidées seules", 17, bullet=True, after=0)],
    margins=(0, 0, 0, 0)))

# ----------------------------------------------------------------------------- 11. Per class
s = new("Au-delà du F1-macro : où M2 aide, où il nuit", notes=(
    "Le F1-macro ne dit pas tout. Ici, douze émotions, des plus rares aux plus fréquentes. Step 0 en gris, M1+M2 en vert. "
    "M2 aide surtout les émotions de fréquence moyenne : desire, annoyance, curiosity. "
    "grief et relief montent aussi, mais avec une poignée d'exemples seulement. Je ne m'y fie pas trop. "
    "Et ce n'est pas que du gain : nervousness et embarrassment baissent de près d'un dixième. "
    "L'écart entre classes fréquentes et classes très rares se réduit. "
    "Avec M1 seul, il s'aggrave. Les valeurs sont sur les cartes. Le gain de M2 ne vient donc pas des classes les plus rares."))
s.add(*fpic(FIG / "per_class.png", 0.6, 1.5, 7.0, "F1 test par émotion, DistilBERT, moyenne sur 3 graines, labels par effectif croissant."))
s.add(*card(8.6, 1.6, 4.1, 1.9, "Gains",
            ["Fréquence intermédiaire : desire, annoyance, curiosity", "grief et relief progressent, avec 6 et 11 exemples de test : fragile"], accent=GREEN, size=14))
s.add(*card(8.6, 3.65, 4.1, 1.4, "Pertes", ["nervousness et embarrassment reculent d'environ 0,09"], accent=ORANGE, size=14))
s.add(*card(8.6, 5.2, 4.1, 1.55, "Écart fréquent vs rare",
            ["0,365 (flat) → 0,296 (M1+M2)", "M1 seul l'élargit : 0,405"], accent=(0x5B, 0x7B, 0xA6), size=14))
s.add(Box(0.6, 6.1, 7.4, 0.75, [P("Le gain de M2 vient des émotions de fréquence moyenne, pas des classes les plus rares.", 15, NAVY, True, after=0)],
          fill=LNAVY, anchor="m", margins=(0.2, 0.08, 0.2, 0.08)))

# ----------------------------------------------------------------------------- 12. Errors
s = new("Les erreurs : neutral est au centre", notes=(
    "Les erreurs confirment autre chose. Les dix motifs les plus fréquents, pour M1+M2, sur les trois essais réunis, "
    "passent tous par neutral. approval est prédit neutral, ou l'inverse. On retrouve aussi annoyance et disapproval. "
    "Ce n'est pas une confusion entre deux émotions proches. "
    "C'est la frontière entre une émotion faible et pas d'émotion du tout. "
    "D'où la piste, à droite : mieux séparer neutral."))
s.add(*fpic(FIG / "errors.png", 0.6, 1.6, 8.1, "Dix motifs d'erreur les plus fréquents, M1+M2, DistilBERT, 3 graines cumulées (test)."))
s.add(*card(9.0, 1.6, 3.7, 2.4, "Constat", ["Les dix motifs impliquent tous neutral", "Approval, annoyance, disapproval vs neutral dominent"], accent=ORANGE, size=15))
s.add(*card(9.0, 4.2, 3.7, 2.4, "Lecture", ["Limite floue entre une émotion faible et pas d'émotion", "Piste : mieux séparer neutral"], accent=GREEN, size=15))

# ----------------------------------------------------------------------------- 13. Hypotheses verdict
s = new("Bilan des hypothèses : une seule est soutenue", notes=(
    "On peut maintenant trancher. "
    "H1 n'est pas soutenue : M1 seul est plus bas que Step 0, sur les deux modèles. "
    "La seule ligne verte, c'est H2. M1+M2 dépasse bien M1 seul. "
    "H3, je la laisse prudente : le lexique ne donne pas un gain stable. Sur RoBERTa, NRC bouge beaucoup d'un essai à l'autre. "
    "H4 aussi : avec SenticNet, M4 ne fait pas mieux, donc je ne le retiens pas. "
    "En bas, le choix : M1+M2, sur DistilBERT comme sur RoBERTa. "
    "NRC avec M4 est un peu plus haut, mais seulement sur DistilBERT. Ce test n'existe pas pour RoBERTa, je ne peux pas le généraliser."))
assert abs((mean(ROWS["m12"][0]) - mean(ROWS["m1"][0])) - 0.048) < 0.0015 and abs((mean(ROWS["m12"][1]) - mean(ROWS["m1"][1])) - 0.041) < 0.0015
s.add(Table(0.6, 1.6, [0.7, 2.7, 3.3, 3.1, 2.3], [
    ["Hyp.", "Comparaison", "DistilBERT", "RoBERTa", "Verdict"],
    [Cell("H1", True), "Step 0 contre +M1", f"{D('step0')} contre {D('m1')}", f"{R('step0')} contre {R('m1')}", Cell("Non soutenue", True, fill=LORANGE)],
    [Cell("H2", True), "+M1 contre +M1+M2", f"{D('m1')} contre {D('m12')}", f"{R('m1')} contre {R('m12')}", Cell("Soutenue", True, fill=LGREEN)],
    [Cell("H3", True), "+M1+M2 contre +M3", f"{D('m12')} contre NRC {D('nrc')} ; SenticNet {D('sent')}", f"{R('m12')} contre NRC {R('nrc')} ; SenticNet {R('sent')}", Cell("Prudente : pas de gain stable", True, fill=LIGHT)],
    [Cell("H4", True), "+M3 contre +M4, SenticNet", f"{D('sent')} contre {D('m4s')}", f"{R('sent')} contre {R('m4s')}", Cell("Prudente : M4 non retenu", True, fill=LIGHT)],
], size=14, min_row=0.8))
s.add(Box(0.6, 5.95, 12.1, 0.85, [P("Modèle retenu, d'après la validation : **M1+M2**, sur DistilBERT comme sur RoBERTa. "
                                    "NRC avec M4 fait 0,510 ± 0,003 sur DistilBERT seulement. Ce test n'existe pas pour RoBERTa.", 16, NAVY, after=0)],
          fill=LNAVY, anchor="m", margins=(0.3, 0.08, 0.3, 0.08)))

# ----------------------------------------------------------------------------- 04 Conclusion
divider(3, "Pour finir, je replace ces scores, je dis clairement ce que le travail ne couvre pas, puis ce que j'en retiens.")

# ----------------------------------------------------------------------------- Discussion and limits
s = new("Discussion : comparaison et limites", notes=(
    "À gauche, nos scores M1+M2 : un peu au-dessus de 0,50 pour DistilBERT, un peu plus haut pour RoBERTa. "
    "En face, BERT sur GoEmotions est autour de 0,46, Seq2Emo autour de 0,52. "
    "Je les montre comme des repères, pas comme une compétition : les modèles et les réglages ne sont pas les mêmes. "
    "Ce qui compte, ce sont nos comparaisons, faites avec le même entraînement. "
    "À droite, les limites, et elles sont réelles. Une seule dataset, GoEmotions, des commentaires Reddit. "
    "Trois essais, sans test statistique. Quatre époques, les mêmes réglages pour tous. Un seul seuil, 0,5. "
    "Des textes coupés en quatre phrases. Et NRC avec M4, non testé sur RoBERTa."))
s.add(Box(0.6, 1.5, 5.9, 0.45, [P("Comparaison avec la littérature (F1-macro)", 18, NAVY, True, after=0)], margins=(0, 0, 0, 0)))
s.add(Table(0.6, 2.05, [3.3, 2.6], [
    ["Travail", "F1-macro"],
    ["BERT-base, GoEmotions (2020)", "≈ 0,46"],
    ["Seq2Emo (2021)", "≈ 0,52"],
    [Cell("HAKE-MER M1+M2, DistilBERT", True, LGREEN), Cell(fr(mean(ROWS["m12"][0])), True, LGREEN)],
    [Cell("HAKE-MER M1+M2, RoBERTa", True, LGREEN), Cell(fr(mean(ROWS["m12"][1])), True, LGREEN)],
], size=15, min_row=0.55))
s.add(Box(0.6, 5.1, 5.9, 1.65, [P("Les modèles et les réglages ne sont pas les mêmes : ces chiffres sont des repères. "
                                  "L'argument principal reste nos comparaisons, avec le même entraînement.", 15, GRAY, italic=True, after=0)], margins=(0, 0, 0, 0)))
s.add(*card(6.9, 1.55, 5.8, 4.4, "Limites du travail", [
    "Une seule dataset : GoEmotions, des commentaires Reddit",
    "Trois essais, sans test statistique",
    "Quatre époques, et les mêmes réglages pour tous",
    "Un seul seuil de décision : 0,5",
    "Textes coupés en 4 phrases de 32 tokens",
    "NRC + M4 non testé sur RoBERTa"], accent=ORANGE, size=17, title_size=20))

# ----------------------------------------------------------------------------- 15. Conclusion
s = new("Contributions, conclusion et perspectives", notes=(
    "Trois colonnes pour fermer. "
    "Contributions : une architecture en quatre modules qu'on peut allumer un par un, des tests comparables sur deux modèles "
    "et trois essais, et une lecture qui va jusqu'aux erreurs. "
    "Conclusion : je retiens M1+M2. Le gain vient du lien entre les émotions. M3 et M4 n'apportent pas un gain stable ici. "
    "Et le F1-macro cache des différences entre classes. "
    "Perspectives : plus de poids aux classes rares, lier le lexique à chaque émotion plutôt qu'avec une seule porte, "
    "tester d'autres datasets, par exemple SemEval, et comparer avec des modèles plus généraux."))
cw = 3.9
s.add(*card(0.6, 1.6, cw, 5.1, "Contributions", [
    "HAKE-MER : quatre modules qu'on peut activer un par un",
    "Tests avec le même entraînement, deux modèles, trois essais",
    "Lecture par classe et par type d'erreur"], accent=GREEN, size=18))
s.add(*card(0.6 + cw + 0.2, 1.6, cw, 5.1, "Conclusion", [
    "On retient M1+M2 sur les deux modèles",
    "Le gain vient du lien entre les émotions (M2)",
    "M3 et M4 : pas de gain stable ici",
    "Le F1-macro cache des différences entre classes"], accent=(0x5B, 0x7B, 0xA6), size=18))
s.add(*card(0.6 + 2 * (cw + 0.2), 1.6, cw, 5.1, "Perspectives", [
    "Plus de poids aux classes rares, et un seuil par émotion",
    "Lier le lexique à chaque émotion, pas avec une seule porte",
    "Tester sur d'autres datasets, par exemple SemEval",
    "Comparer avec des modèles plus généraux"], accent=ORANGE, size=18))

# ----------------------------------------------------------------------------- 16. Questions
s = new(footer=False, notes=(
    "Je vous remercie de votre attention. Je suis prête pour vos questions.\n"
    "Si on me demande pourquoi M1 seul baisse : le modèle plat voit 128 tokens d'un bloc. M1, lui, coupe le texte en quatre phrases, "
    "et il n'y a pas encore de lien entre les émotions. C'est M2 qui rattrape cette baisse.\n"
    "Si on me demande pourquoi les lexiques aident peu : le lexique est le même pour toutes les émotions, et il entre par une seule porte. "
    "Le gain reste petit, et il n'est pas stable.\n"
    "Si on me demande si les résultats sont solides : j'ai trois essais, et le gain de M2 est plus grand que l'écart entre ces essais. "
    "En revanche, je n'ai pas fait de test statistique, donc je ne dis pas que la différence est prouvée.\n"
    "Si on me demande pourquoi le F1-macro : il donne le même poids à chaque émotion, y compris grief, qui n'a que 6 exemples. "
    "Un score global cacherait les classes rares."))
s.add(Box(0.0, 0.0, 13.333, 7.5, fill=NAVY))
s.add(Box(0.6, 2.4, 12.1, 1.4, [P("Merci de votre attention", 44, WHITE, True, align="c", after=0)], anchor="m", margins=(0, 0, 0, 0)))
s.add(Box(0.6, 3.9, 12.1, 0.9, [P("Questions et discussion", 26, (0xB4, 0xE0, 0xC8), align="c", after=0)], anchor="m", margins=(0, 0, 0, 0)))
s.add(Box(0.6, 5.7, 12.1, 0.9, [P("Mariem El WEDANI  |  Encadrant : Sahbi BAHROUN  |  Institut Supérieur d'Informatique, 2025 - 2026", 14, (0xC9, 0xD1, 0xDC), align="c", after=0)],
          anchor="m", margins=(0, 0, 0, 0)))

# ----------------------------------------------------------------------------- A1. Full results
s = new("Annexe A : tous les résultats", notes=(
    "Si on me demande un chiffre précis, il est ici. "
    "Chaque ligne, c'est le F1-macro au test, moyenne et écart-type, sur les trois essais. "
    "La ligne verte est le modèle que je retiens, M1+M2. "
    "La dernière ligne, NRC avec M4, s'arrête à DistilBERT : elle n'a pas été testée sur RoBERTa."))
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
s.add(Box(0.6, 6.2, 12.1, 0.6, [P("F1-macro au test, moyenne ± écart-type sur trois essais (42, 123, 456), avec les mêmes réglages.", 13, GRAY, italic=True, after=0)],
          margins=(0, 0, 0, 0)))

# ----------------------------------------------------------------------------- A2. Recommendations
s = new("Annexe B : quel modèle choisir", notes=(
    "Et si on me demande quel modèle choisir. Par défaut, M1+M2. "
    "RoBERTa, si un gain d'environ 0,02 vaut le coût en calcul. DistilBERT, si on veut quelque chose de plus léger. "
    "On peut aussi rester sur le modèle plat, quand on veut un système simple. "
    "Et je n'ajoute ni M3 ni M4 si seul le F1-macro compte. "
    "Dernier point : M1+M2 lit jusqu'à quatre phrases. Si on change ce découpage, les chiffres d'aujourd'hui ne tiennent plus. Il faut refaire les tests."))
s.add(*card(0.6, 1.6, 5.95, 2.4, "Choix par défaut", [
    f"M1+M2 avec RoBERTa si le gain d'environ {fr(gap_bb)} vaut le coût",
    "M1+M2 avec DistilBERT si on veut un modèle plus léger"], accent=GREEN, size=18))
s.add(*card(6.75, 1.6, 5.95, 2.4, "Quand simplifier", [
    "Le modèle plat (Step 0) si on veut un système simple",
    "Pas de M3 ni de M4 si seul le F1-macro compte"], accent=(0x5B, 0x7B, 0xA6), size=18))
s.add(*card(0.6, 4.2, 12.1, 2.5, "Coût et risques", [
    "M1+M2 lit jusqu'à quatre phrases par commentaire : le coût reste modéré",
    "Si on change le nombre de phrases ou la limite de 32 tokens, il faut refaire les tests"], accent=ORANGE, size=18))

# ----------------------------------------------------------------------------- build
for i, sl in enumerate(slides, 1):
    sl.number = i
    if sl.footer_label is None:
        sl.footer_label = LABEL
    numbox = getattr(sl, "_numbox", None)
    if numbox is not None:
        numbox["paras"][0]["runs"][0]["t"] = f"{i:02d}"

if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/deck/Mariem_ElWedani_Soutenance.pptx")
    out.parent.mkdir(parents=True, exist_ok=True)
    render_pptx(slides, out, LABEL, dict(title="HAKE-MER : soutenance de Master", author="Mariem El WEDANI"))
    problems = render_preview(slides, "/tmp/deck/preview", LABEL)
    print(f"{len(slides)} slides -> {out}")
    for p in problems:
        print("PROBLEM", p)
