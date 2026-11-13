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
SECTIONS = ["Contexte", "État de l'art", "Approche proposée", "Expérimentation", "Conclusion"]

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
    title = SECTIONS[active]
    title_size = 44 if len(title) > 14 else 56
    s.add(Box(0.55, 2.70, 12.2, 1.25, [P(title, title_size, WHITE, True, after=0)], anchor="m", margins=(0, 0, 0, 0)))
    s.add(Box(0.98, 4.20, 1.60, 0.07, fill=MINT))
    n = len(SECTIONS)
    gap, x0 = 0.12, 0.48
    bw = (13.333 - x0 - 0.40 - gap * (n - 1)) / n
    for i, name in enumerate(SECTIONS):
        x = x0 + i * (bw + gap)
        on = i == active
        s.add(Box(x, 5.75, bw, 0.09, fill=MINT if on else DIM))
        s.add(Box(x, 5.95, bw, 0.62, [P(name, 13, WHITE if on else PALE, on, align="l", after=0)], margins=(0, 0, 0, 0)))
    numbox = Box(11.05, 6.85, 1.80, 0.48, [P("00", 24, MINT, True, align="r", after=0)], anchor="m", margins=(0, 0, 0, 0))
    s.add(numbox)
    s._numbox = numbox
    return s


# ----------------------------------------------------------------------------- 1. Title
s = new(footer=False, notes=(
    "Bonjour. Je suis Mariem El Wedani. Mon encadrant est M. Sahbi Bahroun. "
    "Un même texte peut porter plusieurs émotions. Le modèle s'appelle HAKE-MER."))
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
    "Cinq parties. "
    "Le contexte : la tâche, les embeddings, l'attention. "
    "L'état de l'art : ce qui existe déjà. "
    "L'approche proposée : HAKE-MER, étape par étape. "
    "L'expérimentation : le protocole, puis l'ablation. "
    "Et la conclusion."))
for i, (title, sub) in enumerate([
    ("Contexte", "La tâche, les embeddings et l'attention"),
    ("État de l'art", "Ce qui a déjà été proposé"),
    ("Approche proposée", "HAKE-MER, étape par étape"),
    ("Expérimentation", "Le protocole et l'étude d'ablation"),
    ("Conclusion", "Le bilan et les pistes pour la suite"),
]):
    y = 1.48 + i * 1.02
    s.add(Box(0.6, y, 0.85, 0.68, [P(f"{i + 1:02d}", 26, GREEN, True, after=0)], anchor="m", margins=(0, 0, 0, 0)))
    s.add(Box(1.55, y, 4.85, 0.68, [P(title, 26, NAVY, True, after=0)], anchor="m", margins=(0, 0, 0, 0)))
    s.add(Box(6.55, y, 6.15, 0.68, [P(sub, 16, GRAY, after=0)], anchor="m", margins=(0, 0, 0, 0)))
    s.add(Box(0.6, y + 0.82, 12.1, 0.015, fill=LINE))

# ----------------------------------------------------------------------------- 01 Contexte
divider(0, "D'abord la tâche. Puis comment un texte devient un vecteur. Puis pourquoi l'attention compte.")

# ----------------------------------------------------------------------------- Context and data
s = new("Contexte : un texte, plusieurs émotions", notes=(
    "La tâche : prédire toutes les émotions du commentaire, pas une seule. "
    "GoEmotions : des commentaires Reddit, 27 émotions plus neutral, donc 28 classes. Le découpage est celui des auteurs. "
    "Les classes sont très inégales. neutral a 1 787 exemples, grief seulement 6. "
    "Et la plupart des commentaires de test n'ont qu'une étiquette. Le graphique le montre."))
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
    "Trois limites. Les émotions sont décidées seules. "
    "La structure du texte se perd dans un seul vecteur. "
    "Les émotions rares ont très peu d'exemples. "
    "La question : est-ce qu'on fait mieux si on sépare les phrases, si on relie les émotions, et si on ajoute un lexique ?"))
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

# ----------------------------------------------------------------------------- Embeddings
s = new("Des méthodes d'embeddings", notes=(
    "D'abord un vecteur fixe : Word2Vec et GloVe. Le mot ne change pas. "
    "Ensuite ELMo : le mot bouge avec la phrase. "
    "Puis BERT. GoEmotions part de là. "
    "Ce mémoire utilise DistilBERT et RoBERTa. Step 0, c'est cet affinage. "
    "En bas : même ce vecteur ne dit pas quelle phrase porte l'émotion."))
gens = [
    ("01", "Vecteur fixe", "Word2Vec, GloVe", "Le mot garde le même vecteur, quelle que soit la phrase.", LORANGE, ORANGE),
    ("02", "La phrase compte", "ELMo", "Le même mot bouge selon les mots qui l'entourent.", LNAVY, (0x5B, 0x7B, 0xA6)),
    ("03", "Affinage", "BERT", "GoEmotions part de BERT, avec une sortie par émotion.", LIGHT, NAVY),
    ("04", "Ce mémoire", "DistilBERT, RoBERTa", "Step 0 est cet affinage. Les modules viennent après.", LGREEN, GREEN),
]
gw, ggap = 2.85, 0.32
for i, (num, title, name, line, fill, accent) in enumerate(gens):
    x = 0.55 + i * (gw + ggap)
    s.add(Box(x, 1.6, gw, 3.35, [
        P(num, 14, accent, True, align="c", after=2),
        P(title, 20, NAVY, True, align="c", after=6),
        P(name, 16, accent, True, align="c", after=8),
        P(line, 14, INK, align="c", after=0)],
        fill=fill, line=accent, line_w=1.5, shape="round", anchor="m", margins=(0.16, 0.12, 0.16, 0.12)))
    if i < len(gens) - 1:
        s.add(Arrow(x + gw + 0.03, 3.05, ggap - 0.06, 0.26))
s.add(Box(0.55, 5.2, 12.2, 1.45, [
    P("GloVe reste dans des modèles plus tardifs, comme Seq2Emo, à côté d'ELMo.", 16, INK, after=6),
    P("Même contextuel, ce vecteur ne dit pas encore quelle phrase porte quelle émotion.", 16, NAVY, True, after=0)],
    fill=LIGHT, anchor="m", margins=(0.28, 0.12, 0.28, 0.12)))

# ----------------------------------------------------------------------------- Attention
s = new("Pourquoi le mécanisme d'attention compte", notes=(
    "Sans poids, « merci » et « déçu » tombent dans le même vecteur. "
    "Avec une attention de phrase, chaque phrase a son vecteur. "
    "En bas : BERT et RoBERTa ont déjà cette attention à l'intérieur. M1 et M2 en ajoutent une au-dessus."))
s.add(Box(0.55, 1.5, 2.3, 0.4, [P("Sans poids", 14, ORANGE, True, after=0)], margins=(0, 0, 0, 0)))
s.add(Box(0.55, 1.95, 12.2, 1.35, [
    P("merci pour l'aide,   mais je suis déçu", 22, INK, align="c", after=4),
    P("Les deux phrases tombent dans le même vecteur.", 15, GRAY, align="c", after=0)],
    fill=LORANGE, shape="round", anchor="m"))
s.add(Box(0.55, 3.5, 3.4, 0.4, [P("Avec une attention de phrase", 14, GREEN, True, after=0)], margins=(0, 0, 0, 0)))
s.add(Box(0.55, 3.95, 5.7, 1.25, [
    P("merci   ·   pour   ·   l'aide", 18, NAVY, True, align="c", after=4),
    P("un vecteur", 14, GRAY, align="c", after=0)],
    fill=LGREEN, line=GREEN, line_w=1.25, shape="round", anchor="m"))
s.add(Box(6.55, 3.95, 6.2, 1.25, [
    P("mais   ·   je suis   ·   déçu", 18, NAVY, True, align="c", after=4),
    P("un autre vecteur", 14, GRAY, align="c", after=0)],
    fill=LGREEN, line=GREEN, line_w=1.25, shape="round", anchor="m"))
s.add(Box(0.55, 5.45, 12.2, 1.2, [
    P("Dans BERT et RoBERTa, l'auto-attention est déjà là : elle rend le vecteur dépendant de la phrase.", 16, INK, after=4),
    P("M1 et M2 ajoutent une attention au-dessus. D'abord la phrase, ensuite chaque émotion qui la relit.", 16, NAVY, True, after=0)],
    fill=LIGHT, anchor="m", margins=(0.25, 0.1, 0.25, 0.1)))

# ----------------------------------------------------------------------------- 4. Objectives and hypotheses
s = new("Objectifs et hypothèses", notes=(
    "À gauche, quatre objectifs : concevoir, tester, comparer NRC et SenticNet, lire classe par classe. "
    "À droite, quatre hypothèses. H1 : M1 bat le modèle plat. H2 : M2 bat M1. "
    "H3 : un lexique améliore encore. H4 : un poids par émotion fait mieux. "
    "Les résultats ne vont pas toutes les confirmer."))
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

# ----------------------------------------------------------------------------- 02 État de l'art
divider(1, "D'abord les chiffres déjà publiés. Puis les idées que je reprends.")

s = new("Deux repères publiés, deux protocoles", notes=(
    "BERT-base : environ 0,46. Seq2Emo : 0,473. "
    "Ce n'est pas le même protocole. Je ne dis pas que je les dépasse. "
    "Ma comparaison, c'est l'ablation, avec le même entraînement."))
s.add(Table(0.6, 1.6, [4.3, 4.5, 3.3], [
    ["Travail", "Protocole", "F1-macro"],
    ["BERT-base, Demszky et al., 2020", "leur partage, 28 classes", "≈ 0,46"],
    ["Seq2Emo, Ke et al., 2021", "biLSTM, GloVe, ELMo, 5 plis", "0,473"],
], size=16, min_row=0.7))
s.add(Box(0.6, 4.15, 12.1, 2.55, [
    P("Comment lire ces chiffres", 20, NAVY, True, after=8),
    P("Le partage, le modèle et la façon de voter ne sont pas ceux de ce mémoire. Ces deux lignes ne servent pas de classement.", 17, after=8),
    P("La comparaison que l'expérimentation défend est interne : le même entraînement, et les modules ajoutés un par un.", 17, after=0)],
    fill=LNAVY, margins=(0.28, 0.16, 0.28, 0.14)))

s = new("Trois idées reprises, et ce qu'elles laissent ouvert", notes=(
    "Trois lignes. Le lien entre étiquettes : Seq2Emo le fait autrement. Ici, c'est M2. "
    "La hiérarchie : le HAN. M1 reprend l'idée, pas tout le réseau. "
    "Les lexiques : NRC et SenticNet. M3 les passe par une porte. "
    "En bas : ils ne sont pas mesurés ensemble, ici, avec le même budget sur deux modèles."))
s.add(Table(0.55, 1.55, [3.1, 4.5, 4.6], [
    ["Idée", "Déjà publié", "Dans HAKE-MER"],
    ["Lien entre étiquettes", "Chaînes de classifieurs. Seq2Emo le fait avec un biLSTM.", "M2 : une attention entre émotions. Ce n'est pas leur modèle."],
    ["Hiérarchie du texte", "HAN : mots, puis phrases, puis document (Yang et al., 2016).", "M1 reprend l'idée, pas le réseau récurrent complet."],
    ["Lexiques", "NRC, un mot vers une émotion (2013). SenticNet, une intensité signée (2018).", "M3 : un prior de document, passé par une porte commune."],
], size=15, min_row=1.05))
s.add(Box(0.55, 6.0, 12.2, 0.75, [
    P("Ces idées ne sont pas mesurées ensemble, ici, avec le même budget sur deux encodeurs.", 16, NAVY, after=0)],
    fill=LNAVY, anchor="m", margins=(0.25, 0.08, 0.25, 0.08)))

# ----------------------------------------------------------------------------- 03 Approche proposée
divider(2, "La carte, puis chaque étape, puis ce que j'ai construit.")

# ----------------------------------------------------------------------------- Architecture
s = new("Architecture HAKE-MER : quatre modules", notes=(
    "En haut, le chemin : le texte, l'encodeur, M1, M2, M4, et 28 réponses. "
    "En bas, M3 : NRC ou SenticNet, avec une porte g qui vaut 0 au départ. "
    "Si on éteint M3, on revient à M1+M2."))
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

# ----------------------------------------------------------------------------- Step 1
s = new("Étape 1 : des tokens à un vecteur par phrase", notes=(
    "Découpage : quatre phrases. Encodeur : un état par token. "
    "M1 : une attention, un vecteur par phrase. "
    "Step 0 lit encore 128 tokens d'un bloc. Les tokens gardés peuvent différer. "
    "Ici, les émotions ne se comparent pas encore."))
flow = [
    ("1", "Découpage", "Jusqu'à 4 phrases, 32 tokens"),
    ("2", "Encodeur", "Un état par token, pas un vecteur fixe"),
    ("3", "M1", "Attention additive sur ces tokens"),
    ("4", "Sortie", "Un vecteur par phrase"),
]
fw, fgap = 2.85, 0.32
for i, (num, title, line) in enumerate(flow):
    x = 0.55 + i * (fw + fgap)
    s.add(Box(x, 1.6, fw, 2.15, [
        P(num, 14, GREEN, True, align="c", after=2),
        P(title, 20, NAVY, True, align="c", after=6),
        P(line, 14, INK, align="c", after=0)],
        fill=LGREEN if i == 3 else LIGHT, line=GREEN if i == 3 else LINE, line_w=1.5, shape="round", anchor="m",
        margins=(0.14, 0.1, 0.14, 0.1)))
    if i < len(flow) - 1:
        s.add(Arrow(x + fw + 0.03, 2.5, fgap - 0.06, 0.26))
s.add(Box(0.55, 4.05, 12.2, 1.2, [
    P("Step 0 lit encore 128 tokens d'un bloc.", 18, NAVY, True, after=4),
    P("Le nombre annoncé est le même. Le découpage peut garder d'autres tokens. Il n'y a pas d'encodeur récurrent en plus.", 16, INK, after=0)],
    fill=LORANGE, anchor="m", margins=(0.28, 0.12, 0.28, 0.1)))
s.add(Box(0.55, 5.45, 12.2, 1.15, [
    P("Les émotions ne se comparent pas encore, et la décision n'a pas de tête par émotion.", 18, WHITE, True, align="c", after=0)],
    fill=NAVY, shape="round", anchor="m"))

# ----------------------------------------------------------------------------- Step 2
s = new("Étape 2 : chaque émotion lit, puis les émotions se comparent", notes=(
    "A : chaque émotion regarde les phrases. Joie et tristesse ne lisent pas la même chose. "
    "B : les 28 vecteurs se comparent. La tête lit la sortie de cet encodeur. "
    "En bas : de M1 à M1+M2, la tête change aussi. Le gain est celui de la pile."))
s.add(Box(0.55, 1.55, 1.15, 2.05, [P("A", 28, WHITE, True, align="c", after=0)], fill=(0x5B, 0x7B, 0xA6), anchor="m"))
s.add(Box(1.7, 1.55, 11.05, 2.05, [
    P("Chaque émotion regarde les phrases", 22, NAVY, True, after=6),
    P("Attention croisée. Joie et tristesse ne retiennent pas les mêmes vecteurs de phrases.", 16, INK, after=0)],
    fill=LNAVY, anchor="m", margins=(0.28, 0.14, 0.28, 0.14)))
s.add(Box(0.55, 3.8, 1.15, 1.7, [P("B", 28, WHITE, True, align="c", after=0)], fill=NAVY, anchor="m"))
s.add(Box(1.7, 3.8, 11.05, 1.7, [
    P("Puis les 28 vecteurs se comparent", 22, NAVY, True, after=6),
    P("La tête lit la sortie de cet encodeur, pas le vecteur brut de l'attention.", 16, INK, after=0)],
    fill=LIGHT, anchor="m", margins=(0.28, 0.12, 0.28, 0.12)))
s.add(Box(0.55, 5.7, 12.2, 0.95, [
    P("De M1 à M1+M2, la tête change aussi : le gain mesuré est celui de la pile. Le contrôle enlève seulement l'encodeur.", 16, NAVY, after=0)],
    fill=LORANGE, anchor="m", margins=(0.25, 0.1, 0.25, 0.1)))

# ----------------------------------------------------------------------------- Step 3
s = new("Étape 3 : un prior lexical, puis un poids par émotion", notes=(
    "M3 : NRC ou SenticNet. La porte g vaut 0 au départ. Le prior est le même pour toutes les émotions. "
    "M4 : un poids par émotion, juste avant la décision."))
s.add(Box(0.55, 1.5, 1.3, 0.4, [P("M3", 16, GREEN, True, after=0)], margins=(0, 0, 0, 0)))
s.add(Box(0.55, 1.95, 2.45, 1.45, [
    P("NRC", 16, NAVY, True, align="c", after=3),
    P("8 parts de Plutchik", 13, INK, align="c", after=0)],
    fill=LGREEN, line=GREEN, line_w=1.25, shape="round", anchor="m"))
s.add(Box(3.05, 2.4, 0.55, 0.4, [P("ou", 14, GRAY, True, align="c", after=0)], anchor="m", margins=(0, 0, 0, 0)))
s.add(Box(3.65, 1.95, 2.55, 1.45, [
    P("SenticNet", 16, NAVY, True, align="c", after=3),
    P("5 descripteurs", 13, INK, align="c", after=0)],
    fill=LGREEN, line=GREEN, line_w=1.25, shape="round", anchor="m"))
s.add(Arrow(6.25, 2.5, 0.38, 0.24))
s.add(Box(6.68, 1.95, 2.7, 1.45, [
    P("Porte", 16, NAVY, True, align="c", after=3),
    P("g = 0 au départ", 13, INK, align="c", after=0)],
    fill=WHITE, line=GREEN, line_w=1.5, shape="round", anchor="m"))
s.add(Arrow(9.43, 2.5, 0.38, 0.24))
s.add(Box(9.86, 1.95, 2.85, 1.45, [
    P("Requêtes", 16, NAVY, True, align="c", after=3),
    P("le même prior pour toutes", 13, INK, align="c", after=0)],
    fill=LNAVY, line=(0x5B, 0x7B, 0xA6), line_w=1.25, shape="round", anchor="m"))
s.add(Box(0.55, 3.65, 12.2, 2.9, [
    P("M4", 16, VIOLET, True, after=6),
    P("Un poids par émotion, après cet échange.", 26, NAVY, True, after=8),
    P("Il dépend du vecteur de l'émotion et de la moyenne des émotions, et il s'applique juste avant la décision.", 16, INK, after=6),
    P("Si l'encodeur de M2 est éteint, ce poids porte sur l'autre vecteur.", 16, GRAY, after=0)],
    fill=LILAC, anchor="m", margins=(0.35, 0.18, 0.35, 0.16)))

# ----------------------------------------------------------------------------- Contribution
s = new("Ce que j'apporte dans cette architecture", notes=(
    "M1 : l'idée du HAN, pas le réseau complet. "
    "M2 : une lecture par émotion, puis un échange. "
    "M3 : les lexiques existent déjà. La porte, et la table vers les 28 classes, c'est ce mémoire. "
    "M4 : un poids par émotion."))
built = [
    ("M1", GREEN, LGREEN, "L'idée vient du HAN. Le réseau récurrent n'est pas repris : l'attention porte sur les états du modèle."),
    ("M2", (0x5B, 0x7B, 0xA6), LNAVY, "Une lecture par émotion, puis un échange entre les 28 vecteurs. La tête est par émotion."),
    ("M3", GREEN, LGREEN, "NRC et SenticNet sont publiés. La porte, et la table vers les 28 classes, sont de ce mémoire."),
    ("M4", VIOLET, LILAC, "Un poids par émotion, calculé juste avant la décision."),
]
for i, (name, accent, fill, line) in enumerate(built):
    y = 1.5 + i * 1.22
    s.add(Box(0.55, y, 1.45, 1.08, [P(name, 22, accent, True, align="c", after=0)], fill=fill, anchor="m"))
    s.add(Box(2.15, y, 10.6, 1.08, [P(line, 16, INK, after=0)], anchor="m", margins=(0.2, 0.08, 0.15, 0.08)))

# ----------------------------------------------------------------------------- 04 Expérimentation
divider(3, "Même protocole pour tous. Puis l'ablation : DistilBERT, ensuite RoBERTa.")

# ----------------------------------------------------------------------------- Protocol
s = new("Protocole d'évaluation", notes=(
    "GoEmotions, le découpage des auteurs. DistilBERT et RoBERTa. "
    "J'ajoute les modules un par un. Le score est le F1-macro au test. Je garde la meilleure époque en validation. "
    "Trois essais. À droite, les réglages sont les mêmes pour tous."))
s.add(Box(0.6, 1.55, 6.3, 5.2, [
    P("**Dataset** : GoEmotions, 28 classes, découpage des auteurs", 18, bullet=True, after=12),
    P("**Modèles** : DistilBERT-base et RoBERTa-base", 18, bullet=True, after=12),
    P("**Ablation** : Step 0, puis +M1, +M1+M2, +M3, +M4", 18, bullet=True, after=12),
    P("**Score** : F1-macro au test. L'époque retenue est la meilleure en validation, pour chaque graine.", 18, bullet=True, after=12),
    P("**Trois essais** : graines 42, 123 et 456. Moyenne et écart-type, pas un test statistique.", 18, bullet=True, after=0)],
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

# ----------------------------------------------------------------------------- Ablation DistilBERT
s = new("Étude d'ablation : DistilBERT-base", notes=(
    "M1 seul recule : 0,457 contre 0,486. "
    "M1+M2 monte à 0,505. "
    "M3 et M4 ne dépassent pas M1+M2."))
s.add(*fpic(FIG / "ablation_distilbert.png", 0.6, 1.5, 7.7, "F1-macro au test, DistilBERT, moyenne ± écart-type sur 3 essais. L'axe commence à 0,42."))
s.add(*card(8.6, 1.6, 4.1, 1.55, "M1 seul recule", [f"{fr(mean(ROWS['m1'][0]))} contre {fr(mean(ROWS['step0'][0]))} pour le flat"], accent=ORANGE, size=17))
s.add(*card(8.6, 3.3, 4.1, 1.55, "M1+M2 progresse", [f"{fr(mean(ROWS['m12'][0]))}, soit +{fr(gain_db)} sur Step 0"], accent=GREEN, size=17))
s.add(*card(8.6, 5.0, 4.1, 1.75, "M3 et M4 sans gain", ["0,499 à 0,504, aucun ne dépasse M1+M2"], accent=(0x5B, 0x7B, 0xA6), size=17))

# ----------------------------------------------------------------------------- 9. Results RoBERTa
s = new("La même ablation sur RoBERTa-base", notes=(
    "Même forme. M1 seul ne bat le modèle plat sur aucun des deux. "
    "M1+M2 monte des deux côtés. RoBERTa est un peu devant, d'environ 0,02."))
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
s = new("Ablation interne de M2 : deux contrôles", notes=(
    "Sans le lien entre émotions, le score baisse. "
    "Sans le regard vers les phrases, il bouge peu. "
    "Le gain vient du lien entre les émotions."))
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
    "Le gain n'est pas sur toutes les émotions. "
    "Il aide surtout desire, annoyance et curiosity. "
    "nervousness et embarrassment baissent. "
    "L'écart entre fréquent et rare se réduit. Le gain ne vient pas des classes les plus rares."))
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
    "Les dix erreurs les plus fréquentes passent toutes par neutral. "
    "Surtout approval, annoyance et disapproval. "
    "Ce n'est pas deux émotions proches. C'est la limite entre une émotion faible et pas d'émotion."))
s.add(*fpic(FIG / "errors.png", 0.6, 1.6, 8.1, "Dix motifs d'erreur les plus fréquents, M1+M2, DistilBERT, 3 graines cumulées (test)."))
s.add(*card(9.0, 1.6, 3.7, 2.4, "Constat", ["Les dix motifs impliquent tous neutral", "Approval, annoyance, disapproval vs neutral dominent"], accent=ORANGE, size=15))
s.add(*card(9.0, 4.2, 3.7, 2.4, "Lecture", ["Limite floue entre une émotion faible et pas d'émotion", "Piste : mieux séparer neutral"], accent=GREEN, size=15))

# ----------------------------------------------------------------------------- 13. Hypotheses verdict
s = new("Bilan des hypothèses : une seule est soutenue", notes=(
    "H1 n'est pas soutenue : M1 baisse. "
    "H2 est soutenue : M1+M2 monte. "
    "H3 et H4 : pas de gain stable. "
    "Je retiens M1+M2. NRC avec M4 est un peu plus haut, mais seulement sur DistilBERT."))
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
divider(4, "Je replace les scores. Puis je dis ce que je retiens.")

# ----------------------------------------------------------------------------- Discussion and limits
s = new("Discussion : comparaison et limites", notes=(
    "0,46 et 0,473 sont des repères. Ce n'est pas le même protocole. "
    "À droite, les limites : une seule dataset, trois essais, les mêmes réglages. "
    "NRC avec M4 n'est pas testé sur RoBERTa."))
s.add(Box(0.6, 1.5, 5.9, 0.45, [P("Comparaison avec la littérature (F1-macro)", 18, NAVY, True, after=0)], margins=(0, 0, 0, 0)))
s.add(Table(0.6, 2.05, [3.3, 2.6], [
    ["Travail", "F1-macro"],
    ["BERT-base, GoEmotions (2020)", "≈ 0,46"],
    ["Seq2Emo (2021)", "0,473"],
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
    "Je retiens M1+M2, sur les deux modèles. Le gain est celui de la pile. "
    "M3 et M4 n'ajoutent pas un gain stable. "
    "Pour la suite : les classes rares, un lexique par émotion, d'autres datasets, d'autres modèles au même budget."))
s.add(Box(0.55, 1.55, 4.3, 3.15, [
    P("On retient", 16, MINT, True, after=4),
    P("M1+M2", 36, WHITE, True, after=6),
    P("sur DistilBERT et sur RoBERTa", 16, WHITE, after=0)],
    fill=NAVY, anchor="m", margins=(0.25, 0.15, 0.25, 0.15)))
s.add(Box(5.05, 1.55, 7.7, 1.5, [
    P("Le gain mesuré est celui de la pile. Enlever l'encodeur entre émotions le fait baisser.", 16, INK, after=0)],
    fill=LGREEN, anchor="m", margins=(0.22, 0.12, 0.22, 0.12)))
s.add(Box(5.05, 3.2, 7.7, 1.5, [
    P("M3 et M4 n'ajoutent pas un gain stable. Le F1-macro ne dit pas quelles émotions bougent.", 16, INK, after=0)],
    fill=LORANGE, anchor="m", margins=(0.22, 0.12, 0.22, 0.12)))
persp = [
    ("01", "Classes rares, et un seuil par émotion"),
    ("02", "Un lexique lié à chaque émotion"),
    ("03", "D'autres datasets, comme SemEval"),
    ("04", "D'autres modèles, au même budget"),
]
for i, (num, line) in enumerate(persp):
    x = 0.55 + i * 3.15
    s.add(Box(x, 4.95, 3.0, 1.65, [
        P(num, 14, ORANGE, True, after=4),
        P(line, 15, NAVY, True, after=0)],
        fill=LIGHT, anchor="m", margins=(0.16, 0.12, 0.16, 0.12)))

# ----------------------------------------------------------------------------- 16. Questions
s = new(footer=False, notes=(
    "Merci. Je suis prête pour vos questions.\n"
    "Si on me demande pourquoi M1 baisse : le modèle plat voit le texte d'un bloc. M1 le coupe, sans encore relier les émotions. M2 rattrape.\n"
    "Si on me demande pourquoi les lexiques aident peu : le lexique est le même pour toutes les émotions, et il entre par une seule porte.\n"
    "Si on me demande si c'est prouvé : trois essais, pas de test statistique. Je ne dis pas que la différence est prouvée.\n"
    "Si on me demande pourquoi le F1-macro : il donne le même poids à chaque émotion, même à grief, qui n'a que 6 exemples.\n"
    "Si on me parle d'un score de 0,56 en 2025 : autre partage, autres traits. Je ne le compare pas."))
s.add(Box(0.0, 0.0, 13.333, 7.5, fill=NAVY))
s.add(Box(0.6, 2.4, 12.1, 1.4, [P("Merci de votre attention", 44, WHITE, True, align="c", after=0)], anchor="m", margins=(0, 0, 0, 0)))
s.add(Box(0.6, 3.9, 12.1, 0.9, [P("Questions et discussion", 26, (0xB4, 0xE0, 0xC8), align="c", after=0)], anchor="m", margins=(0, 0, 0, 0)))
s.add(Box(0.6, 5.7, 12.1, 0.9, [P("Mariem El WEDANI  |  Encadrant : Sahbi BAHROUN  |  Institut Supérieur d'Informatique, 2025 - 2026", 14, (0xC9, 0xD1, 0xDC), align="c", after=0)],
          anchor="m", margins=(0, 0, 0, 0)))

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
