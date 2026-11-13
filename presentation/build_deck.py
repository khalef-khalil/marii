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
    "Bonjour. Je m'appelle Mariem El Wedani, et je présente mon mémoire de Master, encadré par M. Sahbi Bahroun. "
    "Le problème est simple à dire. Un même texte peut exprimer plusieurs émotions en même temps. "
    "Le modèle que j'ai construit pour cela s'appelle HAKE-MER."))
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
    "Je vais suivre cinq parties. "
    "Le contexte pose la tâche, puis les embeddings, puis pourquoi l'attention compte. "
    "L'état de l'art dit ce qui existe déjà, et ce que ces travaux ne règlent pas. "
    "L'approche proposée détaille HAKE-MER étape par étape, et ce que j'y apporte. "
    "L'expérimentation fixe le protocole, puis l'étude d'ablation. "
    "Je termine par ce que je retiens, et par ce que je ne peux pas encore affirmer."))
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
divider(0, "Avant le modèle, je pose la tâche, puis comment un texte devient un vecteur, et pourquoi l'attention sert à ne pas tout mélanger. C'est de là que viennent les hypothèses.")

# ----------------------------------------------------------------------------- Context and data
s = new("Contexte : un texte, plusieurs émotions", notes=(
    "Un commentaire peut porter plusieurs émotions à la fois, donc la tâche n'est pas de choisir une seule étiquette. "
    "On a travaillé sur la dataset GoEmotions. Elle se compose de commentaires Reddit, annotés avec 27 émotions plus neutral, ce qui fait 28 classes. "
    "Le découpage est celui des auteurs, avec plus de 40 000 exemples pour l'entraînement, et un peu plus de 5 000 pour la validation comme pour le test. "
    "Ce qui change un peu l'image, c'est que autour de 85 % des commentaires de test n'ont en fait qu'une seule étiquette. "
    "Les classes sont aussi très inégales. neutral en a près de 1 800, et grief seulement 6. Ce déséquilibre va compter dans les résultats."))
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
    "Sur cette dataset, trois limites reviennent. "
    "D'abord, les émotions sont souvent décidées seules, sans lien entre elles, alors que la joie et la tristesse vont rarement ensemble. "
    "Ensuite, la structure du texte se perd, parce que tout le commentaire est mis dans un seul vecteur et que les phrases se mélangent. "
    "Enfin, les émotions rares ont très peu d'exemples. grief n'en a que 6 au test, neutral près de 1 800, et avec si peu d'exemples le modèle apprend mal. "
    "D'où la question du bas, qui demande si on fait mieux en séparant les phrases, en reliant les émotions, et en ajoutant un lexique."))
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
    "Avant l'architecture, il faut dire comment un texte devient des nombres. "
    "Les premières méthodes donnent un vecteur fixe à chaque mot. Word2Vec et GloVe sont de cette famille. "
    "Le vecteur ne change pas avec la phrase, donc l'ironie ou le contexte proche restent difficiles. "
    "Ensuite sont venus les embeddings contextuels. ELMo, puis BERT. Le même mot n'a plus le même vecteur selon la phrase. "
    "GoEmotions a été évalué en affinant BERT. Ce mémoire part de cette famille, avec DistilBERT et RoBERTa. "
    "Un embedding, même contextuel, reste un vecteur. Il ne dit pas encore quelle phrase porte quelle émotion."))
s.add(*card(0.6, 1.55, 6.05, 5.15, "Un vecteur fixe par mot", [
    "Word2Vec et GloVe : le mot a le même vecteur dans toutes les phrases",
    "Utile comme point de départ, et encore présent dans des modèles plus anciens",
    "Le contexte proche, ou l'ironie, ne déplace pas ce vecteur",
    "GloVe reste utilisé plus tard, par exemple dans Seq2Emo, à côté d'ELMo"], accent=ORANGE, size=16))
s.add(*card(6.85, 1.55, 5.85, 5.15, "Un vecteur qui dépend de la phrase", [
    "ELMo, puis BERT : le vecteur change avec les mots autour",
    "Les auteurs de GoEmotions affinent BERT-base",
    "Ce mémoire part de DistilBERT et de RoBERTa",
    "Le modèle plat, Step 0, est cet affinage, avec une sortie par émotion"], accent=GREEN, size=16))

# ----------------------------------------------------------------------------- Attention
s = new("Pourquoi le mécanisme d'attention compte", notes=(
    "Un embedding de phrase est encore une moyenne, si on ne choisit pas quels mots comptent. "
    "L'attention donne un poids à chaque token selon la prédiction. Le modèle peut s'appuyer sur les mots qui portent l'émotion. "
    "Dans BERT et RoBERTa, cette idée est déjà à l'intérieur : l'auto-attention rend le vecteur contextuel. "
    "Pour plusieurs émotions dans un même commentaire, ça ne suffit pas. Une phrase de colère et une phrase de gratitude peuvent se mélanger dans un seul vecteur. "
    "HAKE-MER ajoute donc une attention au-dessus de l'encodeur. D'abord sur les mots d'une phrase, ensuite entre les émotions et les phrases. "
    "C'est ce que les modules M1 et M2 vont détailler."))
s.add(*card(0.6, 1.55, 6.05, 2.35, "Sans poids sur les tokens", [
    "Tout le commentaire finit dans un seul vecteur",
    "Deux phrases d'émotions différentes se mélangent"], accent=ORANGE, size=16))
s.add(*card(6.85, 1.55, 5.85, 2.35, "Avec une attention", [
    "Chaque token reçoit un poids pour la prédiction",
    "Le modèle peut s'appuyer sur les mots qui portent l'émotion"], accent=GREEN, size=16))
s.add(Box(0.6, 4.15, 12.1, 2.55, [
    P("Deux niveaux, pas le même rôle", 20, NAVY, True, after=8),
    P("**Dans l'encodeur.** L'auto-attention de BERT et de RoBERTa rend le vecteur dépendant de la phrase. Elle est déjà là avant HAKE-MER.", 17, after=8),
    P("**Au-dessus de l'encodeur.** M1 résume les tokens d'une phrase. M2 laisse chaque émotion relire ces phrases, puis les émotions se comparent.", 17, after=0)],
    fill=LIGHT, margins=(0.28, 0.16, 0.28, 0.14)))

# ----------------------------------------------------------------------------- 4. Objectives and hypotheses
s = new("Objectifs et hypothèses", notes=(
    "Pour répondre, j'ai fixé quatre objectifs. Concevoir HAKE-MER avec quatre modules qu'on ajoute un par un, "
    "tester chaque ajout dans les mêmes conditions sur deux modèles, comparer NRC et SenticNet, "
    "et ne pas s'arrêter au F1-macro. "
    "Chaque module a aussi son hypothèse. H1 dit que M1 bat le modèle plat, H2 que M2 bat M1 seul, "
    "H3 qu'un lexique améliore encore, et H4 qu'un poids par émotion fait mieux. "
    "Ce sont ces quatre idées que les résultats vont trancher, et elles ne seront pas toutes confirmées."))
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
divider(1, "Ces hypothèses ne partent pas de zéro. Je situe d'abord les chiffres publiés, puis les idées que HAKE-MER reprend, et ce qu'elles ne tranchent pas.")

s = new("Deux repères publiés, deux protocoles", notes=(
    "Sur GoEmotions, les auteurs rapportent 0,46 de F1-macro pour BERT-base. "
    "C'est leur partage, leur filtrage, leurs réglages. "
    "Seq2Emo rapporte 0,473, avec un biLSTM, GloVe et ELMo, et un vote sur cinq plis. "
    "Je les cite comme des repères. Je ne dis pas que nos scores les dépassent, parce que le protocole n'est pas le même. "
    "Un travail de 2025 annonce 0,56 sur un autre partage, avec des traits en plus. Je ne le mets pas dans le tableau. "
    "La comparaison que je défends est l'ablation, plus loin, avec le même budget pour chaque module."))
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
    "Trois idées existent déjà. "
    "Relier les étiquettes, pour ne pas décider chaque émotion toute seule. Seq2Emo va dans ce sens, avec un autre modèle. "
    "Garder la structure du texte. Le HAN de Yang lit d'abord les mots, puis les phrases. "
    "Ajouter un lexique. NRC relie un mot à une émotion, SenticNet donne une intensité signée. "
    "Aucune de ces idées n'est reprise telle quelle. M1 n'est pas le HAN complet. M2 n'est pas Seq2Emo. "
    "Et personne n'a mesuré ces ajouts, ici, avec le même budget sur DistilBERT et sur RoBERTa. C'est le rôle de l'approche proposée."))
cw = 3.9
s.add(*card(0.6, 1.55, cw, 3.7, "Lien entre étiquettes", [
    "Sorties séparées, ou chaînes de classifieurs",
    "Seq2Emo modélise des liens, avec un biLSTM",
    "M2 reprend la question, par une attention entre émotions"], accent=ORANGE, size=16))
s.add(*card(0.6 + cw + 0.2, 1.55, cw, 3.7, "Hiérarchie du texte", [
    "HAN : mots, puis phrases, puis document (Yang et al., 2016)",
    "Répond à un vecteur unique qui mélange les phrases",
    "M1 reprend l'idée, pas le réseau récurrent complet"], accent=GREEN, size=16))
s.add(*card(0.6 + 2 * (cw + 0.2), 1.55, cw, 3.7, "Lexiques externes", [
    "NRC : un mot vers une émotion (Mohammad et Turney, 2013)",
    "SenticNet : une intensité signée (Cambria et al., 2018)",
    "M3 les injecte comme un prior, par une porte commune"], accent=(0x5B, 0x7B, 0xA6), size=16))
s.add(Box(0.6, 5.5, 12.1, 1.2, [
    P("Ces idées ne sont pas encore mesurées ensemble, avec le même budget, sur deux encodeurs. C'est ce que l'approche proposée organise.",
      17, NAVY, after=0)],
    fill=LNAVY, anchor="m", margins=(0.28, 0.1, 0.28, 0.1)))

# ----------------------------------------------------------------------------- 03 Approche proposée
divider(2, "Je peux maintenant montrer comment HAKE-MER est construit. D'abord la carte, puis chaque étape, puis ce que j'apporte dans cette architecture.")

# ----------------------------------------------------------------------------- Architecture
s = new("Architecture HAKE-MER : quatre modules", notes=(
    "Le commentaire est d'abord coupé en quatre phrases courtes, puis lu par DistilBERT ou RoBERTa. "
    "M1 résume les mots de chaque phrase, pour ne pas tout mélanger. "
    "M2 fait ensuite deux choses. Chaque émotion regarde les phrases, puis les 28 émotions se comparent entre elles. "
    "M4 donne un poids à chaque émotion, et on sort 28 réponses. "
    "M3, en dessous, lit NRC ou SenticNet et l'ajoute avec une porte g. Au départ g vaut 0, donc le lexique n'entre pas tout de suite. "
    "Si on éteint M3, on retombe simplement sur M1+M2. "
    "Le lien avec les limites est direct. M1 répond à la structure du texte perdue, M2 aux émotions décidées seules, et M3 aux émotions rares."))
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
    "Le commentaire est coupé en quatre phrases au plus, de 32 tokens. "
    "Step 0, lui, lit 128 tokens d'un seul bloc. Le nombre annoncé est le même, mais le découpage peut ne pas garder les mêmes tokens. "
    "DistilBERT ou RoBERTa lit ensuite chaque phrase. On obtient un état par token, pas un vecteur de mot fixe. "
    "M1 résume les tokens de la phrase par une attention additive. Une phrase, un vecteur. "
    "Je n'ajoute pas un deuxième encodeur récurrent. Le modèle pré-entraîné a déjà lu les tokens. "
    "Cette étape répond à la structure du texte perdue. Elle ne relie pas encore les émotions."))
s.add(*card(0.6, 1.55, 6.05, 5.15, "Ce que fait cette étape", [
    "Jusqu'à 4 phrases, 32 tokens chacune",
    "L'encodeur produit un état par token",
    "M1 : attention additive, un vecteur par phrase",
    "Pas d'encodeur récurrent en plus du modèle pré-entraîné"], accent=GREEN, size=17))
s.add(*card(6.85, 1.55, 5.85, 5.15, "Ce que cette étape ne fait pas", [
    "Step 0 reste un bloc de 128 tokens",
    "Le budget annoncé est le même, les tokens gardés peuvent différer",
    "Les émotions ne se comparent pas encore",
    "La décision, à ce stade, n'a pas de tête par émotion"], accent=ORANGE, size=17))

# ----------------------------------------------------------------------------- Step 2
s = new("Étape 2 : chaque émotion lit, puis les émotions se comparent", notes=(
    "M2 commence par une attention croisée. Chaque émotion regarde les vecteurs de phrases, et elle ne retient pas la même chose. "
    "Les 28 vecteurs passent ensuite dans un petit encodeur, qui les laisse se comparer. La joie et la tristesse ne sont plus deux sorties isolées. "
    "La décision se fait par émotion, sur la sortie de cet encodeur. "
    "Il faut le dire tout de suite. En passant de M1 seul à M1+M2, la tête change aussi. Le gain qu'on mesurera est celui de la pile, pas d'une seule pièce. "
    "Le contrôle, dans l'ablation, enlève l'encodeur entre émotions sans enlever le reste."))
s.add(*card(0.6, 1.55, 6.05, 5.15, "Deux mouvements", [
    "Chaque émotion regarde les phrases : attention croisée",
    "Les 28 vecteurs se comparent dans un encodeur",
    "La sortie utilisée est celle de cet encodeur, pas le vecteur brut",
    "Une tête par émotion, au lieu d'une tête partagée"], accent=GREEN, size=17))
s.add(*card(6.85, 1.55, 5.85, 5.15, "Ce qu'on pourra en conclure", [
    "Cette étape répond aux émotions décidées seules",
    "De M1 à M1+M2, la tête change en même temps",
    "Le gain mesuré est celui de l'ensemble",
    "Le contrôle enlève seulement l'encodeur entre émotions"], accent=(0x5B, 0x7B, 0xA6), size=17))

# ----------------------------------------------------------------------------- Step 3
s = new("Étape 3 : un prior lexical, puis un poids par émotion", notes=(
    "M3 et M4 arrivent après, et ils ne changent pas le découpage. "
    "M3 lit un lexique et construit un vecteur pour tout le commentaire. "
    "Avec NRC, ce sont huit parts d'émotions de Plutchik. Avec SenticNet, cinq descripteurs d'intensité signée. "
    "Une projection apprise et une porte g, nulle au départ, ajoutent ce vecteur aux requêtes de toutes les émotions. "
    "Si g reste près de zéro, le lexique n'entre presque pas. "
    "M4 calcule ensuite un poids par émotion, à partir de son vecteur et de la moyenne, juste avant la décision. "
    "Le même branchement sert les deux lexiques. Seul le contenu du vecteur change."))
s.add(*card(0.6, 1.55, 6.05, 5.15, "M3 : le prior", [
    "NRC : huit parts, sur les émotions de Plutchik",
    "SenticNet : moyenne, max, min, couverture, tokens distincts",
    "Projection apprise, porte g nulle au départ",
    "Le prior est ajouté aux requêtes, le même pour toutes les émotions"], accent=GREEN, size=16))
s.add(*card(6.85, 1.55, 5.85, 5.15, "M4 : le poids", [
    "Un poids par émotion, après M2",
    "Il dépend du vecteur de l'émotion et de la moyenne",
    "Il s'applique juste avant la décision",
    "Si l'encodeur de M2 est éteint, le poids porte sur l'autre vecteur"], accent=VIOLET, size=16))

# ----------------------------------------------------------------------------- Contribution
s = new("Ce que j'apporte dans cette architecture", notes=(
    "L'encodeur, l'idée du HAN et les deux lexiques existent déjà. Je ne les réinvente pas. "
    "Ce que je construis, c'est l'enchaînement. Des phrases résumées par attention, une lecture par émotion, un prior lexical passé par une porte, puis un poids par émotion. "
    "La correspondance entre les 28 classes de GoEmotions et les catégories NRC est définie pour ce mémoire. Elle n'est pas dans le lexique publié. "
    "Chaque module peut être éteint. L'étude d'ablation sert à dire lequel mérite d'être gardé, au lieu de supposer que les quatre aident."))
s.add(*card(0.6, 1.55, 6.05, 5.15, "Déjà disponible", [
    "DistilBERT et RoBERTa, et leur affinage",
    "L'idée d'une attention mots puis phrases",
    "NRC et SenticNet, comme lexiques publiés",
    "L'auto-attention, déjà dans l'encodeur"], accent=(0x5B, 0x7B, 0xA6), size=17))
s.add(*card(6.85, 1.55, 5.85, 5.15, "Construit pour ce mémoire", [
    "Quatre modules qu'on allume un par un",
    "M1 sur les états du modèle, sans HAN complet",
    "M2 : lecture par émotion, puis échange",
    "M3 par une porte, et la table vers les 28 classes"], accent=GREEN, size=17))

# ----------------------------------------------------------------------------- 04 Expérimentation
divider(3, "Le protocole est le même pour tous les modules. L'étude d'ablation dit ensuite ce que chacun change, d'abord sur DistilBERT, puis sur RoBERTa.")

# ----------------------------------------------------------------------------- Protocol
s = new("Protocole d'évaluation", notes=(
    "Pour que la comparaison soit juste, tous les tests ont le même entraînement, sur GoEmotions, avec le découpage des auteurs. "
    "Les deux modèles sont DistilBERT et RoBERTa. On part du modèle plat, Step 0, puis on ajoute les modules un par un. C'est l'étude d'ablation. "
    "Le score que je regarde est le F1-macro au test. Pour chaque graine, je garde l'époque qui était la meilleure en validation. "
    "Chaque configuration est lancée trois fois, avec trois graines fixes. L'écart-type n'est pas un intervalle de confiance, et je n'ai pas fait de test statistique. "
    "Quatre époques, des lots de 16, un taux très faible, et un seuil à 0,5. "
    "Step 0 lit le texte d'un bloc. À partir de M1, on le coupe en quatre phrases courtes."))
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
    "Ce qu'on voit ici, c'est que le premier module, à lui seul, ne suffit pas. "
    "Le score baisse même par rapport au modèle de départ, de 0,486 à 0,457. "
    "En revanche, quand j'ajoute M2, le résultat change, et on arrive à 0,505. "
    "C'est un gain d'environ 0,02 sur Step 0, et c'est la seule étape qui dépasse clairement le modèle de départ. "
    "M3 et M4 restent au même niveau. Ils ne font pas mieux."))
s.add(*fpic(FIG / "ablation_distilbert.png", 0.6, 1.5, 7.7, "F1-macro au test, DistilBERT, moyenne ± écart-type sur 3 essais. L'axe commence à 0,42."))
s.add(*card(8.6, 1.6, 4.1, 1.55, "M1 seul recule", [f"{fr(mean(ROWS['m1'][0]))} contre {fr(mean(ROWS['step0'][0]))} pour le flat"], accent=ORANGE, size=17))
s.add(*card(8.6, 3.3, 4.1, 1.55, "M1+M2 progresse", [f"{fr(mean(ROWS['m12'][0]))}, soit +{fr(gain_db)} sur Step 0"], accent=GREEN, size=17))
s.add(*card(8.6, 5.0, 4.1, 1.75, "M3 et M4 sans gain", ["0,499 à 0,504, aucun ne dépasse M1+M2"], accent=(0x5B, 0x7B, 0xA6), size=17))

# ----------------------------------------------------------------------------- 9. Results RoBERTa
s = new("La même ablation sur RoBERTa-base", notes=(
    "La question qui suit est de savoir si ce profil vient seulement de DistilBERT. "
    "On retrouve la même forme avec RoBERTa. M1 seul ne dépasse le modèle plat sur aucun des deux. "
    "M1+M2 monte des deux côtés, jusqu'à 0,505 et 0,526. "
    "RoBERTa est un peu devant, d'environ 0,02, mais le gain de M2 apparaît dans les deux cas. "
    "Ce n'est donc pas un effet du modèle choisi."))
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
    "M2 aide, et je veux comprendre pourquoi. "
    "Si on enlève le lien entre les émotions, la perte est nette, près de 0,03 sur DistilBERT et environ 0,02 sur RoBERTa. "
    "Si on enlève seulement le regard vers les phrases, les scores restent proches, et l'écart tient dans l'écart-type. "
    "Le gain vient donc surtout du lien entre les émotions. "
    "Cela répond aux émotions décidées seules, pas à la structure du texte perdue."))
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
    "Mais ce gain global ne signifie pas que toutes les émotions sont mieux détectées. "
    "Quand on regarde classe par classe, la situation est plus contrastée. "
    "M2 aide surtout les émotions de fréquence moyenne, comme desire, annoyance et curiosity. "
    "grief et relief montent aussi, mais avec une poignée d'exemples seulement, donc je ne m'y fie pas trop. "
    "Et ce n'est pas que du gain. nervousness et embarrassment baissent de près d'un dixième. "
    "L'écart entre classes fréquentes et classes très rares se réduit, alors qu'avec M1 seul il s'aggrave. "
    "Le gain de M2 ne vient donc pas des classes les plus rares."))
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
    "Les erreurs disent encore autre chose. "
    "Les dix motifs les plus fréquents, pour M1+M2 sur les trois essais réunis, passent tous par neutral. "
    "approval est prédit neutral, ou l'inverse, et on retrouve aussi annoyance et disapproval. "
    "Ce n'est pas une confusion entre deux émotions proches. "
    "C'est la frontière entre une émotion faible et pas d'émotion du tout, et la piste qui en sort est de mieux séparer neutral."))
s.add(*fpic(FIG / "errors.png", 0.6, 1.6, 8.1, "Dix motifs d'erreur les plus fréquents, M1+M2, DistilBERT, 3 graines cumulées (test)."))
s.add(*card(9.0, 1.6, 3.7, 2.4, "Constat", ["Les dix motifs impliquent tous neutral", "Approval, annoyance, disapproval vs neutral dominent"], accent=ORANGE, size=15))
s.add(*card(9.0, 4.2, 3.7, 2.4, "Lecture", ["Limite floue entre une émotion faible et pas d'émotion", "Piste : mieux séparer neutral"], accent=GREEN, size=15))

# ----------------------------------------------------------------------------- 13. Hypotheses verdict
s = new("Bilan des hypothèses : une seule est soutenue", notes=(
    "On peut maintenant revenir aux hypothèses, et le premier résultat est important parce qu'il ne va pas dans le sens de H1. "
    "M1, qui devait mieux garder la structure du texte, fait en réalité baisser le score, sur les deux modèles. "
    "C'est seulement quand j'ajoute M2, avec le lien entre les émotions, que les performances remontent. H2 est donc la seule hypothèse soutenue. "
    "H3 reste prudente, parce que le lexique ne donne pas un gain stable, et sur RoBERTa NRC bouge beaucoup d'un essai à l'autre. "
    "H4 aussi. Avec SenticNet, M4 ne fait pas mieux, donc je ne le retiens pas. "
    "Le modèle que je garde est M1+M2, sur DistilBERT comme sur RoBERTa. "
    "NRC avec M4 est un peu plus haut, à 0,510, mais seulement sur DistilBERT. Ce test n'existe pas pour RoBERTa, donc je ne peux pas le généraliser."))
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
divider(4, "Ces scores ont un sens seulement si je dis aussi ce qu'ils ne permettent pas de conclure. Je les replace d'abord par rapport à d'autres travaux, puis je dis ce que je retiens.")

# ----------------------------------------------------------------------------- Discussion and limits
s = new("Discussion : comparaison et limites", notes=(
    "Nos scores M1+M2 sont un peu au-dessus de 0,50 pour DistilBERT, et un peu plus haut pour RoBERTa. "
    "BERT sur GoEmotions est autour de 0,46, et Seq2Emo est à 0,473. "
    "Je les montre comme des repères, pas comme une compétition, parce que les modèles et les réglages ne sont pas les mêmes. "
    "Ce qui compte, ce sont nos comparaisons, faites avec le même entraînement. "
    "Il faut quand même garder à l'esprit que ces résultats ont leurs limites. "
    "Je n'ai travaillé que sur GoEmotions, donc sur des commentaires Reddit, et j'ai fait trois essais par configuration, sans test statistique. "
    "J'ai aussi gardé quatre époques et les mêmes réglages pour tous, un seul seuil à 0,5, et des textes coupés en quatre phrases. "
    "NRC avec M4 n'a pas été testé sur RoBERTa. "
    "Les résultats sont donc encourageants, mais ils ne suffisent pas à dire que le même gain se retrouverait dans d'autres conditions."))
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
    "Ce que le travail apporte, c'est une architecture en quatre modules qu'on peut allumer un par un, "
    "des tests comparables sur deux modèles et trois essais, et une lecture qui va jusqu'aux erreurs. "
    "Ce que j'en retiens est plus étroit. Je garde M1+M2, parce que le gain vient du lien entre les émotions. "
    "M3 et M4 n'apportent pas un gain stable ici, et le F1-macro cache des différences entre classes. "
    "Pour la suite, je donnerais plus de poids aux classes rares, je lierais le lexique à chaque émotion plutôt qu'avec une seule porte, "
    "je testerais d'autres datasets comme SemEval, et je comparerais avec des modèles plus généraux."))
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
    "Si on me demande pourquoi M1 seul baisse, c'est parce que le modèle plat voit le texte d'un bloc, "
    "alors que M1 le coupe en quatre phrases, sans encore de lien entre les émotions. C'est M2 qui rattrape cette baisse.\n"
    "Si on me demande pourquoi les lexiques aident peu, c'est parce que le lexique est le même pour toutes les émotions, "
    "et qu'il entre par une seule porte. Le gain reste petit, et il n'est pas stable.\n"
    "Si on me demande si les résultats sont solides, je dirais que j'ai trois essais, "
    "et que le gain de M2 est plus grand que l'écart entre ces essais. "
    "Je n'ai pas fait de test statistique, donc je ne dis pas que la différence est prouvée.\n"
    "Si on me demande pourquoi le F1-macro, c'est parce qu'il donne le même poids à chaque émotion, y compris grief, qui n'a que 6 exemples. "
    "Un score global cacherait les classes rares."))
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
