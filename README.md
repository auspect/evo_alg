# Introduction aux algorithmes évolutionnaires

Notebook interactif [marimo](https://marimo.io) qui illustre comment utiliser des algorithmes d'optimisation globaux pour résoudre un problème de scoring.

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/auspect/evo_alg/blob/master/intro_algos_evolutionnaires.py)

## Contexte

On cherche les coefficients d'une combinaison linéaire de variables qui maximise le **tau-b de Kendall** entre un score continu et une cible ordinale — typique d'un problème de scoring en risque de crédit où l'on souhaite obtenir un ranking pertinent. La fonction "objectif" n'est pas différentiable, ce qui motive l'utilisation d'algorithmes agnostiques.

*Remarque : L'évolution différentielle n'utilise pas la différentiabilité, contrairement à ce que son nom pourrait laisser entendre. Mais la différence (-) entre deux combinaisons de paramètres intervient dans le processus d'évolution, d'où le nom.*

Les variables explicatives continues sont d'abord discrétisées (4 modalités chacune), puis encodées en dummies. L'espace de recherche est donc discret.

## Algorithmes

- **Baseline** : population entièrement renouvelée à chaque génération (recherche purement aléatoire)
- **Algorithme génétique** : sélection par tournoi, crossover par moyenne, mutation aléatoire, élitisme
- **Évolution différentielle** : stratégie *best1bin* — mutation à partir du meilleur individu et de différences entre individus aléatoires

## Lancer le notebook

```bash
uv run marimo edit intro_algos_evolutionnaires.py
```

## Dépendances

Python ≥ 3.12 — géré via [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```