# Prédictions du Tournoi des Candidats 2026

Modèle de prédiction du **Tournoi des Candidats FIDE 2026** (sections open et féminine),
combinant un classifieur **LightGBM** calibré et une **simulation Monte-Carlo** pour estimer,
ronde après ronde, la probabilité de titre de chaque joueur.

**Démo en ligne :** <https://valentinmartine.com/projets/echecs>

## Objectif

Le Tournoi des Candidats désigne le·la challenger·euse au titre mondial. À partir des parties
déjà jouées, le projet répond à une question simple : *au vu des résultats connus après la
ronde R, quelle est la probabilité que chaque joueur remporte le tournoi ?*

L'évaluation est **rétrospective et honnête** : la prévision « après la ronde R » n'utilise que
les rondes ≤ R — le modèle ne voit jamais l'avenir.

## Méthode

1. **Données** : parties récupérées via TWIC et l'API chess.com, stockées en SQLite.
2. **Variables** : écart d'Elo, forme récente, confrontations directes, couleur, points au
   tournoi, écart au leader et contexte de ronde (22 variables).
3. **Modèle** : `LightGBM` multiclasse (défaite / nulle / victoire) avec pondération des classes
   et **calibration de Platt**.
4. **Simulation** : les rondes restantes sont rejouées 1 000 fois (Monte-Carlo) pour estimer la
   probabilité de titre de chaque joueur.

## Performance

Le modèle se comporte comme un **suiveur de momentum** : il part du favori sur l'Elo puis
réévalue dès qu'un leader se détache.

| Section | Vainqueur réel | Favori confirmé à partir de | Proba moyenne du vainqueur |
| --- | --- | --- | --- |
| Open | Javokhir Sindarov | ronde 4 | 72,1 % |
| Féminin | Vaishali Rameshbabu | ronde 8 | 26,6 % |

Log-loss par partie sur les 15 % les plus récents (défaite / nulle / victoire) :

| Section | Modèle | Elo seul | Fréquences |
| --- | --- | --- | --- |
| Open | 0,955 | 0,997 | 1,095 |
| Féminin | 1,021 | 1,034 | 1,113 |

Deux tournois ne suffisent pas à juger la probabilité de titre : la log-loss par partie est la
mesure à suivre. Jusqu'au 30/09/2026, les rondes fermées étaient dupliquées avant le calcul des
variables et les copies voyaient leur propre résultat : l'ancienne log-loss de 0,67 sur les
Candidats open venait de cette fuite.

## Structure

```text
chess_src/            Cœur du projet (package Python)
  features/           Construction des variables
  models/             Modèle LightGBM
  simulation/         Simulation Monte-Carlo
  tournament_data.py  Accès aux données de tournoi (SQLite)
  export.py           Export des prédictions vers data/predictions.json
config/               Joueurs et hyperparamètres (open / féminin)
scripts/              Entraînement, import, backtest
  dev/                Utilitaires de débogage
data/                 Bases SQLite, modèles entraînés, prédictions
```

## Installation et usage

```bash
pip install -r requirements.txt

# Entraînement des modèles
python scripts/train.py
python scripts/train_women.py

# Génération des prédictions ronde par ronde -> data/predictions.json
python -m chess_src.export
```

Les prédictions sont visualisées sur la page de démo : <https://valentinmartine.com/projets/echecs>

## Limites

Le corpus d'entraînement est très déséquilibré selon la section : environ 11 800 parties d'élite
pour l'open, mais seulement environ 400 pour la section féminine, dont les prédictions sont donc
nettement plus fragiles. Les performances restent exploratoires et la calibration perfectible.
Voir les axes d'amélioration sur la page de démo.

## Licence

MIT.
