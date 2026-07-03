# Prédictions du Tournoi des Candidats 2026

Modèle de prédiction du **Tournoi des Candidats FIDE 2026** (sections open et féminine),
combinant un classifieur **LightGBM** calibré et une **simulation Monte-Carlo** pour estimer,
ronde après ronde, la probabilité de titre de chaque joueur.

**Démo en ligne :** https://valentinmartine.com/projets/echecs

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
| Open | Javokhir Sindarov | ronde 4 | 71,6 % |
| Féminin | Vaishali Rameshbabu | ronde 10 | 29,0 % |

La section open (vainqueur qui se détache tôt) est bien captée ; la section féminine, très
disputée, l'est plus tardivement — un profil attendu pour ce type de modèle.

## Structure

```
chess_src/            Cœur du projet (package Python)
  features/           Construction des variables
  models/             Modèle LightGBM
  simulation/         Simulation Monte-Carlo
  tournament_data.py  Accès aux données de tournoi (SQLite)
  export.py           Export des prédictions vers data/predictions.json
config/               Joueurs et hyperparamètres (open / féminin)
scripts/              Entraînement, import, backtest
  dev/                Utilitaires de débogage
dashboard/            Interface Streamlit
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

# Dashboard interactif
streamlit run dashboard/app.py
```

## Limites

Le corpus d'entraînement est restreint (~70 parties) : les performances sont exploratoires et
la calibration reste perfectible. Voir les axes d'amélioration sur la page de démo.

## Licence

MIT.
