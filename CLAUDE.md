# chess-predictions : repères projet

Prédiction du Tournoi des Candidats FIDE 2026, sections open et féminine.
Démo : https://valentinmartine.com/projets/echecs. Règles générales dans `~/.claude/CLAUDE.md`.

## Chemins clés
- `chess_src/database.py` : SQLite des parties (TWIC et API chess.com).
- `chess_src/features/` : 22 variables : écart d'Elo, forme, H2H, couleur, points au tournoi,
  écart au leader, contexte de ronde.
- `chess_src/models/` : LightGBM multiclasse (défaite / nulle / victoire), pondération des classes,
  **calibration de Platt**.
- `chess_src/simulation/` : Monte-Carlo, 1 000 rejeux des rondes restantes.
- `chess_src/export.py` : JSON consommé par le site.
- `scripts/` : import TWIC (open et féminin), entraînement (`train.py`, `train_women.py`),
  `verify_pipeline.py`.
- `ruff.toml` : le lint passe par ruff.

## Règle cardinale
L'évaluation est **rétrospective et honnête** : la prévision « après la ronde R » n'utilise que les
rondes <= R. Le modèle ne voit jamais l'avenir.

## Comportement connu
Le modèle est un **suiveur de momentum** : il part du favori Elo et réévalue dès qu'un leader se détache.
Open : favori confirmé dès la ronde 4 (proba moyenne du vainqueur 72,1 %). Féminin, très disputé :
ronde 8 (26,6 %). Log-loss par partie en test : 0,955 open, 1,021 féminin (Elo seul : 0,997 et 1,034).

## Fuite corrigée le 30/09/2026
La surpondération des rondes fermées se fait **après** le calcul des variables et sur le seul
entraînement (`surponderer_round_robin`). Dupliquées avant, les copies d'une partie voyaient son
résultat dans l'Elo et les points au tournoi. Test : `tests/test_preparation_entrainement.py`.

## Historique
Streamlit a été **retiré** le 03/07/2026 au profit de la page web du site : ne pas le réintroduire.
