"""Les variables d'une partie ne doivent jamais dépendre de son propre résultat.

Régression du 30/09/2026 : les rondes fermées étaient triplées avant le calcul des
variables, si bien que les copies d'une partie voyaient déjà son résultat dans l'Elo,
les points au tournoi et la performance intra-tournoi.
"""

import sqlite3

import pandas as pd
import yaml

from scripts.train import preparer_donnees, surponderer_round_robin

TOURNOI = "Norway Chess 2025"
APPARIEMENTS = [(1, 2), (3, 4), (1, 3), (2, 4), (1, 4), (2, 3)]


def _creer_base(tmp_path, resultat_partie_suivie):
    db = tmp_path / "parties.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE players (id INTEGER PRIMARY KEY, name TEXT, country TEXT, rating_initial REAL)")
    conn.execute(
        "CREATE TABLE matches (id INTEGER PRIMARY KEY AUTOINCREMENT, white_id INTEGER, black_id INTEGER,"
        " result REAL, played_at TEXT, tournament TEXT, round INTEGER)"
    )
    conn.executemany("INSERT INTO players VALUES (?, ?, ?, ?)", [(i, f"J{i}", "FRA", 2700.0) for i in range(1, 5)])
    for n, (blancs, noirs) in enumerate(APPARIEMENTS):
        ronde = n // 2 + 1
        resultat = resultat_partie_suivie if n == 0 else 0.5
        conn.execute(
            "INSERT INTO matches (white_id, black_id, result, played_at, tournament, round) VALUES (?, ?, ?, ?, ?, ?)",
            (blancs, noirs, resultat, f"2025-06-0{ronde}", TOURNOI, ronde),
        )
    conn.commit()
    conn.close()
    config = tmp_path / "settings.yaml"
    config.write_text(yaml.safe_dump({"players": []}), encoding="utf-8")
    return db, config


def _variables_partie_suivie(tmp_path, resultat):
    dossier = tmp_path / str(resultat)
    dossier.mkdir()
    df = preparer_donnees(*_creer_base(dossier, resultat))
    partie = df[(df["white_id"] == 1) & (df["black_id"] == 2)]
    return partie.drop(columns=["result"]).reset_index(drop=True)


def test_les_variables_d_une_partie_ignorent_son_resultat(tmp_path):
    victoire = _variables_partie_suivie(tmp_path, 1.0)
    defaite = _variables_partie_suivie(tmp_path, 0.0)
    pd.testing.assert_frame_equal(victoire, defaite)


def test_une_partie_n_apparait_qu_une_fois_avant_surponderation(tmp_path):
    df = _variables_partie_suivie(tmp_path, 1.0)
    assert len(df) == 1


def test_la_surponderation_triple_les_rondes_fermees_seulement():
    df = pd.DataFrame(
        {
            "tournament": [TOURNOI, "Open de Paris", "Candidates 2026"],
            "played_at": pd.to_datetime(["2025-06-01", "2025-06-02", "2026-04-01"]),
        }
    )
    sortie = surponderer_round_robin(df, "Candidates 2026")
    assert sortie["tournament"].value_counts().to_dict() == {TOURNOI: 3, "Candidates 2026": 3, "Open de Paris": 1}
    assert sortie["played_at"].is_monotonic_increasing
