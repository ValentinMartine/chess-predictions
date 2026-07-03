"""Accès aux données de tournoi (SQLite) partagé par le dashboard et l'export.

Regroupe les requêtes de lecture des parties jouées et à venir afin d'éviter
toute duplication entre l'interface Streamlit et le module d'export.
"""

import sqlite3

import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def sql_ids(ids: list) -> str:
    return ",".join(str(i) for i in ids)


def get_completed(db_path: str, fide_ids: list, tournament: str) -> pd.DataFrame:
    """Parties déjà jouées (résultat renseigné) du tournoi."""
    conn = sqlite3.connect(db_path)
    ids_sql = sql_ids(fide_ids)
    query = f"""
        SELECT m.id, m.round, m.white_id, m.black_id, m.result, m.played_at, m.tournament
        FROM matches m
        WHERE m.result IS NOT NULL
          AND m.tournament = ?
          AND m.white_id IN ({ids_sql})
          AND m.black_id IN ({ids_sql})
        ORDER BY m.round ASC, m.id ASC
    """
    df = pd.read_sql_query(query, conn, params=(tournament,))
    conn.close()
    return df


def get_upcoming(db_path: str, fide_ids: list, tournament: str) -> pd.DataFrame:
    """Parties à venir (résultat non renseigné) du tournoi."""
    conn = sqlite3.connect(db_path)
    ids_sql = sql_ids(fide_ids)
    query = f"""
        SELECT
            m.id, m.round,
            pw.name AS white, pb.name AS black,
            m.white_id, m.black_id, m.played_at, m.tournament
        FROM matches m
        JOIN players pw ON m.white_id = pw.id
        JOIN players pb ON m.black_id = pb.id
        WHERE m.result IS NULL
          AND m.tournament = ?
          AND m.white_id IN ({ids_sql})
          AND m.black_id IN ({ids_sql})
        ORDER BY m.round ASC, m.id ASC
    """
    df = pd.read_sql_query(query, conn, params=(tournament,))
    conn.close()
    return df
