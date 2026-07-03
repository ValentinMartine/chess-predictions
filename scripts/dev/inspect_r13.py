
import sqlite3
import pandas as pd

DB_MEN = "data/chess_matches.db"
DB_WOMEN = "data/chess_matches_women.db"

def inspect_round(db_path, tournament_name, round_num):
    conn = sqlite3.connect(db_path)
    print(f"\n--- {tournament_name} Round {round_num} ---")
    
    query = """
    SELECT m.id, m.round, pw.name as white, pb.name as black, m.result
    FROM matches m
    JOIN players pw ON m.white_id = pw.id
    JOIN players pb ON m.black_id = pb.id
    WHERE m.tournament = ? AND m.round = ?
    """
    df = pd.read_sql_query(query, conn, params=(tournament_name, round_num))
    print(df)
    conn.close()

if __name__ == "__main__":
    inspect_round(DB_MEN, "Candidates 2026", 13)
    inspect_round(DB_WOMEN, "Women's Candidates 2026", 13)
