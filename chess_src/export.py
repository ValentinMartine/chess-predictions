"""Export des prédictions vers un artefact JSON web-ready.

Produit `data/predictions.json` : pour chaque tournoi (open et féminin), la
prévision **ronde par ronde** (classement + probabilités de titre issues de la
simulation Monte-Carlo, calculées uniquement à partir des rondes ≤ R), le
classement final réel, et un bloc de performance du modèle.

Cet artefact est le contrat de données consommé par le site (page projet Échecs).

Usage :
    python -m chess_src.export
"""

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

from chess_src.models.lgbm_model import ChessLGBMModel
from chess_src.features.pipeline import ChessFeaturePipeline
from chess_src.simulation.monte_carlo import CandidatesSimulator
from chess_src.tournament_data import load_config, get_completed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data"
N_SIMULATIONS = 1000

TOURNAMENTS = [
    {
        "id": "candidates_2026_open",
        "name": "FIDE Candidates 2026",
        "section": "open",
        "db": DATA / "chess_matches.db",
        "model": DATA / "chess_lgbm.pkl",
        "config": PROJECT_ROOT / "config" / "settings.yaml",
        "tournament": "Candidates 2026",
        "fide_ids": [2004887, 2020009, 8603405, 24116068, 14205481, 25059650, 4661654, 24175439],
    },
    {
        "id": "candidates_2026_women",
        "name": "FIDE Women's Candidates 2026",
        "section": "women",
        "db": DATA / "chess_matches_women.db",
        "model": DATA / "chess_lgbm_women.pkl",
        "config": PROJECT_ROOT / "config" / "settings_women.yaml",
        "tournament": "Women's Candidates 2026",
        "fide_ids": [4147103, 35006916, 14109336, 8608059, 5091756, 14111330, 8603642, 13708694],
    },
]


def load_model(model_path: Path) -> ChessLGBMModel:
    model = ChessLGBMModel.load(str(model_path))
    sidecar = Path(model_path).with_suffix(".features.json")
    if sidecar.exists():
        model.feature_cols = json.loads(sidecar.read_text(encoding="utf-8"))
    return model


def standings_after(config: dict, matches) -> list:
    """Classement reconstruit à partir des parties fournies (rondes ≤ R)."""
    players = config["players"]
    points = {p["fide_id"]: 0.0 for p in players}
    played = {p["fide_id"]: 0 for p in players}
    for _, r in matches.iterrows():
        w, b, res = int(r["white_id"]), int(r["black_id"]), float(r["result"])
        if w in points:
            points[w] += res
            played[w] += 1
        if b in points:
            points[b] += 1.0 - res
            played[b] += 1
    rows = [
        {
            "player": p["name"],
            "country": p.get("country"),
            "elo": int(p.get("rating_april_2026", 0)),
            "points": round(points[p["fide_id"]], 1),
            "games_played": played[p["fide_id"]],
            "fide_id": p["fide_id"],
        }
        for p in players
    ]
    rows.sort(key=lambda x: (-x["points"], -x["elo"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def title_odds_at(model, pipeline, config, all_matches, cutoff: int) -> list:
    """Probabilité de titre par joueur en ne connaissant que les rondes ≤ cutoff."""
    current = all_matches[all_matches["round"] <= cutoff].copy()
    remaining = all_matches[all_matches["round"] > cutoff].copy()
    remaining["result"] = np.nan

    simulator = CandidatesSimulator(model, pipeline, config["players"], num_simulations=N_SIMULATIONS)
    raw = simulator.simulate(current, remaining)

    id_to_name = {p["fide_id"]: p["name"] for p in config["players"]}
    odds = []
    for pid, prob in raw.items():
        se = float(np.sqrt(prob * (1 - prob) / N_SIMULATIONS))
        odds.append(
            {
                "player": id_to_name.get(pid, str(pid)),
                "fide_id": pid,
                "p_win": round(float(prob), 4),
                "ci95_low": round(max(0.0, prob - 1.96 * se), 4),
                "ci95_high": round(min(1.0, prob + 1.96 * se), 4),
            }
        )
    odds.sort(key=lambda x: -x["p_win"])
    return odds


def build_tournament(t: dict) -> dict:
    config = load_config(str(t["config"]))
    model = load_model(t["model"])
    pipeline = ChessFeaturePipeline(config, db_path=str(t["db"]))
    all_matches = get_completed(str(t["db"]), t["fide_ids"], t["tournament"])
    all_matches["round"] = all_matches["round"].astype(int)
    total_rounds = int(all_matches["round"].max())

    # Classement final réel et vainqueur
    final_standings = standings_after(config, all_matches)
    actual_winner = final_standings[0]["player"]

    # Prévisions ronde par ronde (1 .. total-1)
    rounds = []
    performance = []
    for R in range(1, total_rounds):
        print(f"  [{t['name']}] prévision après ronde {R}/{total_rounds - 1}", flush=True)
        odds = title_odds_at(model, pipeline, config, all_matches, R)
        standings = standings_after(config, all_matches[all_matches["round"] <= R])
        winner_prob = next((o["p_win"] for o in odds if o["player"] == actual_winner), 0.0)
        rounds.append({"round": R, "standings": standings, "title_odds": odds})
        performance.append(
            {
                "round": R,
                "top_pick": odds[0]["player"],
                "top_pick_prob": odds[0]["p_win"],
                "winner_prob": winner_prob,
                "hit": odds[0]["player"] == actual_winner,
            }
        )

    # Ronde à partir de laquelle le favori du modèle reste le vrai vainqueur
    converged_from = None
    for p in performance:
        if p["hit"] and converged_from is None:
            converged_from = p["round"]
        elif not p["hit"]:
            converged_from = None

    meta = next((v for k, v in config.items() if k.startswith("candidates")), {}) or {}
    return {
        "id": t["id"],
        "name": t["name"],
        "section": t["section"],
        "format": meta.get("format"),
        "rounds_total": total_rounds,
        "location": meta.get("location"),
        "start_date": meta.get("start_date"),
        "n_simulations": N_SIMULATIONS,
        "actual_winner": actual_winner,
        "final_standings": final_standings,
        "rounds": rounds,
        "performance": {
            "by_round": performance,
            "converged_from_round": converged_from,
            "mean_winner_prob": round(float(np.mean([p["winner_prob"] for p in performance])), 4),
        },
    }


def build() -> Path:
    tournaments = [build_tournament(t) for t in TOURNAMENTS]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "chess_candidates_2026_predictions",
        "method": f"LightGBM multiclasse calibré + simulation Monte-Carlo ({N_SIMULATIONS} tirages)",
        "tournaments": tournaments,
    }
    out = DATA / "predictions.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Écrit : {out} ({len(tournaments)} tournois, prévisions ronde par ronde)")
    return out


if __name__ == "__main__":
    build()
