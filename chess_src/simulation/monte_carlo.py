"""Simulation Monte-Carlo vectorisée du tournoi.

Toutes les simulations sont menées en parallèle avec numpy : à chaque ronde, les
probabilités des parties de toutes les simulations sont prédites en un seul appel
au modèle, puis les issues sont tirées et l'état mis à jour de façon vectorisée.
Le modèle n'exploitant que les features dynamiques points/écart au leader/last2,
seules celles-ci sont recalculées par simulation.
"""

import numpy as np
import pandas as pd

from chess_src.models.lgbm_model import ChessLGBMModel
from chess_src.features.pipeline import ChessFeaturePipeline

TOTAL_ROUNDS = 14

# Features dynamiques recalculées par simulation (celles réellement utilisées par le modèle).
DYNAMIC_COLS = [
    "white_tournament_points",
    "black_tournament_points",
    "tournament_points_diff",
    "white_gap_to_leader",
    "black_gap_to_leader",
    "white_last2_score",
    "black_last2_score",
]


class CandidatesSimulator:
    def __init__(
        self,
        model: ChessLGBMModel,
        pipeline: ChessFeaturePipeline,
        players_config: list,
        num_simulations: int = 1000,
    ):
        self.model = model
        self.pipeline = pipeline
        self.players = players_config
        self.num_simulations = num_simulations

    def simulate(self, current_matches: pd.DataFrame, remaining_matches: pd.DataFrame) -> dict:
        p_ids = [p["fide_id"] for p in self.players]
        n_p = len(p_ids)
        idx = {pid: i for i, pid in enumerate(p_ids)}
        N = self.num_simulations

        # ── Features statiques calculées une seule fois ────────────────────
        full_df = pd.concat([current_matches, remaining_matches], ignore_index=True)
        full_df["white_id"] = full_df["white_id"].astype(int)
        full_df["black_id"] = full_df["black_id"].astype(int)
        full_df["result"] = pd.to_numeric(full_df["result"], errors="coerce")
        df_proc = self.pipeline.process(full_df)

        rem_df = df_proc[df_proc["result"].isna()].copy()
        rem_df["round"] = pd.to_numeric(rem_df["round"], errors="coerce").astype(int)
        rem_df = rem_df.sort_values(["round", "white_id"]).reset_index(drop=True)

        total_rounds = int(pd.to_numeric(full_df["round"], errors="coerce").max())

        # ── État initial issu des parties déjà jouées ──────────────────────
        points0 = np.zeros(n_p)
        sbr0 = np.zeros((n_p, total_rounds + 2))  # score par [joueur, ronde]
        for _, r in current_matches.iterrows():
            w, b = int(r["white_id"]), int(r["black_id"])
            if w not in idx or b not in idx:
                continue
            res = float(r["result"])
            rnd = int(pd.to_numeric(r.get("round", 1)))
            points0[idx[w]] += res
            points0[idx[b]] += 1.0 - res
            sbr0[idx[w], rnd] = res
            sbr0[idx[b], rnd] = 1.0 - res

        if rem_df.empty:
            winner = int(np.argmax(points0))
            return {pid: (1.0 if i == winner else 0.0) for pid, i in idx.items()}

        points = np.tile(points0, (N, 1))                 # (N, n_p)
        sbr = np.tile(sbr0, (N, 1, 1))                     # (N, n_p, R+2)

        feature_cols = list(self.model.feature_cols)
        F = len(feature_cols)
        col_pos = {name: (feature_cols.index(name) if name in feature_cols else None) for name in DYNAMIC_COLS}

        rounds_grouped = [(int(rnd), grp.reset_index(drop=True)) for rnd, grp in rem_df.groupby("round")]
        rng = np.random.default_rng()

        for rnd, gm in rounds_grouped:
            g = len(gm)
            w_idx = np.array([idx[int(x)] for x in gm["white_id"]])
            b_idx = np.array([idx[int(x)] for x in gm["black_id"]])

            base = gm[feature_cols].to_numpy(dtype=float)             # (g, F)
            feat = np.broadcast_to(base, (N, g, F)).copy()            # (N, g, F)

            w_pts = points[:, w_idx]                                  # (N, g)
            b_pts = points[:, b_idx]
            leader = points.max(axis=1, keepdims=True)                # (N, 1)

            r1 = sbr[:, :, rnd - 1] if rnd - 1 >= 1 else np.zeros((N, n_p))
            r2 = sbr[:, :, rnd - 2] if rnd - 2 >= 1 else np.zeros((N, n_p))
            last2 = r1 + r2                                           # (N, n_p)

            vals = {
                "white_tournament_points": w_pts,
                "black_tournament_points": b_pts,
                "tournament_points_diff": w_pts - b_pts,
                "white_gap_to_leader": leader - w_pts,
                "black_gap_to_leader": leader - b_pts,
                "white_last2_score": last2[:, w_idx],
                "black_last2_score": last2[:, b_idx],
            }
            for name, v in vals.items():
                j = col_pos[name]
                if j is not None:
                    feat[:, :, j] = v

            X = pd.DataFrame(feat.reshape(N * g, F), columns=feature_cols)
            probs = np.asarray(self.model.predict_proba(X), dtype=np.float64).reshape(N, g, 3)
            probs /= probs.sum(axis=2, keepdims=True)

            cum = np.cumsum(probs, axis=2)
            u = rng.random((N, g, 1))
            outcome = (u > cum).sum(axis=2)      # 0=Noirs, 1=Nulle, 2=Blancs
            res = outcome * 0.5                   # score des Blancs : 0.0 / 0.5 / 1.0

            points[:, w_idx] += res
            points[:, b_idx] += 1.0 - res
            sbr[:, w_idx, rnd] = res
            sbr[:, b_idx, rnd] = 1.0 - res

        # ── Vainqueur de chaque simulation (départage aléatoire) ───────────
        max_pts = points.max(axis=1, keepdims=True)
        is_max = points >= max_pts - 1e-9
        tie = rng.random((N, n_p))
        tie[~is_max] = -1.0
        winner_idx = tie.argmax(axis=1)
        counts = np.bincount(winner_idx, minlength=n_p)

        return {pid: float(counts[i]) / N for pid, i in idx.items()}
