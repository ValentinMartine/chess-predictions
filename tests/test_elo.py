"""Invariants de l'Elo échecs.

Protège la symétrie des probabilités, la conservation des points, les paliers
de K par niveau, et surtout le fait que les Elo pré-partie d'une ronde
n'utilisent que les rondes antérieures.
"""

import pandas as pd

from chess_src.features.elo import (
    ChessEloCalculator,
    EloState,
    expected_score,
    update_elo,
)


class TestExpectedScore:
    def test_deux_joueurs_egaux_donnent_une_chance_sur_deux(self):
        assert expected_score(2700.0, 2700.0) == 0.5

    def test_les_probabilites_des_deux_joueurs_somment_a_un(self):
        assert expected_score(2800.0, 2600.0) + expected_score(2600.0, 2800.0) == 1.0

    def test_la_probabilite_croit_avec_l_ecart_de_niveau(self):
        valeurs = [expected_score(2700.0 + e, 2700.0) for e in (0, 50, 100, 400)]
        assert valeurs == sorted(valeurs)


class TestUpdateElo:
    def test_la_somme_des_elo_est_conservee(self):
        ra, rb = update_elo(2750.0, 2650.0, score=1.0)
        assert round(ra + rb, 9) == round(2750.0 + 2650.0, 9)

    def test_le_vainqueur_monte_et_le_perdant_descend(self):
        ra, rb = update_elo(2700.0, 2700.0, score=1.0)
        assert ra > 2700.0 > rb

    def test_une_nulle_entre_egaux_ne_change_rien(self):
        ra, rb = update_elo(2700.0, 2700.0, score=0.5)
        assert ra == rb == 2700.0

    def test_une_nulle_contre_plus_fort_rapporte_des_points(self):
        ra, _ = update_elo(2600.0, 2800.0, score=0.5)
        assert ra > 2600.0

    def test_un_k_plus_grand_amplifie_la_variation(self):
        petit, _ = update_elo(2700.0, 2700.0, score=1.0, k=10.0)
        grand, _ = update_elo(2700.0, 2700.0, score=1.0, k=40.0)
        assert grand - 2700.0 > petit - 2700.0


class TestPaliersDeK:
    def test_le_k_decroit_par_paliers_avec_le_niveau(self):
        calc = ChessEloCalculator()
        assert calc._k_for(2750.0) < calc._k_for(2600.0) < calc._k_for(2400.0)

    def test_les_bornes_des_paliers_sont_incluses(self):
        calc = ChessEloCalculator()
        assert calc._k_for(2700.0) == calc._k_for(2900.0)
        assert calc._k_for(2500.0) == calc._k_for(2699.0)


class TestEtatInitial:
    def test_un_joueur_inconnu_part_du_classement_par_defaut(self):
        calc = ChessEloCalculator(initial_rating=2650.0)
        assert calc._get_state(42).rating == 2650.0

    def test_le_classement_fourni_en_configuration_prime(self):
        calc = ChessEloCalculator(initial_ratings_dict={7: 2800.0})
        assert calc._get_state(7).rating == 2800.0
        assert calc._get_state(8).rating == 1500.0

    def test_l_etat_par_defaut_demarre_a_zero_partie(self):
        assert EloState().matches_played == 0


def _parties(resultats):
    """Construit un mini-tournoi : une partie par ronde, blancs = joueur 1."""
    return pd.DataFrame(
        [
            {"white_id": 1, "black_id": 2, "result": r, "played_at": f"2026-01-{j:02d}"}
            for j, r in enumerate(resultats, start=1)
        ]
    )


class TestAbsenceDeFuite:
    def test_les_elo_pre_partie_ignorent_les_rondes_posterieures(self):
        # L'Elo affiché avant la ronde N doit être le même, que le tournoi
        # s'arrête là ou qu'il continue : c'est la garantie d'une évaluation
        # rétrospective honnête, ronde par ronde.
        court = ChessEloCalculator().compute(_parties([1.0, 0.0]))
        long = ChessEloCalculator().compute(_parties([1.0, 0.0, 1.0, 0.5]))
        assert list(long["white_elo"][:2]) == list(court["white_elo"])
        assert list(long["black_elo"][:2]) == list(court["black_elo"])

    def test_la_premiere_ronde_part_des_classements_initiaux(self):
        resultat = ChessEloCalculator().compute(_parties([1.0]))
        assert resultat["white_elo"].iloc[0] == 1500.0
        assert resultat["black_elo"].iloc[0] == 1500.0

    def test_les_parties_sont_rejouees_dans_l_ordre_chronologique(self):
        parties = _parties([1.0, 0.0])
        desordre = parties.iloc[::-1].reset_index(drop=True)
        assert list(ChessEloCalculator().compute(desordre)["white_elo"]) == list(
            ChessEloCalculator().compute(parties)["white_elo"]
        )

    def test_une_partie_sans_resultat_ne_modifie_pas_les_classements(self):
        # Les rondes à venir sont présentes dans la table mais non jouées.
        resultat = ChessEloCalculator().compute(_parties([None, None]))
        assert list(resultat["white_elo"]) == [1500.0, 1500.0]
