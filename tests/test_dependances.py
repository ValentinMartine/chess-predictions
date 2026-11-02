import ast
import re
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
# Nom de paquet pip différent du nom importé
ALIAS = {"python-chess": "chess", "beautifulsoup4": "bs4", "pyyaml": "yaml", "scikit-learn": "sklearn"}
LOCAUX = {"chess_src", "scripts", "tests", "conftest", "consolidate_players"}


def _modules_declares() -> set[str]:
    declares = set()
    for ligne in (RACINE / "requirements.txt").read_text(encoding="utf-8").splitlines():
        nom = re.split(r"[<>=\[ ;]", ligne.strip(), maxsplit=1)[0].lower()
        if nom and not nom.startswith("#"):
            declares.add(ALIAS.get(nom, nom.replace("-", "_")))
    return declares | {"pytest"}


def _modules_importes() -> set[str]:
    importes = set()
    for dossier in ("chess_src", "scripts", "tests"):
        for fichier in (RACINE / dossier).rglob("*.py"):
            if "dev" in fichier.parts:
                continue
            for noeud in ast.walk(ast.parse(fichier.read_text(encoding="utf-8"))):
                if isinstance(noeud, ast.Import):
                    importes |= {a.name.split(".")[0] for a in noeud.names}
                elif isinstance(noeud, ast.ImportFrom) and noeud.level == 0 and noeud.module:
                    importes.add(noeud.module.split(".")[0])
    return importes - set(sys.stdlib_module_names) - LOCAUX


def test_chaque_import_tiers_est_dans_requirements():
    # La CI n'installe que requirements.txt : un import absent casse la régénération quotidienne
    manquants = _modules_importes() - _modules_declares()
    assert not manquants, f"absents de requirements.txt : {sorted(manquants)}"
