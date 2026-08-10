from __future__ import annotations
from pathlib import Path
import pandas as pd


def read_semicolon_csv(path):
    # Nouveau comportement : lecture robuste comma/semicolon.
    # On garde le nom de fonction pour ne pas casser les imports existants.
    last_err = None

    for sep in [",", ";"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python", encoding="utf-8-sig", na_values=["NULL", "null", ""], )
            # Si tout est lu dans une seule colonne, mauvais séparateur.
            if len(df.columns) <= 1:
                continue

            df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
            return df

        except Exception as e:
            last_err = e

    raise RuntimeError(f"Impossible de lire le CSV {path}. Dernière erreur: {last_err}")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
