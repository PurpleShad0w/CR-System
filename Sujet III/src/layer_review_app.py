from __future__ import annotations

from pathlib import Path
import argparse
import yaml
import pandas as pd
import streamlit as st


def _normalize_decisions(decisions: dict) -> dict:
    """
    Nettoie les décisions existantes :
    priorité keep > drop > undecided.
    """
    keep = set(decisions.get("keep", []) or [])
    drop = set(decisions.get("drop", []) or [])
    undecided = set(decisions.get("undecided", []) or [])

    drop = drop - keep
    undecided = undecided - keep - drop

    return {
        "keep": sorted(keep),
        "drop": sorted(drop),
        "undecided": sorted(undecided),
    }


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {"keep": [], "drop": [], "undecided": []}

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {
        "keep": [],
        "drop": [],
        "undecided": [],
    }

    return _normalize_decisions(data)


def _save_yaml(path: Path, data: dict) -> None:
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _set_decision(decisions: dict, layer: str, target: str) -> dict:
    """
    Déplace un layer vers keep/drop/undecided en le retirant d'abord
    de toutes les catégories.

    Garantit qu'un layer ne peut jamais être présent dans plusieurs listes.
    """
    valid_targets = {"keep", "drop", "undecided"}
    if target not in valid_targets:
        raise ValueError(f"Target invalide : {target}")

    # initialise les listes si absentes
    for key in valid_targets:
        decisions.setdefault(key, [])

    # retire le layer de toutes les listes
    for key in valid_targets:
        decisions[key] = [x for x in decisions[key] if x != layer]

    # ajoute dans la liste cible
    decisions[target].append(layer)

    # déduplique sans perdre l'ordre alphabétique
    for key in valid_targets:
        decisions[key] = sorted(set(decisions[key]))

    return decisions


def main(review_dir: Path) -> None:
    st.set_page_config(
        page_title="Layer review",
        layout="wide",
    )

    manifest_path = review_dir / "layers_manifest.csv"
    decisions_path = review_dir / "layer_decisions.yaml"

    if not manifest_path.exists():
        st.error(f"Manifest introuvable : {manifest_path}")
        st.stop()

    df = pd.read_csv(manifest_path)
    decisions = _load_yaml(decisions_path)
    _save_yaml(decisions_path, decisions)

    st.title("Revue interactive des layers DXF")

    st.caption(
        "Décide couche par couche si elle doit être gardée ou supprimée. "
        "Les choix sont enregistrés dans layer_decisions.yaml."
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Layers total", len(df))
    col_b.metric("Keep", len(decisions.get("keep", [])))
    col_c.metric("Drop", len(decisions.get("drop", [])))
    col_d.metric("Undecided", len(decisions.get("undecided", [])))

    status_filter = st.sidebar.selectbox(
        "Statut",
        ["undecided", "keep", "drop", "all"],
        index=0,
    )

    search = st.sidebar.text_input("Recherche layer", "")

    df_view = df.copy()

    if status_filter != "all":
        selected_set = set(decisions.get(status_filter, []))
        df_view = df_view[df_view["layer"].isin(selected_set)]

    if search.strip():
        s = search.lower().strip()
        df_view = df_view[df_view["layer"].str.lower().str.contains(s, na=False)]

    df_view = df_view.sort_values(["segment_count", "total_length"], ascending=False)

    if df_view.empty:
        st.info("Aucun layer pour ce filtre.")
        return

    layer = st.sidebar.selectbox(
        "Layer à examiner",
        df_view["layer"].tolist(),
        index=0,
    )

    row = df[df["layer"] == layer].iloc[0].to_dict()

    st.subheader(layer)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Segments", int(row["segment_count"]))
    c2.metric("Longueur totale", round(float(row["total_length"]), 2))
    c3.metric("Largeur bbox", round(float(row["width"]), 2) if row["width"] != "" else "-")
    c4.metric("Hauteur bbox", round(float(row["height"]), 2) if row["height"] != "" else "-")

    preview = Path(str(row.get("preview", "")))
    if preview.exists():
        st.image(str(preview), caption="Preview du layer seul", use_container_width=True)
    else:
        st.warning("Preview PNG indisponible pour ce layer.")

    current_status = "undecided"
    for key in ("keep", "drop", "undecided"):
        if layer in decisions.get(key, []):
            current_status = key
            break

    st.write(f"Statut actuel : `{current_status}`")

    b1, b2, b3 = st.columns(3)

    if b1.button("KEEP - garder ce layer", use_container_width=True):
        decisions = _set_decision(decisions, layer, "keep")
        _save_yaml(decisions_path, decisions)
        st.rerun()

    if b2.button("DROP - supprimer ce layer", use_container_width=True):
        decisions = _set_decision(decisions, layer, "drop")
        _save_yaml(decisions_path, decisions)
        st.rerun()

    if b3.button("UNDECIDED - remettre en attente", use_container_width=True):
        decisions = _set_decision(decisions, layer, "undecided")
        _save_yaml(decisions_path, decisions)
        st.rerun()

    st.divider()

    st.subheader("Résumé des décisions")
    st.code(
        yaml.safe_dump(decisions, allow_unicode=True, sort_keys=False),
        language="yaml",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", type=Path, default=Path("data/work/layer_review"))
    args = parser.parse_args()

    main(args.review_dir)