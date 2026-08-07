from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_yaml(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or default


def save_yaml(data: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def normalize_decisions(decisions: dict[str, Any] | None) -> dict[str, list[str]]:
    decisions = decisions or {}
    return {
        "keep": sorted(set(map(str, decisions.get("keep", []) or []))),
        "drop": sorted(set(map(str, decisions.get("drop", []) or []))),
        "undecided": sorted(set(map(str, decisions.get("undecided", []) or []))),
    }


def _contains_any(value: str, keywords: list[str]) -> bool:
    u = str(value or "").upper()
    return any(str(k).upper() in u for k in keywords)


def _text_blob(df: pd.DataFrame) -> pd.Series:
    pieces = []
    for col in ["layer", "source_block", "block_name", "entity_type"]:
        if col in df.columns:
            pieces.append(df[col].fillna("").astype(str))
    if not pieces:
        return pd.Series("", index=df.index)
    out = pieces[0]
    for p in pieces[1:]:
        out = out + " " + p
    return out


def _score_keyword(blob: pd.Series, keywords: list[str], val: float) -> pd.Series:
    return blob.apply(lambda s: val if _contains_any(s, keywords) else 0.0)


def classify_features(df: pd.DataFrame, rules: dict[str, Any] | None, decisions: dict[str, Any] | None) -> pd.DataFrame:
    rules = rules or {}
    clf = rules.get("classification", {}) or {}
    decisions = normalize_decisions(decisions)
    out = df.copy()
    if out.empty:
        out["feature_class"] = []
        return out

    blob = _text_blob(out)
    length = out["length"].fillna(0.0)
    area = out["area"].fillna(0.0)
    width = out["bbox_width"].abs().fillna(0.0)
    height = out["bbox_height"].abs().fillna(0.0)
    min_dim = width.combine(height, min)
    max_dim = width.combine(height, max)
    aspect = max_dim / min_dim.replace(0, 1)
    etype = out["entity_type"].fillna("").astype(str).str.upper()

    min_wall_length = float(clf.get("min_wall_length", 20.0))
    min_feature_length = float(clf.get("min_feature_length", 3.0))

    out["wall_score"] = 0.0
    out["door_score"] = 0.0
    out["window_score"] = 0.0
    out["stairs_score"] = 0.0
    out["furniture_score"] = 0.0
    out["annotation_score"] = 0.0

    out["wall_score"] += _score_keyword(blob, clf.get("wall_keywords", []), 0.34)
    out["wall_score"] += etype.isin(["LWPOLYLINE", "POLYLINE", "LINE"]).astype(float) * 0.18
    out["wall_score"] += out["closed"].fillna(False).astype(bool).astype(float) * 0.12
    out["wall_score"] += (length >= min_wall_length).astype(float) * 0.22
    out["wall_score"] += (area > 0).astype(float) * 0.06
    out["wall_score"] += ((width > 0) & (height > 0)).astype(float) * 0.08

    out["door_score"] += _score_keyword(blob, clf.get("door_keywords", []), 0.48)
    out["door_score"] += etype.isin(["ARC", "INSERT", "LINE", "LWPOLYLINE", "POLYLINE"]).astype(float) * 0.15
    out["door_score"] += (length >= min_feature_length).astype(float) * 0.12
    out["door_score"] += (out.get("arc_angle", pd.Series(0, index=out.index)).fillna(0).between(35, 115)).astype(float) * 0.22
    out["door_score"] += ((aspect >= 1.2) & (aspect <= 12)).astype(float) * 0.03

    out["window_score"] += _score_keyword(blob, clf.get("window_keywords", []), 0.45)
    out["window_score"] += etype.isin(["LINE", "LWPOLYLINE", "POLYLINE", "INSERT"]).astype(float) * 0.16
    out["window_score"] += (length >= min_feature_length).astype(float) * 0.12
    out["window_score"] += ((aspect >= 2.0) & (min_dim <= max_dim)).astype(float) * 0.17
    out["window_score"] += out["closed"].fillna(False).astype(bool).astype(float) * 0.05

    out["stairs_score"] += _score_keyword(blob, clf.get("stairs_keywords", []), 0.55)
    out["stairs_score"] += etype.isin(["LINE", "LWPOLYLINE", "POLYLINE", "INSERT"]).astype(float) * 0.15
    out["stairs_score"] += (length >= min_feature_length).astype(float) * 0.10

    out["furniture_score"] += _score_keyword(blob, clf.get("furniture_keywords", []), 0.55)
    out["furniture_score"] += etype.isin(["INSERT", "CIRCLE", "LWPOLYLINE", "POLYLINE"]).astype(float) * 0.08

    out["annotation_score"] += _score_keyword(blob, clf.get("annotation_keywords", []), 0.55)
    drop_types = set(map(str.upper, rules.get("drop_entity_types", []) or []))
    out["annotation_score"] += etype.isin(drop_types).astype(float) * 0.45

    thresholds = {
        "wall": float(clf.get("wall_score_threshold", 0.48)),
        "door": float(clf.get("door_score_threshold", 0.45)),
        "window": float(clf.get("window_score_threshold", 0.42)),
        "stairs": float(clf.get("stairs_score_threshold", 0.45)),
        "furniture": float(clf.get("furniture_score_threshold", 0.50)),
    }

    score_cols = ["wall_score", "door_score", "window_score", "stairs_score", "furniture_score"]
    out["feature_class"] = "other"
    for idx, row in out.iterrows():
        best_col = max(score_cols, key=lambda c: float(row.get(c, 0.0)))
        label = best_col.replace("_score", "")
        if float(row.get(best_col, 0.0)) >= thresholds[label]:
            out.at[idx, "feature_class"] = label
    out.loc[out["annotation_score"] >= 0.55, "feature_class"] = "annotation"

    out["decision"] = "undecided"
    out.loc[out["layer"].isin(decisions["keep"]), "decision"] = "keep"
    out.loc[out["layer"].isin(decisions["drop"]), "decision"] = "drop"

    # Les portes/fenêtres/escalier ne doivent plus être perdus si leur layer n'a pas été explicitement drop.
    auto_keep = out["feature_class"].isin(["wall", "door", "window", "stairs"])
    auto_drop = out["feature_class"].isin(["annotation"])
    drop_contains = rules.get("drop_layers_contains", []) or []
    auto_drop |= out["layer"].fillna("").astype(str).apply(lambda s: _contains_any(s, drop_contains))
    drop_small = rules.get("drop_small_entities", {}) or {}
    if bool(drop_small.get("enabled", False)):
        min_len = float(drop_small.get("min_length", 0.0) or 0.0)
        auto_drop |= length.lt(min_len) & area.eq(0.0) & ~auto_keep
    out.loc[(out["decision"] == "undecided") & auto_keep, "decision"] = "keep"
    out.loc[(out["decision"] == "undecided") & auto_drop, "decision"] = "drop"
    out.loc[(out["decision"] == "undecided") & out["feature_class"].eq("other"), "decision"] = "keep"

    for col in score_cols + ["annotation_score"]:
        out[col] = out[col].clip(0, 1)
    return out


def build_review_table(df: pd.DataFrame, rules: dict[str, Any] | None, decisions: dict[str, Any] | None) -> pd.DataFrame:
    classified = classify_features(df, rules, decisions)
    if classified.empty:
        return pd.DataFrame()
    tab = (
        classified.groupby("layer")
        .agg(
            n_entities=("entity_id", "count"),
            entity_types=("entity_type", lambda s: ", ".join(sorted(set(map(str, s))))),
            blocks=("source_block", lambda s: ", ".join(sorted({str(x) for x in s if str(x)}))[:300]),
            classes=("feature_class", lambda s: ", ".join(sorted(set(map(str, s))))),
            wall_score=("wall_score", "mean"),
            door_score=("door_score", "mean"),
            window_score=("window_score", "mean"),
            stairs_score=("stairs_score", "mean"),
            keep_count=("decision", lambda s: int((s == "keep").sum())),
            drop_count=("decision", lambda s: int((s == "drop").sum())),
        )
        .reset_index()
    )
    decisions = normalize_decisions(decisions)
    tab["decision"] = "undecided"
    tab.loc[tab["layer"].isin(decisions["keep"]), "decision"] = "keep"
    tab.loc[tab["layer"].isin(decisions["drop"]), "decision"] = "drop"
    tab["suggested_action"] = "keep"
    tab.loc[tab["drop_count"] > tab["keep_count"], "suggested_action"] = "drop"
    return tab.sort_values(["decision", "n_entities"], ascending=[True, False])


def decisions_from_review_table(review: pd.DataFrame) -> dict[str, list[str]]:
    result = {"keep": [], "drop": [], "undecided": []}
    if review.empty:
        return result
    for _, row in review.iterrows():
        decision = str(row.get("decision", "undecided")).lower()
        if decision not in result:
            decision = "undecided"
        result[decision].append(str(row["layer"]))
    return {k: sorted(set(v)) for k, v in result.items()}
