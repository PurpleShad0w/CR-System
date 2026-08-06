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


def _contains_any(value: str, keywords: list[str]) -> bool:
    u = value.upper()
    return any(k.upper() in u for k in keywords)


def normalize_decisions(decisions: dict[str, Any] | None) -> dict[str, list[str]]:
    decisions = decisions or {}
    return {
        "keep": sorted(set(map(str, decisions.get("keep", []) or []))),
        "drop": sorted(set(map(str, decisions.get("drop", []) or []))),
        "undecided": sorted(set(map(str, decisions.get("undecided", []) or []))),
    }


def compute_wall_score(df: pd.DataFrame, rules: dict[str, Any] | None = None) -> pd.Series:
    rules = rules or {}
    clf = rules.get("classification", {}) or {}
    keep_keywords = clf.get("keep_layer_keywords", []) or []
    min_wall_length = float(clf.get("min_wall_length", 30.0))
    min_wall_area = float(clf.get("min_wall_area", 20.0))

    if df.empty:
        return pd.Series(dtype=float)

    score = pd.Series(0.0, index=df.index)
    score += df["layer"].fillna("").astype(str).apply(lambda s: 0.30 if _contains_any(s, keep_keywords) else 0.0)
    score += df["entity_type"].isin(["LWPOLYLINE", "POLYLINE"]).astype(float) * 0.20
    score += df["closed"].fillna(False).astype(bool).astype(float) * 0.15
    score += (df["length"].fillna(0.0) >= min_wall_length).astype(float) * 0.20
    score += (df["area"].fillna(0.0) >= min_wall_area).astype(float) * 0.10
    score += ((df["bbox_width"].abs() > 0) & (df["bbox_height"].abs() > 0)).astype(float) * 0.05
    return score.clip(0.0, 1.0)


def classify_entities(df: pd.DataFrame, rules: dict[str, Any] | None, decisions: dict[str, Any] | None) -> pd.DataFrame:
    rules = rules or {}
    decisions = normalize_decisions(decisions)
    out = df.copy()
    out["wall_score"] = compute_wall_score(out, rules)

    drop_types = set(map(str.upper, rules.get("drop_entity_types", []) or []))
    drop_contains = rules.get("drop_layers_contains", []) or []
    drop_small = rules.get("drop_small_entities", {}) or {}
    min_len = float(drop_small.get("min_length", 0.0) or 0.0)
    threshold = float((rules.get("classification", {}) or {}).get("wall_score_threshold", 0.55))

    out["decision"] = "undecided"
    out.loc[out["layer"].isin(decisions["keep"]), "decision"] = "keep"
    out.loc[out["layer"].isin(decisions["drop"]), "decision"] = "drop"

    auto_drop = out["entity_type"].fillna("").astype(str).str.upper().isin(drop_types)
    auto_drop |= out["layer"].fillna("").astype(str).apply(lambda s: _contains_any(s, drop_contains))
    if bool(drop_small.get("enabled", False)):
        auto_drop |= out["length"].fillna(0.0).lt(min_len) & out["area"].fillna(0.0).eq(0.0)
    out.loc[(out["decision"] == "undecided") & auto_drop, "decision"] = "drop"
    out.loc[(out["decision"] == "undecided") & out["wall_score"].ge(threshold), "decision"] = "keep"
    return out


def build_review_table(df: pd.DataFrame, rules: dict[str, Any] | None, decisions: dict[str, Any] | None) -> pd.DataFrame:
    classified = classify_entities(df, rules, decisions)
    if classified.empty:
        return pd.DataFrame(columns=["layer", "decision", "suggested_action", "n_entities", "entity_types", "wall_score_mean"])
    tab = (
        classified.groupby("layer")
        .agg(
            n_entities=("entity_id", "count"),
            entity_types=("entity_type", lambda s: ", ".join(sorted(set(map(str, s))))),
            total_length=("length", "sum"),
            total_area=("area", "sum"),
            wall_score_mean=("wall_score", "mean"),
            keep_count=("decision", lambda s: int((s == "keep").sum())),
            drop_count=("decision", lambda s: int((s == "drop").sum())),
        )
        .reset_index()
    )
    tab["suggested_action"] = "undecided"
    tab.loc[tab["keep_count"] >= tab["drop_count"], "suggested_action"] = "keep"
    tab.loc[tab["drop_count"] > tab["keep_count"], "suggested_action"] = "drop"
    decisions = normalize_decisions(decisions)
    tab["decision"] = "undecided"
    tab.loc[tab["layer"].isin(decisions["keep"]), "decision"] = "keep"
    tab.loc[tab["layer"].isin(decisions["drop"]), "decision"] = "drop"
    return tab.sort_values(["decision", "n_entities", "wall_score_mean"], ascending=[True, False, False])


def decisions_from_review_table(review: pd.DataFrame) -> dict[str, list[str]]:
    result = {"keep": [], "drop": [], "undecided": []}
    if review.empty:
        return result
    for _, row in review.iterrows():
        decision = str(row.get("decision", "undecided")).lower()
        layer = str(row["layer"])
        if decision not in result:
            decision = "undecided"
        result[decision].append(layer)
    return {k: sorted(set(v)) for k, v in result.items()}
