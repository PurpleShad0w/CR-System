from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd, yaml
def load_yaml(path, default=None):
    p=Path(path)
    if not p.exists(): return default
    with p.open("r",encoding="utf-8") as f: return yaml.safe_load(f) or default
def _contains_any(v, kws):
    u=str(v or "").upper(); return any(str(k).upper() in u for k in kws)
def _blob(df):
    cols=[c for c in ["layer","source_block","block_name","entity_type"] if c in df.columns]
    out=pd.Series("",index=df.index)
    for c in cols: out=out+" "+df[c].fillna("").astype(str)
    return out
def _kw(blob,kws,val): return blob.apply(lambda x: val if _contains_any(x,kws) else 0.0)
def normalize_decisions(decisions):
    decisions=decisions or {}; return {"keep":sorted(set(map(str,decisions.get("keep",[]) or []))),"drop":sorted(set(map(str,decisions.get("drop",[]) or []))),"undecided":sorted(set(map(str,decisions.get("undecided",[]) or [])))}
def classify_physical(df:pd.DataFrame,rules:dict[str,Any]|None,decisions:dict[str,Any]|None)->pd.DataFrame:
    rules=rules or {}; decisions=normalize_decisions(decisions); clf=rules.get("classification",{}) or {}; out=df.copy()
    if out.empty: out["feature_class"]=[]; return out
    blob=_blob(out); etype=out["entity_type"].fillna("").astype(str).str.upper(); length=out["length"].fillna(0.0); width=out["bbox_width"].abs().fillna(0.0); height=out["bbox_height"].abs().fillna(0.0); min_dim=width.combine(height,min).replace(0,1); max_dim=width.combine(height,max); aspect=max_dim/min_dim; arc_angle=out.get("arc_angle",pd.Series(0,index=out.index)).fillna(0.0)
    out["hard_noise"]=etype.isin(set(map(str.upper,rules.get("hard_drop_entity_types",[]) or []))) | blob.apply(lambda x:_contains_any(x,rules.get("hard_drop_layer_keywords",[]) or []))
    out["wall_score"]=_kw(blob,clf.get("wall_keywords",[]),0.42)+etype.isin(["LINE","LWPOLYLINE","POLYLINE"]).astype(float)*0.16+(length>=float(clf.get("min_wall_length",10.0))).astype(float)*0.22+((width>0)&(height>0)).astype(float)*0.08+out["closed"].fillna(False).astype(bool).astype(float)*0.06
    out["door_score"]=_kw(blob,clf.get("door_keywords",[]),0.48)+etype.isin(["ARC","LINE","LWPOLYLINE","POLYLINE","INSERT"]).astype(float)*0.12+arc_angle.between(35,115).astype(float)*0.25+(length>=float(clf.get("min_feature_length",2.0))).astype(float)*0.1+((aspect>=1.2)&(aspect<=18)).astype(float)*0.05
    out["window_score"]=_kw(blob,clf.get("window_keywords",[]),0.48)+etype.isin(["LINE","LWPOLYLINE","POLYLINE","INSERT"]).astype(float)*0.14+(length>=float(clf.get("min_feature_length",2.0))).astype(float)*0.1+(aspect>=2.0).astype(float)*0.15
    out["stairs_score"]=_kw(blob,clf.get("stairs_keywords",[]),0.55)+etype.isin(["LINE","LWPOLYLINE","POLYLINE","INSERT"]).astype(float)*0.15+(length>=float(clf.get("min_feature_length",2.0))).astype(float)*0.1
    thr={"wall":float(clf.get("wall_score_threshold",0.38)),"door":float(clf.get("door_score_threshold",0.32)),"window":float(clf.get("window_score_threshold",0.30)),"stairs":float(clf.get("stairs_score_threshold",0.38))}
    out["feature_class"]="detail"
    for idx,row in out.iterrows():
        best=max(["wall_score","door_score","window_score","stairs_score"], key=lambda c:float(row.get(c,0)))
        label=best.replace("_score","")
        if float(row.get(best,0))>=thr[label]: out.at[idx,"feature_class"]=label
    out.loc[out["hard_noise"],"feature_class"]="noise"
    out["decision"]="undecided"; out.loc[out["layer"].isin(decisions["keep"]),"decision"]="keep"; out.loc[out["layer"].isin(decisions["drop"]),"decision"]="drop"
    return out
