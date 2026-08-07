from __future__ import annotations
import math, subprocess, tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import ezdxf, numpy as np, pandas as pd
LINEAR_TYPES={"LINE","LWPOLYLINE","POLYLINE","ARC","CIRCLE","ELLIPSE","SPLINE"}
TEXTUAL_TYPES={"TEXT","MTEXT","DIMENSION","LEADER","MULTILEADER","TABLE"}
@dataclass
class ConversionConfig:
    backend:str="oda"; exe_path:str=""; version:str="ACAD2018"
def _safe_float(v, default=0.0):
    try: return default if v is None else float(v)
    except Exception: return default
def _xy(p): return (_safe_float(p[0]), _safe_float(p[1]))
def _bbox(points):
    if not points: return (0.0,0.0,0.0,0.0)
    xs=[p[0] for p in points]; ys=[p[1] for p in points]
    return (min(xs),min(ys),max(xs),max(ys))
def _length(points, closed=False):
    if len(points)<2: return 0.0
    pts=points+[points[0]] if closed and points[0]!=points[-1] else points
    return float(sum(math.dist(a,b) for a,b in zip(pts[:-1],pts[1:])))
def _area(points, closed=False):
    if len(points)<3: return 0.0
    pts=points+[points[0]] if points[0]!=points[-1] else points
    if not closed and points[0]!=points[-1]: return 0.0
    return abs(sum(x1*y2-x2*y1 for (x1,y1),(x2,y2) in zip(pts[:-1],pts[1:])))/2.0
def _arc(center,radius,start,end,n=40):
    if end<start: end+=360.0
    a=np.linspace(math.radians(start),math.radians(end),max(10,n)); cx,cy=_xy(center)
    return [(cx+radius*math.cos(t),cy+radius*math.sin(t)) for t in a]
def _circle(center,radius,n=96):
    a=np.linspace(0,2*math.pi,n,endpoint=True); cx,cy=_xy(center)
    return [(cx+radius*math.cos(t),cy+radius*math.sin(t)) for t in a]
def _entity_points(e):
    t=e.dxftype()
    try:
        if t=="LINE": return [_xy(e.dxf.start),_xy(e.dxf.end)],False
        if t=="LWPOLYLINE": return [(float(x),float(y)) for x,y,*_ in e.get_points("xy")], bool(e.closed)
        if t=="POLYLINE": return [_xy(v.dxf.location) for v in e.vertices], bool(getattr(e,"is_closed",False))
        if t=="ARC": return _arc(e.dxf.center,float(e.dxf.radius),float(e.dxf.start_angle),float(e.dxf.end_angle)), False
        if t=="CIRCLE": return _circle(e.dxf.center,float(e.dxf.radius)), True
        if t=="ELLIPSE": return [_xy(p) for p in e.flattening(distance=1.5)], bool(getattr(e,"closed",False))
        if t=="SPLINE": return [_xy(p) for p in e.flattening(distance=1.5)], False
    except Exception: return [],False
    return [],False
def _extra(e):
    t=e.dxftype(); d={}
    try:
        if t=="ARC":
            s=float(e.dxf.start_angle); en=float(e.dxf.end_angle); d["arc_radius"]=float(e.dxf.radius); d["arc_angle"]=abs(en-s)%360
        elif t=="CIRCLE": d["arc_radius"]=float(e.dxf.radius); d["arc_angle"]=360.0
        elif t=="INSERT": d["block_name"]=str(getattr(e.dxf,"name",""))
    except Exception: pass
    return d
def _record(e,i,source="modelspace",source_block=""):
    layer=getattr(e.dxf,"layer","0") if hasattr(e,"dxf") else "0"; typ=e.dxftype()
    pts,closed=_entity_points(e); x0,y0,x1,y1=_bbox(pts); length=_length(pts,closed); area=_area(pts,closed)
    rec={"row_id":i,"entity_id":str(getattr(e.dxf,"handle",None) or f"entity_{i}"),"parent_id":str(getattr(e.dxf,"owner","") or ""),"source":source,"source_block":source_block,"layer":str(layer),"entity_type":typ,"points":pts,"closed":bool(closed),"n_points":len(pts),"bbox_min_x":x0,"bbox_min_y":y0,"bbox_max_x":x1,"bbox_max_y":y1,"bbox_width":x1-x0,"bbox_height":y1-y0,"bbox_diag":math.hypot(x1-x0,y1-y0),"length":length,"area":area,"color":getattr(e.dxf,"color",None) if hasattr(e,"dxf") else None,"lineweight":getattr(e.dxf,"lineweight",None) if hasattr(e,"dxf") else None,"is_textual":typ in TEXTUAL_TYPES,"is_linear":typ in LINEAR_TYPES}
    rec.update(_extra(e)); return rec
def convert_dwg_to_dxf(dwg_path, out_dir, cfg=None):
    cfg=cfg or ConversionConfig(); dwg_path=Path(dwg_path); out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    if dwg_path.suffix.lower()==".dxf": return dwg_path
    if not cfg.exe_path or not Path(cfg.exe_path).exists(): raise FileNotFoundError("Convertisseur ODA introuvable")
    subprocess.run([cfg.exe_path,str(dwg_path.parent),str(out_dir),cfg.version,"DXF","0","1",str(dwg_path.name)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    c=list(out_dir.glob(dwg_path.stem+"*.dxf"))
    if not c: raise FileNotFoundError(f"Aucun DXF généré pour {dwg_path}")
    return c[0]
def extract_entities_df(path, converter=None, include_blocks=True):
    path=Path(path)
    if path.suffix.lower()==".dwg": path=convert_dwg_to_dxf(path, Path(tempfile.mkdtemp(prefix="sujet3_dwg_")), ConversionConfig(**(converter or {})))
    doc=ezdxf.readfile(path); rec=[]
    for e in doc.modelspace():
        if e.dxftype()=="INSERT" and include_blocks:
            bn=str(getattr(e.dxf,"name","")); rec.append(_record(e,len(rec),"insert",bn))
            try:
                for v in e.virtual_entities(): rec.append(_record(v,len(rec),"block_virtual",bn))
            except Exception: pass
        else: rec.append(_record(e,len(rec)))
    df=pd.DataFrame(rec)
    for col in ["arc_radius","arc_angle","block_name"]:
        if col not in df.columns: df[col]=None
    return df
