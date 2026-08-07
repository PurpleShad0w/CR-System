from __future__ import annotations
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt, pandas as pd
from .component_cleanup import clean_component_entities, raster_cleanup

def _iter(points,closed):
    pts=points+[points[0]] if closed and points and points[0]!=points[-1] else points
    for a,b in zip(pts[:-1],pts[1:]): yield a,b
def _style(cls,cfg):
    if bool(cfg.get("draw_debug_colors",False)):
        return {"wall":"black","door":"#d95f02","window":"#1b9e77","stairs":"#7570b3","detail":"#666"}.get(cls,"#333"),0.75,"-",5
    if cls=="wall": return cfg.get("wall_color","#050505"),float(cfg.get("wall_linewidth",1.18)),"-",3
    if cls=="door": return cfg.get("detail_color","#151515"),float(cfg.get("door_linewidth",0.68)),"-",7
    if cls=="window": return cfg.get("detail_color","#151515"),float(cfg.get("window_linewidth",0.62)),"-",8
    if cls=="stairs": return cfg.get("detail_color","#151515"),float(cfg.get("stairs_linewidth",0.55)),"-",6
    return cfg.get("detail_color","#151515"),float(cfg.get("detail_linewidth",0.48)),"-",4
def render_clean_plan_v5(df:pd.DataFrame,out_path:str|Path,rules:dict[str,Any]|None=None,decisions:dict[str,Any]|None=None)->Path:
    rules=rules or {}; cfg=rules.get("render",{}) or {}; out_path=Path(out_path); out_path.parent.mkdir(parents=True,exist_ok=True); clean=clean_component_entities(df,rules,decisions)
    fig,ax=plt.subplots(figsize=(float(cfg.get("figure_size",12)),float(cfg.get("figure_size",12))),dpi=int(cfg.get("dpi",300))); fig.patch.set_facecolor(cfg.get("background","white")); ax.set_facecolor(cfg.get("background","white")); xs=[]; ys=[]
    order={"detail":1,"wall":2,"stairs":3,"door":4,"window":5}
    if not clean.empty: clean=clean.assign(_o=clean.render_class.map(order).fillna(1)).sort_values("_o")
    for _,r in clean.iterrows():
        color,lw,ls,z=_style(str(r.get("render_class","detail")),cfg)
        for (x1,y1),(x2,y2) in _iter(r.points,bool(r.get("closed",False))):
            ax.plot([x1,x2],[y1,y2],color=color,linewidth=lw,linestyle=ls,zorder=z,solid_capstyle="round"); xs += [x1,x2]; ys += [y1,y2]
    if xs:
        minx,maxx,miny,maxy=min(xs),max(xs),min(ys),max(ys); dx=max(maxx-minx,1.0); dy=max(maxy-miny,1.0); m=float(cfg.get("margins_ratio",0.018)); ax.set_xlim(minx-dx*m,maxx+dx*m); ax.set_ylim(miny-dy*m,maxy+dy*m)
    ax.set_aspect("equal",adjustable="box"); ax.axis("off"); fig.savefig(out_path,bbox_inches="tight",pad_inches=0.0,facecolor=fig.get_facecolor()); plt.close(fig); raster_cleanup(out_path,rules); return out_path
