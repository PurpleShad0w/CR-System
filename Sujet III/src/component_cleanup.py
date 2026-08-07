from __future__ import annotations
from collections import defaultdict, deque
from pathlib import Path
from typing import Any
import numpy as np, pandas as pd
from PIL import Image
from .classify_physical import classify_physical

def _points_ok(p): return isinstance(p,list) and len(p)>=2
def _bbox(df): return float(df.bbox_min_x.min()),float(df.bbox_min_y.min()),float(df.bbox_max_x.max()),float(df.bbox_max_y.max())
def _inflate(b,ratio):
    x0,y0,x1,y1=b; m=max(x1-x0,y1-y0,1.0)*ratio; return x0-m,y0-m,x1+m,y1+m
def _intersects(row,b):
    x0,y0,x1,y1=b
    return not (row.bbox_max_x<x0 or row.bbox_min_x>x1 or row.bbox_max_y<y0 or row.bbox_min_y>y1)
def _cells(row,minx,miny,sx,sy,g,dilate=1):
    cells=set(); pts=row.points if _points_ok(row.points) else [((row.bbox_min_x+row.bbox_max_x)/2,(row.bbox_min_y+row.bbox_max_y)/2)]
    for x,y in pts:
        ix=max(0,min(g-1,int((x-minx)*sx))); iy=max(0,min(g-1,int((y-miny)*sy)))
        for dx in range(-dilate,dilate+1):
            for dy in range(-dilate,dilate+1):
                nx,ny=ix+dx,iy+dy
                if 0<=nx<g and 0<=ny<g: cells.add((nx,ny))
    return cells
def _components(cell_to_rows, connectivity=8):
    all_cells=set(cell_to_rows); remain=set(all_cells); comps=[]; neigh=[(1,0),(-1,0),(0,1),(0,-1)] + ([(1,1),(1,-1),(-1,1),(-1,-1)] if connectivity==8 else [])
    while remain:
        s=remain.pop(); q=deque([s]); cc={s}
        while q:
            x,y=q.popleft()
            for dx,dy in neigh:
                nb=(x+dx,y+dy)
                if nb in remain:
                    remain.remove(nb); cc.add(nb); q.append(nb)
        rows=set()
        for c in cc: rows.update(cell_to_rows[c])
        comps.append(rows)
    return comps
def _score_component(g,cfg):
    if g.empty: return -1e9
    wall=g.feature_class.eq("wall"); opening=g.feature_class.isin(["door","window","stairs"]); detail=g.feature_class.eq("detail")
    x0,y0,x1,y1=_bbox(g); area=max((x1-x0)*(y1-y0),1.0); length=float(g.length.fillna(0).sum())
    useful=float(g.loc[wall,"length"].sum())*float(cfg.get("score_wall_weight",2.0))+float(g.loc[opening,"length"].sum())*float(cfg.get("score_opening_weight",1.4))+float(g.loc[detail,"length"].sum())*float(cfg.get("score_detail_weight",0.35))
    density=(length/area)**0.5
    micro=((g.length.fillna(0)<2.0)|(g.bbox_diag.fillna(0)<2.0)).mean()
    return useful + density*float(cfg.get("density_penalty_weight",0.45))*1000 - micro*float(cfg.get("micro_entity_penalty_weight",0.35))*len(g)
def clean_component_entities(df,rules=None,decisions=None):
    rules=rules or {}; cfg=rules.get("component_cleanup",{}) or {}; out=classify_physical(df,rules,decisions)
    out=out[out.points.apply(_points_ok)].copy(); out=out[~out.feature_class.eq("noise")].copy()
    if out.empty: return out
    out=out[(out.length.fillna(0)>=float(cfg.get("min_render_length",0.8))) & (out.bbox_diag.fillna(0)>=float(cfg.get("min_render_bbox_diag",0.5)))].copy()
    if out.empty: return out
    gsize=int(cfg.get("grid_size",650)); minx,miny,maxx,maxy=_bbox(out); sx=(gsize-1)/max(maxx-minx,1.0); sy=(gsize-1)/max(maxy-miny,1.0); dil=int(cfg.get("entity_cell_dilation",1))
    cell_to_rows=defaultdict(set)
    for i,row in out.iterrows():
        for c in _cells(row,minx,miny,sx,sy,gsize,dil): cell_to_rows[c].add(i)
    comps=_components(cell_to_rows,int(cfg.get("connectivity",8)))
    scored=[]
    for rows in comps:
        gg=out.loc[list(rows)]; scored.append((_score_component(gg,cfg),rows))
    scored.sort(reverse=True,key=lambda x:x[0])
    best_score,best_rows=scored[0]
    keep_rows=set(best_rows)
    # keep secondaries only if strong enough and close in size/score, avoids remote legends/barcodes
    ratio=float(cfg.get("min_component_score_ratio",0.22))
    for score,rows in scored[1:]:
        if best_score>0 and score/best_score>=ratio: keep_rows.update(rows)
    main=out.loc[list(keep_rows)].copy()
    main_bbox=_inflate(_bbox(main), float(cfg.get("component_margin_cells",7))/gsize)
    if bool(cfg.get("rescue_near_main_component",True)):
        rb=_inflate(_bbox(main), float(cfg.get("rescue_buffer_ratio",0.055)))
        et=out.entity_type.fillna("").astype(str).str.upper(); rescue_types=set(map(str.upper,cfg.get("rescue_entity_types",[]) or []))
        max_dim=max(_bbox(main)[2]-_bbox(main)[0],_bbox(main)[3]-_bbox(main)[1],1.0); max_len=max_dim*float(cfg.get("rescue_max_length_ratio",0.18))
        rescue=out.apply(lambda r:_intersects(r,rb),axis=1) & et.isin(rescue_types) & (out.length.fillna(0)>=float(cfg.get("rescue_min_length",1.2))) & (out.bbox_diag.fillna(0)>=float(cfg.get("rescue_min_bbox_diag",0.8))) & (out.length.fillna(0)<=max_len)
        main=pd.concat([main,out[rescue]],axis=0).drop_duplicates("row_id")
    main["render_class"]=main.feature_class.where(main.feature_class.isin(["wall","door","window","stairs"]),"detail")
    return main

def raster_cleanup(path,rules=None):
    rules=rules or {}; cfg=rules.get("raster_cleanup",{}) or {}
    if not bool(cfg.get("enabled",True)): return Path(path)
    path=Path(path); img=Image.open(path).convert("L"); arr=np.array(img); ink=arr<245; h,w=ink.shape; vis=np.zeros_like(ink,bool); comps=[]
    for y in range(h):
        for x in np.where(ink[y]&~vis[y])[0]:
            if vis[y,x]: continue
            q=deque([(x,y)]); vis[y,x]=True; pix=[]
            while q:
                cx,cy=q.popleft(); pix.append((cx,cy))
                for nx in (cx-1,cx,cx+1):
                    for ny in (cy-1,cy,cy+1):
                        if 0<=nx<w and 0<=ny<h and not vis[ny,nx] and ink[ny,nx]: vis[ny,nx]=True; q.append((nx,ny))
            comps.append(pix)
    keep=np.zeros_like(ink,bool); remove_lt=int(cfg.get("remove_components_area_lt",12)); keep_largest=bool(cfg.get("keep_largest_component",False)); largest=max(comps,key=len) if comps else []
    for comp in comps:
        if len(comp)>=remove_lt and (not keep_largest or comp is largest):
            for x,y in comp: keep[y,x]=True
    out=np.full_like(arr,255); out[keep]=arr[keep]
    if bool(cfg.get("crop_to_content",True)) and keep.any():
        ys,xs=np.where(keep); m=int(cfg.get("crop_margin_px",30)); out=out[max(0,ys.min()-m):min(h,ys.max()+m+1), max(0,xs.min()-m):min(w,xs.max()+m+1)]
    Image.fromarray(out).save(path); return path
