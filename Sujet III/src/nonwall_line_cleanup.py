from __future__ import annotations
from collections import deque
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image, ImageFilter

def _runs_1d(vals):
    n=len(vals); i=0
    while i<n:
        if not vals[i]: i+=1; continue
        j=i+1
        while j<n and vals[j]: j+=1
        yield i,j; i=j

def _dilate(mask,radius):
    if radius<=0: return mask.copy()
    im=Image.fromarray((mask.astype(np.uint8)*255),'L').filter(ImageFilter.MaxFilter(radius*2+1))
    return np.array(im)>0

def _erode(mask,radius):
    if radius<=0: return mask.copy()
    im=Image.fromarray((mask.astype(np.uint8)*255),'L').filter(ImageFilter.MinFilter(radius*2+1))
    return np.array(im)>0

def _open_thick(mask,erode_radius,dilate_radius):
    return _dilate(_erode(mask,erode_radius), erode_radius+dilate_radius)

def _integral(mask):
    return np.pad(mask.astype(np.int32).cumsum(0).cumsum(1), ((1,0),(1,0)), mode='constant')

def _rect_sum(ii,y0,x0,y1,x1):
    return int(ii[y1,x1]-ii[y0,x1]-ii[y1,x0]+ii[y0,x0])

def _components(mask):
    h,w=mask.shape; visited=np.zeros_like(mask,bool); comps=[]
    for y in range(h):
        for x in np.where(mask[y] & ~visited[y])[0]:
            if visited[y,x] or not mask[y,x]: continue
            q=deque([(x,y)]); visited[y,x]=True; pix=[]
            while q:
                cx,cy=q.popleft(); pix.append((cx,cy))
                for nx in (cx-1,cx,cx+1):
                    for ny in (cy-1,cy,cy+1):
                        if 0<=nx<w and 0<=ny<h and mask[ny,nx] and not visited[ny,nx]:
                            visited[ny,nx]=True; q.append((nx,ny))
            comps.append(pix)
    return comps

def _remove_tiny(mask,min_area):
    if min_area<=0: return mask
    keep=np.zeros_like(mask,bool)
    for comp in _components(mask):
        if len(comp)>=min_area:
            for x,y in comp: keep[y,x]=True
    return keep

def _crop(img,mask,margin):
    ys,xs=np.where(mask)
    if len(xs)==0: return img
    w,h=img.size
    return img.crop((max(0,xs.min()-margin), max(0,ys.min()-margin), min(w,xs.max()+margin+1), min(h,ys.max()+margin+1)))

def _remove_runs(mask,wall_mask,cfg,axis):
    h,w=mask.shape; max_dim=max(h,w)
    min_run=max(int(cfg.get('min_run_length_px',22)), int(max_dim*float(cfg.get('min_run_length_ratio',0.010))))
    strip_r=int(cfg.get('strip_radius_px',3)); dense_r=int(cfg.get('dense_zone_radius_px',10))
    max_strip=float(cfg.get('max_strip_ink_density',0.42)); dense_min=float(cfg.get('dense_zone_min_density',0.18))
    ii=_integral(mask); candidate=mask & ~wall_mask; remove=np.zeros_like(mask,bool)
    if axis=='h':
        for y in range(h):
            for x0,x1 in _runs_1d(candidate[y,:]):
                if x1-x0<min_run: continue
                ya=max(0,y-strip_r); yb=min(h,y+strip_r+1); area=max((yb-ya)*(x1-x0),1)
                strip=_rect_sum(ii,ya,x0,yb,x1)/area
                yd0=max(0,y-dense_r); yd1=min(h,y+dense_r+1); xd0=max(0,x0-dense_r); xd1=min(w,x1+dense_r)
                darea=max((yd1-yd0)*(xd1-xd0),1); dense=_rect_sum(ii,yd0,xd0,yd1,xd1)/darea
                if strip<=max_strip and dense<dense_min: remove[y,x0:x1]=True
    else:
        for x in range(w):
            for y0,y1 in _runs_1d(candidate[:,x]):
                if y1-y0<min_run: continue
                xa=max(0,x-strip_r); xb=min(w,x+strip_r+1); area=max((y1-y0)*(xb-xa),1)
                strip=_rect_sum(ii,y0,xa,y1,xb)/area
                yd0=max(0,y0-dense_r); yd1=min(h,y1+dense_r); xd0=max(0,x-dense_r); xd1=min(w,x+dense_r+1)
                darea=max((yd1-yd0)*(xd1-xd0),1); dense=_rect_sum(ii,yd0,xd0,yd1,xd1)/darea
                if strip<=max_strip and dense<dense_min: remove[y0:y1,x]=True
    out=mask.copy(); out[remove]=False; return out

def clean_nonwall_lines(input_png, output_png, cfg=None):
    cfg=cfg or {}; input_png=Path(input_png); output_png=Path(output_png); output_png.parent.mkdir(parents=True,exist_ok=True)
    img=Image.open(input_png).convert('L'); arr=np.array(img); mask=arr<int(cfg.get('threshold',245))
    wall_mask=_open_thick(mask,int(cfg.get('wall_protect_radius_px',2)),int(cfg.get('wall_protect_dilate_px',2)))
    cleaned=mask.copy()
    if bool(cfg.get('remove_horizontal_runs',True)): cleaned=_remove_runs(cleaned,wall_mask,cfg,'h')
    if bool(cfg.get('remove_vertical_runs',True)): cleaned=_remove_runs(cleaned,wall_mask,cfg,'v')
    if bool(cfg.get('remove_tiny_components',True)): cleaned=_remove_tiny(cleaned,int(cfg.get('tiny_component_area_lt',8)))
    action=str(cfg.get('action','erase')).lower()
    if action=='fade':
        out=arr.copy(); out[mask & ~cleaned]=int(cfg.get('fade_gray',228)); content=out<250
    else:
        out=np.full_like(arr,255); out[cleaned]=arr[cleaned]; content=cleaned
    out_img=Image.fromarray(out,'L').convert('RGB')
    if bool(cfg.get('crop_to_content',True)): out_img=_crop(out_img,content,int(cfg.get('crop_margin_px',34)))
    out_img.save(output_png); return output_png
