from collections import deque
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

def _dilate(mask,r):
    if r<=0: return mask.copy()
    return np.array(Image.fromarray((mask.astype(np.uint8)*255),'L').filter(ImageFilter.MaxFilter(r*2+1)))>0

def _erode(mask,r):
    if r<=0: return mask.copy()
    return np.array(Image.fromarray((mask.astype(np.uint8)*255),'L').filter(ImageFilter.MinFilter(r*2+1)))>0

def _thick_mask(mask,cfg):
    return _dilate(_erode(mask,int(cfg.get('wall_erode_px',2))),int(cfg.get('wall_dilate_px',3))+int(cfg.get('wall_erode_px',2)))

def _integral(mask):
    return np.pad(mask.astype(np.int32).cumsum(0).cumsum(1),((1,0),(1,0)),mode='constant')

def _rect(ii,y0,x0,y1,x1):
    return int(ii[y1,x1]-ii[y0,x1]-ii[y1,x0]+ii[y0,x0])

def _dense_mask(mask,cfg):
    if not bool(cfg.get('protect_dense_zones',True)): return np.zeros_like(mask,bool)
    h,w=mask.shape; r=int(cfg.get('dense_radius_px',9)); thr=float(cfg.get('dense_threshold',0.30)); ii=_integral(mask); out=np.zeros_like(mask,bool)
    # sampled block acceleration: compute per pixel is still ok-ish but use slices by rows; image sizes acceptable here.
    for y in range(h):
        y0=max(0,y-r); y1=min(h,y+r+1)
        for x in range(w):
            x0=max(0,x-r); x1=min(w,x+r+1); area=(y1-y0)*(x1-x0)
            if _rect(ii,y0,x0,y1,x1)/max(area,1)>=thr: out[y,x]=True
    return out

def _candidate_indices(mask,axis,cfg):
    h,w=mask.shape; n=h if axis=='h' else w; dim=w if axis=='h' else h
    min_span=int(dim*float(cfg.get('min_span_ratio',0.18))); min_pix=int(cfg.get('min_total_pixels',18)); lo=float(cfg.get('min_pixels_ratio_in_span',0.18)); hi=float(cfg.get('max_pixels_ratio_in_span',0.80))
    out=[]
    for i in range(n):
        line=mask[i,:] if axis=='h' else mask[:,i]
        xs=np.where(line)[0]
        if len(xs)<min_pix: continue
        span=int(xs.max()-xs.min()+1)
        if span<min_span: continue
        ratio=len(xs)/max(span,1)
        if lo<=ratio<=hi: out.append(i)
    return out

def _expand_indices(indices,max_i,neighbor):
    s=set()
    for i in indices:
        for j in range(i-neighbor,i+neighbor+1):
            if 0<=j<max_i: s.add(j)
    return sorted(s)

def _remove_aligned(mask,thick,dense,cfg):
    h,w=mask.shape; thin=mask & ~thick; remove=np.zeros_like(mask,bool); band=int(cfg.get('remove_band_halfwidth_px',1)); neigh=int(cfg.get('group_neighbor_px',2))
    if bool(cfg.get('remove_horizontal_aligned',True)):
        rows=_expand_indices(_candidate_indices(thin,'h',cfg),h,neigh)
        for y in rows:
            y0=max(0,y-band); y1=min(h,y+band+1); remove[y0:y1,:] |= thin[y0:y1,:] & ~dense[y0:y1,:]
    if bool(cfg.get('remove_vertical_aligned',True)):
        cols=_expand_indices(_candidate_indices(thin,'v',cfg),w,neigh)
        for x in cols:
            x0=max(0,x-band); x1=min(w,x+band+1); remove[:,x0:x1] |= thin[:,x0:x1] & ~dense[:,x0:x1]
    cleaned=mask.copy(); cleaned[remove]=False; return cleaned

def _components(mask):
    h,w=mask.shape; vis=np.zeros_like(mask,bool); comps=[]
    for y in range(h):
        for x in np.where(mask[y]&~vis[y])[0]:
            if vis[y,x] or not mask[y,x]: continue
            q=deque([(x,y)]); vis[y,x]=True; pix=[]
            while q:
                cx,cy=q.popleft(); pix.append((cx,cy))
                for nx in (cx-1,cx,cx+1):
                    for ny in (cy-1,cy,cy+1):
                        if 0<=nx<w and 0<=ny<h and mask[ny,nx] and not vis[ny,nx]: vis[ny,nx]=True; q.append((nx,ny))
            comps.append(pix)
    return comps

def _remove_tiny(mask,min_area):
    keep=np.zeros_like(mask,bool)
    for comp in _components(mask):
        if len(comp)>=min_area:
            for x,y in comp: keep[y,x]=True
    return keep

def _crop(img,mask,margin):
    ys,xs=np.where(mask)
    if len(xs)==0: return img
    w,h=img.size
    return img.crop((max(0,xs.min()-margin),max(0,ys.min()-margin),min(w,xs.max()+margin+1),min(h,ys.max()+margin+1)))

def clean_projection_lines(input_png,output_png,cfg=None):
    cfg=cfg or {}; input_png=Path(input_png); output_png=Path(output_png); output_png.parent.mkdir(parents=True,exist_ok=True)
    img=Image.open(input_png).convert('L'); arr=np.array(img); mask=arr<int(cfg.get('threshold',245))
    thick=_thick_mask(mask,cfg); dense=_dense_mask(mask,cfg); cleaned=_remove_aligned(mask,thick,dense,cfg)
    if bool(cfg.get('remove_tiny_components',True)): cleaned=_remove_tiny(cleaned,int(cfg.get('tiny_component_area_lt',8)))
    action=str(cfg.get('action','erase')).lower()
    if action=='fade':
        out=arr.copy(); out[mask & ~cleaned]=int(cfg.get('fade_gray',232)); content=out<250
    else:
        out=np.full_like(arr,255); out[cleaned]=arr[cleaned]; content=cleaned
    out_img=Image.fromarray(out,'L').convert('RGB')
    if bool(cfg.get('crop_to_content',True)): out_img=_crop(out_img,content,int(cfg.get('crop_margin_px',34)))
    out_img.save(output_png); return output_png
