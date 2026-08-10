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

def _wall_mask(mask,cfg):
    er=int(cfg.get('wall_erode_px',2)); di=int(cfg.get('wall_dilate_px',3)); keep=int(cfg.get('wall_keep_px',1))
    return _dilate(_dilate(_erode(mask,er),er+di),keep)

def _runs(vals):
    n=len(vals); i=0
    while i<n:
        if not vals[i]: i+=1; continue
        j=i+1
        while j<n and vals[j]: j+=1
        yield i,j; i=j

def _line_masks(thin,cfg):
    h,w=thin.shape; min_run=int(cfg.get('min_line_run_px',14)); min_diag=int(cfg.get('min_diag_run_px',12))
    mh=np.zeros_like(thin,bool); mv=np.zeros_like(thin,bool); md1=np.zeros_like(thin,bool); md2=np.zeros_like(thin,bool)
    for y in range(h):
        for x0,x1 in _runs(thin[y,:]):
            if x1-x0>=min_run: mh[y,x0:x1]=True
    for x in range(w):
        for y0,y1 in _runs(thin[:,x]):
            if y1-y0>=min_run: mv[y0:y1,x]=True
    # main diagonals: y-x = k
    for k in range(-w+1,h):
        coords=[]; vals=[]
        y0=max(0,k); y1=min(h,w+k)
        for y in range(y0,y1):
            x=y-k; coords.append((y,x)); vals.append(thin[y,x])
        vals=np.array(vals,bool)
        for a,b in _runs(vals):
            if b-a>=min_diag:
                for y,x in coords[a:b]: md1[y,x]=True
    # anti diagonals: y+x = k
    for k in range(h+w-1):
        coords=[]; vals=[]
        y0=max(0,k-(w-1)); y1=min(h-1,k)
        for y in range(y0,y1+1):
            x=k-y; coords.append((y,x)); vals.append(thin[y,x])
        vals=np.array(vals,bool)
        for a,b in _runs(vals):
            if b-a>=min_diag:
                for y,x in coords[a:b]: md2[y,x]=True
    return mh,mv,md1,md2

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
            xs=[p[0] for p in pix]; ys=[p[1] for p in pix]
            comps.append({'pixels':pix,'area':len(pix),'bbox':(min(xs),min(ys),max(xs),max(ys))})
    return comps

def _mesh_remove_mask(thin,wall,cfg):
    mh,mv,md1,md2=_line_masks(thin,cfg); line=mh|mv|md1|md2
    linked=_dilate(line,int(cfg.get('line_mask_dilate_px',2)))
    h,w=thin.shape; remove=np.zeros_like(thin,bool)
    min_w=int(cfg.get('min_mesh_bbox_w_px',36)); min_h=int(cfg.get('min_mesh_bbox_h_px',28)); min_pix=int(cfg.get('min_mesh_line_pixels',220)); min_den=float(cfg.get('min_mesh_density',0.045)); min_or=int(cfg.get('min_mesh_orientations',2)); margin=int(cfg.get('mesh_bbox_margin_px',3))
    for comp in _components(linked):
        x0,y0,x1,y1=comp['bbox']; bw=x1-x0+1; bh=y1-y0+1
        if bw<min_w or bh<min_h: continue
        sl=(slice(y0,y1+1),slice(x0,x1+1))
        lp=int(line[sl].sum()); area=max(bw*bh,1); den=lp/area
        orientations=sum([mh[sl].sum()>0,mv[sl].sum()>0,md1[sl].sum()>0,md2[sl].sum()>0])
        if lp>=min_pix and den>=min_den and orientations>=min_or:
            xa=max(0,x0-margin); xb=min(w,x1+margin+1); ya=max(0,y0-margin); yb=min(h,y1+margin+1)
            # remove thin pixels in mesh bbox, keep thick walls automatically
            remove[ya:yb,xa:xb] |= thin[ya:yb,xa:xb] & ~wall[ya:yb,xa:xb]
    return remove

def _remove_specks(mask,wall,cfg):
    if not bool(cfg.get('remove_specks',True)): return mask
    area_lt=int(cfg.get('speck_area_lt',45)); near=int(cfg.get('speck_near_wall_px',7)); near_wall=_dilate(wall,near)
    keep=mask.copy()
    for comp in _components(mask & ~wall):
        if comp['area']>=area_lt: continue
        touches=False
        for x,y in comp['pixels']:
            if near_wall[y,x]: touches=True; break
        if not touches:
            for x,y in comp['pixels']: keep[y,x]=False
    return keep

def _crop(img,mask,margin):
    ys,xs=np.where(mask)
    if len(xs)==0: return img
    w,h=img.size
    return img.crop((max(0,xs.min()-margin),max(0,ys.min()-margin),min(w,xs.max()+margin+1),min(h,ys.max()+margin+1)))

def clean_meshes(input_png,output_png,cfg=None):
    cfg=cfg or {}; input_png=Path(input_png); output_png=Path(output_png); output_png.parent.mkdir(parents=True,exist_ok=True)
    img=Image.open(input_png).convert('L'); arr=np.array(img); mask=arr<int(cfg.get('threshold',245))
    wall=_wall_mask(mask,cfg); thin=mask & ~wall
    remove=_mesh_remove_mask(thin,wall,cfg)
    cleaned=mask.copy(); cleaned[remove]=False; cleaned=_remove_specks(cleaned,wall,cfg)
    action=str(cfg.get('action','erase')).lower()
    if action=='fade':
        out=arr.copy(); out[mask & ~cleaned]=int(cfg.get('fade_gray',232)); content=out<250
    else:
        out=np.full_like(arr,255); out[cleaned]=arr[cleaned]; content=cleaned
    out_img=Image.fromarray(out,'L').convert('RGB')
    if bool(cfg.get('crop_to_content',True)): out_img=_crop(out_img,content,int(cfg.get('crop_margin_px',34)))
    out_img.save(output_png); return output_png
