from pathlib import Path
import numpy as np
from PIL import Image, ImageColor, ImageFilter

def _rgb(v): return ImageColor.getrgb(v)
def _crop(img,content,margin):
    ys,xs=np.where(content)
    if len(xs)==0: return img
    w,h=img.size
    return img.crop((max(0,xs.min()-margin),max(0,ys.min()-margin),min(w,xs.max()+margin+1),min(h,ys.max()+margin+1)))
def render_shadow225(input_png,output_png,cfg=None):
    cfg=cfg or {}; input_png=Path(input_png); output_png=Path(output_png); output_png.parent.mkdir(parents=True,exist_ok=True)
    img=Image.open(input_png).convert('L'); ink=np.array(img)<int(cfg.get('threshold',245)); h,w=ink.shape
    depth=int(cfg.get('depth_px',10)); steps=max(1,int(cfg.get('shadow_steps',10))); sx=1 if float(cfg.get('direction_x',1))>=0 else -1; sy=1 if float(cfg.get('direction_y',1))>=0 else -1
    pad=depth+12; H,W=h+pad*2,w+pad*2; canvas=np.full((H,W,3),255,dtype=np.float32)
    shadow_rgb=np.array(_rgb(str(cfg.get('shadow_color','#b9b9b9'))),dtype=np.float32); top_rgb=np.array(_rgb(str(cfg.get('top_line_color','#050505'))),dtype=np.float32)
    a0=float(cfg.get('shadow_alpha_start',0.30)); a1=float(cfg.get('shadow_alpha_end',0.04))
    sm=np.array(Image.fromarray((ink.astype(np.uint8)*255),'L').filter(ImageFilter.GaussianBlur(float(cfg.get('blur_radius',1.15)))))/255.0
    yy,xx=np.where(sm>0.03); vals=sm[yy,xx]
    for i in range(steps,0,-1):
        t=i/steps; ox=int(round(depth*sx*t)); oy=int(round(depth*sy*t)); alpha=(a1+(a0-a1)*t)*vals
        y2=yy+pad+oy; x2=xx+pad+ox; valid=(y2>=0)&(y2<H)&(x2>=0)&(x2<W); a=alpha[valid][:,None]
        canvas[y2[valid],x2[valid],:]=canvas[y2[valid],x2[valid],:]*(1-a)+shadow_rgb*a
    yy,xx=np.where(ink); canvas[yy+pad,xx+pad,:]=top_rgb
    out=Image.fromarray(np.clip(canvas,0,255).astype(np.uint8),'RGB')
    if bool(cfg.get('crop_to_content',True)): out=_crop(out,np.array(out.convert('L'))<250,int(cfg.get('crop_margin_px',34)))
    out.save(output_png); return output_png
