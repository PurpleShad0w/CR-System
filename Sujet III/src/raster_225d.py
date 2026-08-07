from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageColor, ImageFilter


def _hex_rgb(value: str) -> tuple[int, int, int]:
    return ImageColor.getrgb(value)


def _crop_rgba_to_content(img: Image.Image, alpha_or_ink: np.ndarray, margin: int) -> Image.Image:
    ys, xs = np.where(alpha_or_ink)
    if len(xs) == 0 or len(ys) == 0:
        return img
    w, h = img.size
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(w, int(xs.max()) + margin + 1)
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(h, int(ys.max()) + margin + 1)
    return img.crop((x0, y0, x1, y1))


def _dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    im = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    for _ in range(px):
        im = im.filter(ImageFilter.MaxFilter(3))
    return np.array(im) > 0


def _shear_rgba(img: Image.Image, shear_x: float, scale_y: float, bg=(255, 255, 255, 255)) -> Image.Image:
    if abs(shear_x) < 1e-6 and abs(scale_y - 1.0) < 1e-6:
        return img
    w, h = img.size
    extra_x = int(abs(shear_x) * h) + 8
    new_w = w + extra_x
    new_h = int(h * scale_y) + 8

    # PIL affine maps output -> input. We use an inverse transform.
    # x_in = x_out - shear * y_in. y_in = y_out / scale_y.
    a = 1.0
    b = -shear_x / max(scale_y, 1e-6)
    c = extra_x if shear_x < 0 else 0
    d = 0.0
    e = 1.0 / max(scale_y, 1e-6)
    f = 0.0
    return img.transform((new_w, new_h), Image.Transform.AFFINE, (a, b, -c, d, e, f), resample=Image.Resampling.BICUBIC, fillcolor=bg)


def render_225d_from_image(input_png: str | Path, output_png: str | Path, cfg: dict[str, Any] | None = None) -> Path:
    """Transforme un rendu 2D propre en vue 2.25D subtile.

    Cette fonction ne tente pas de reconstruire de la géométrie 3D. Elle conserve le plan 2D,
    puis ajoute une profondeur légère par décalage progressif de l'encre noire. C'est fait pour
    partir du meilleur rendu v1/v1plus, sans casser les portes, fenêtres et détails.
    """
    cfg = cfg or {}
    input_png = Path(input_png)
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    img_l = Image.open(input_png).convert("L")
    threshold = int(cfg.get("threshold", 245))
    ink = np.array(img_l) < threshold
    ink = _dilate_mask(ink, int(cfg.get("top_dilate_px", 0)))

    h, w = ink.shape
    depth = int(cfg.get("depth_px", 10))
    steps = max(1, int(cfg.get("depth_steps", 8)))
    dx_sign = 1 if float(cfg.get("direction_x", 1)) >= 0 else -1
    dy_sign = 1 if float(cfg.get("direction_y", 1)) >= 0 else -1

    pad = depth + 8
    canvas_h = h + pad * 2
    canvas_w = w + pad * 2
    rgba = np.full((canvas_h, canvas_w, 4), 255, dtype=np.uint8)

    side_rgb = _hex_rgb(str(cfg.get("side_color", "#d8d8d8")))
    shadow_rgb = _hex_rgb(str(cfg.get("contact_shadow_color", "#b8b8b8")))
    top_rgb = _hex_rgb(str(cfg.get("top_line_color", "#050505")))
    a0 = float(cfg.get("side_alpha_start", 0.30))
    a1 = float(cfg.get("side_alpha_end", 0.06))
    contact_alpha = float(cfg.get("contact_shadow_alpha", 0.20))
    top_alpha = float(cfg.get("top_line_alpha", 1.0))

    # Contact shadow, slightly blurred, below all linework.
    shadow = Image.fromarray((ink.astype(np.uint8) * 255), mode="L")
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=max(1.0, depth / 3.5)))
    shadow_arr = np.array(shadow) > 12
    sy = pad + int(depth * dy_sign * 0.75)
    sx = pad + int(depth * dx_sign * 0.75)
    yy, xx = np.where(shadow_arr)
    yy2 = yy + sy
    xx2 = xx + sx
    valid = (yy2 >= 0) & (yy2 < canvas_h) & (xx2 >= 0) & (xx2 < canvas_w)
    rgba[yy2[valid], xx2[valid], :3] = np.array(shadow_rgb, dtype=np.uint8)
    rgba[yy2[valid], xx2[valid], 3] = np.maximum(rgba[yy2[valid], xx2[valid], 3], int(contact_alpha * 255))

    # Progressive side layers. Later layers are lighter/less opaque.
    yy, xx = np.where(ink)
    for i in range(steps, 0, -1):
        t = i / steps
        ox = int(round(depth * dx_sign * t))
        oy = int(round(depth * dy_sign * t))
        alpha = a1 + (a0 - a1) * t
        yy2 = yy + pad + oy
        xx2 = xx + pad + ox
        valid = (yy2 >= 0) & (yy2 < canvas_h) & (xx2 >= 0) & (xx2 < canvas_w)
        rgba[yy2[valid], xx2[valid], :3] = np.array(side_rgb, dtype=np.uint8)
        rgba[yy2[valid], xx2[valid], 3] = np.maximum(rgba[yy2[valid], xx2[valid], 3], int(alpha * 255))

    # Top plan, always black and on top.
    yy2 = yy + pad
    xx2 = xx + pad
    rgba[yy2, xx2, :3] = np.array(top_rgb, dtype=np.uint8)
    rgba[yy2, xx2, 3] = int(top_alpha * 255)

    out = Image.fromarray(rgba, mode="RGBA")

    # Composite over white to avoid alpha issues in downstream viewers.
    bg = Image.new("RGBA", out.size, (255, 255, 255, 255))
    out = Image.alpha_composite(bg, out)

    if bool(cfg.get("apply_subtle_shear", True)):
        out = _shear_rgba(out, float(cfg.get("shear_x", -0.035)), float(cfg.get("scale_y", 0.965)))

    if bool(cfg.get("crop_to_content", True)):
        gray = np.array(out.convert("L"))
        content = gray < 250
        out = _crop_rgba_to_content(out, content, int(cfg.get("crop_margin_px", 32)))

    out.convert("RGB").save(output_png)
    return output_png
