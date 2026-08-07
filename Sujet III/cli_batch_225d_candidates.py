from __future__ import annotations

import argparse
from pathlib import Path

from src.yaml_utils import load_yaml
from src.raster_225d import render_225d_from_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Génère plusieurs variantes 2.25D depuis le meilleur rendu 2D")
    parser.add_argument("input_png")
    parser.add_argument("--rules", default="config/default_rules_225d.yaml")
    parser.add_argument("--out-dir", default="output/225d_candidates")
    args = parser.parse_args()

    rules = load_yaml(args.rules, default={})
    base_cfg = dict(rules.get("render_225d") or {})
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = {
        "subtle": {"depth_px": 7, "side_alpha_start": 0.22, "side_alpha_end": 0.04, "apply_subtle_shear": True, "shear_x": -0.025, "scale_y": 0.975},
        "balanced": {"depth_px": 10, "side_alpha_start": 0.30, "side_alpha_end": 0.06, "apply_subtle_shear": True, "shear_x": -0.035, "scale_y": 0.965},
        "stronger": {"depth_px": 14, "side_alpha_start": 0.36, "side_alpha_end": 0.08, "apply_subtle_shear": True, "shear_x": -0.045, "scale_y": 0.955},
        "shadow_only": {"depth_px": 10, "side_alpha_start": 0.26, "side_alpha_end": 0.04, "apply_subtle_shear": False},
    }

    for name, override in variants.items():
        cfg = dict(base_cfg)
        cfg.update(override)
        out = out_dir / f"rendered_clean_225d_{name}.png"
        render_225d_from_image(args.input_png, out, cfg)
        print(f"OK - {name}: {out}")


if __name__ == "__main__":
    main()
