# v10 fastfix

La v10 originale pouvait se bloquer sur des grandes images car elle calculait la densité locale pixel par pixel dans chaque run.

Cette version utilise des integral images : la densité est calculée une seule fois par run.

## Commande recommandée

```bash
python cli_clean_nonwall_then_shadow_fast.py output/rendered_clean_v8_2d.png --rules config/default_rules_shadow225_v10_fast.yaml --out-2d output/rendered_clean_v10_fast_2d.png --out-225d output/rendered_clean_v10_fast_shadow225.png
```

## Seulement le nettoyage 2D

```bash
python cli_clean_nonwall_only_fast.py output/rendered_clean_v8_2d.png --rules config/default_rules_shadow225_v10_fast.yaml --out output/rendered_clean_v10_fast_2d.png
```

## Plus agressif

```yaml
nonwall_cleanup:
  min_run_length_ratio: 0.006
  min_run_length_px: 14
  max_strip_ink_density: 0.55
  dense_zone_min_density: 0.25
```

## Plus prudent

```yaml
nonwall_cleanup:
  min_run_length_ratio: 0.018
  min_run_length_px: 38
  max_strip_ink_density: 0.30
  dense_zone_min_density: 0.12
```
