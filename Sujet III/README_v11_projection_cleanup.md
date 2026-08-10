# v11 projection cleanup

Les lignes non-mur restantes traversent souvent les murs, donc les detecteurs de runs locaux ne suffisent pas.

La v11 detecte les lignes par alignement global horizontal/vertical dans le masque fin `ink - murs_epais`. Les murs restent proteges, mais les traits fins alignes sont supprimes ou attenues.

## Commande

```bash
python cli_projection_clean_then_shadow.py output/rendered_clean_v8_2d.png --rules config/default_rules_shadow225_v11.yaml --out-2d output/rendered_clean_v11_projection_2d.png --out-225d output/rendered_clean_v11_projection_shadow225.png
```

## Plus agressif

```yaml
projection_cleanup:
  min_span_ratio: 0.10
  min_pixels_ratio_in_span: 0.10
  remove_band_halfwidth_px: 2
  dense_threshold: 0.38
```

## Plus prudent

```yaml
projection_cleanup:
  action: fade
  fade_gray: 232
  min_span_ratio: 0.25
  remove_band_halfwidth_px: 1
```
