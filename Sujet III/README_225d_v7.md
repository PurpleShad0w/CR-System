# Sujet III - v7 / rendu 2.25D depuis le meilleur rendu v1+

Cette version ne repart pas dans une classification DWG. Elle part du meilleur résultat actuel :

```text
rendered_clean_v1plus_from_existing.png
```

Puis elle applique un effet 2.25D très léger :

```text
plan noir original
+ duplication grise progressive décalée
+ ombre de contact légère
+ cisaillement très faible optionnel
+ crop final
```

L'objectif est une vue aérienne légèrement inclinée, pas une maquette 3D.

## Commande recommandée

```bash
python cli_render_225d_from_existing.py output/rendered_clean_v1plus_from_existing.png \
  --rules config/default_rules_225d.yaml \
  --out output/rendered_clean_225d.png
```

## Générer plusieurs candidats

```bash
python cli_batch_225d_candidates.py output/rendered_clean_v1plus_from_existing.png \
  --rules config/default_rules_225d.yaml \
  --out-dir output/225d_candidates
```

Cela produit :

```text
rendered_clean_225d_subtle.png
rendered_clean_225d_balanced.png
rendered_clean_225d_stronger.png
rendered_clean_225d_shadow_only.png
```

## Réglages importants

Si l'effet est trop 3D :

```yaml
render_225d:
  depth_px: 6
  side_alpha_start: 0.18
  side_alpha_end: 0.03
  apply_subtle_shear: false
```

Si l'effet est trop plat :

```yaml
render_225d:
  depth_px: 14
  side_alpha_start: 0.36
  shear_x: -0.045
  scale_y: 0.955
```

Si tu veux garder le plan strictement non déformé, mais avec du relief :

```yaml
render_225d:
  apply_subtle_shear: false
```
