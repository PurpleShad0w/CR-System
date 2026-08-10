# v12 mesh cleanup

Cette version cible les gros paquets de lignes paralleles: trames, hachures, faux escaliers, meshes avec diagonales.

Elle protege les murs epais puis cherche les composants formes par des lignes longues horizontales, verticales et diagonales. Si un composant contient suffisamment de lignes et plusieurs orientations, la zone fine est supprimee, les murs restent.

## Commande

```bash
python cli_mesh_clean_then_shadow.py output/rendered_clean_v11_projection_2d.png --rules config/default_rules_shadow225_v12.yaml --out-2d output/rendered_clean_v12_mesh_2d.png --out-225d output/rendered_clean_v12_mesh_shadow225.png
```

## Nettoyage 2D uniquement

```bash
python cli_mesh_clean_only.py output/rendered_clean_v11_projection_2d.png --rules config/default_rules_shadow225_v12.yaml --out output/rendered_clean_v12_mesh_2d.png
```

## Plus agressif

```yaml
mesh_cleanup:
  min_mesh_line_pixels: 120
  min_mesh_density: 0.030
  min_mesh_orientations: 1
  mesh_bbox_margin_px: 5
  speck_area_lt: 80
```

## Plus prudent

```yaml
mesh_cleanup:
  min_mesh_line_pixels: 360
  min_mesh_density: 0.070
  min_mesh_orientations: 2
  action: fade
  fade_gray: 232
```
