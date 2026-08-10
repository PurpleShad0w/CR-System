# Sujet III - v8 / précision v1+ + shadow-only 2.25D

Direction retenue : prioriser le candidat `shadow_only`. Le rendu doit rester un plan d'architecte lisible, pas une vue 3D.

## Pipeline recommandé

```text
DWG/DXF
  -> extraction entités + virtual_entities des blocs
  -> rendu v1-like détaillé
  -> rescue contrôlé des détails portes/fenêtres/menuiseries proches du plan
  -> nettoyage post-rendu local
  -> shadow-only 2.25D
```

## Pourquoi v8

Les versions strictes perdaient les détails. Les versions permissives ramenaient les cartouches/artefacts. La v8 revient à la sélection v1, puis ajoute seulement :

- un rescue contrôlé des petites features proches du plan principal ;
- un debug CSV pour comprendre ce qui a été rendu ou supprimé ;
- un rendu shadow-only 2.25D, sans cisaillement et sans extrusion verticale.

## Commande complète depuis le DWG

```bash
python cli_render_shadow225_v8.py data/input_plans/sample.dwg \
  --rules config/default_rules_shadow225_v8.yaml \
  --decisions data/work/layer_review/layer_decisions.yaml \
  --out-2d output/rendered_clean_v8_2d.png \
  --out-225d output/rendered_clean_v8_shadow225.png \
  --entities-csv output/entities_df_v8.csv \
  --selected-csv output/selected_entities_v8.csv \
  --debug-csv output/debug_selection_v8.csv
```

## Appliquer le shadow-only sur le meilleur PNG existant

```bash
python cli_shadow225_from_existing_v8.py output/rendered_clean_v1plus_from_existing.png \
  --rules config/default_rules_shadow225_v8.yaml \
  --out output/rendered_clean_shadow225_v8_from_existing.png
```

## Si les portes/fenêtres sont encore trop faibles

```yaml
selection:
  rescue_bbox_expand_ratio: 0.14
  max_rescue_length_ratio: 0.22
  min_rescue_length: 0.4
  min_rescue_bbox_diag: 0.2
```

## Si trop de bruit revient

```yaml
selection:
  rescue_bbox_expand_ratio: 0.07
  max_rescue_length_ratio: 0.10
  keep_components_score_ratio: 0.28
post_cleanup:
  remove_component_area_lt: 12
  main_bbox_expand_ratio: 0.10
```

## Si l'effet 2.25D est trop fort

```yaml
shadow_225d:
  depth_px: 6
  shadow_alpha_start: 0.20
  shadow_alpha_end: 0.03
```

## Si l'effet 2.25D est trop discret

```yaml
shadow_225d:
  depth_px: 13
  shadow_alpha_start: 0.36
  blur_radius: 1.5
```
