# Addendum v4 - Sujet III - balanced cleanup

La v3 était trop stricte. L'image fournie montre que le rendu a conservé essentiellement l'enveloppe et quelques cloisons, mais a supprimé presque toutes les portes, fenêtres, détails intérieurs et ouvertures.

Cause principale probable dans notre projet : `layer_decisions.yaml` contient beaucoup de layers `CLO`, `porte`, `cloison`, etc. dans `drop`. La v3 respectait ces drops trop fortement. En v4, un layer drop peut être contourné si l'entité ressemble clairement à un objet physique.

## Principe v4

```text
1. Garder les objets physiques reconnus : wall/door/window/stairs
2. Autoriser le rescue d'objets physiques même depuis un layer drop, sauf hard_noise
3. Construire une empreinte à partir des murs
4. Restaurer les petits arcs/lignes/blocs proches de cette empreinte
5. Supprimer seulement les composants raster minuscules, sans keep_largest_component
```

## Commande

```bash
python cli_clean_plan_v4.py data/input_plans/sample.dwg \
  --rules config/default_rules_balanced_v4.yaml \
  --decisions data/work/layer_review/layer_decisions.yaml \
  --out output/rendered_clean_v4.png \
  --entities-csv output/entities_df_v4.csv \
  --clean-csv output/clean_balanced_entities_v4.csv
```

## Streamlit

```bash
streamlit run app_layer_review_v4.py
```

## Réglages si encore trop strict

```yaml
classification:
  wall_score_threshold: 0.38
  door_score_threshold: 0.32
  window_score_threshold: 0.30

balanced_noise:
  footprint_buffer_ratio: 0.06
  rescue_max_length_ratio: 0.30
  raster_remove_components_area_lt: 8
```

## Réglages si trop sale

```yaml
balanced_noise:
  footprint_buffer_ratio: 0.025
  rescue_max_length_ratio: 0.12
  raster_remove_components_area_lt: 35
```
