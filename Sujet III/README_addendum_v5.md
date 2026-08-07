# Addendum v5 - Sujet III - nettoyage par composant spatial principal

La v4 a récupéré trop d'objets éloignés : barres verticales, rectangles de légende/cartouche en haut à droite, petits arcs isolés. La cause : le filtrage par bbox global ne distingue pas le bâtiment principal des blocs CAD séparés.

La v5 sélectionne donc le bâtiment principal par composantes spatiales vectorielles :

```text
entities_df -> classification souple -> composantes spatiales -> meilleur composant dense/physique -> rescue local -> rendu -> crop
```

Le composant gagnant n'est pas le plus grand rectangle, mais celui qui maximise longueur physique + densité + ouvertures, avec pénalité pour micro-objets répétitifs.

## Commande

```bash
python cli_clean_plan_v5.py data/input_plans/sample.dwg \
  --rules config/default_rules_component_v5.yaml \
  --decisions data/work/layer_review/layer_decisions.yaml \
  --out output/rendered_clean_v5.png \
  --entities-csv output/entities_df_v5.csv \
  --clean-csv output/clean_component_entities_v5.csv
```

## Si la sortie garde encore des blocs éloignés

```yaml
component_cleanup:
  min_component_score_ratio: 0.35
  rescue_buffer_ratio: 0.035
  rescue_max_length_ratio: 0.12
```

## Si la sortie perd des détails du bâtiment

```yaml
component_cleanup:
  min_component_score_ratio: 0.12
  rescue_buffer_ratio: 0.075
  rescue_max_length_ratio: 0.25
raster_cleanup:
  remove_components_area_lt: 6
```
