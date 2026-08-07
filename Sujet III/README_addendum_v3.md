# Addendum v3 - Sujet III - nettoyage du bruit CAD

La v2 gardait trop d'éléments non physiques : trames, motifs, petits traits isolés, mobilier/plafond/sol, et quelques artefacts éloignés du bâtiment. La v3 change la logique : on ne rend plus les entités incertaines par défaut.

## Ce que l'étude d'inspiration faisait contre le bruit

L'étude ne nettoyait pas le bruit par morphologie image. Elle évitait surtout le bruit en amont :

- conversion du DWG en DataFrame structuré ;
- sélection des objets par `Layer`, `ID`, `ParentID` et coordonnées ;
- filtrage des entités dont le layer contient `wall` ;
- récupération des objets liés via `ParentID` ;
- reconstruction des polylignes avant visualisation.

Adaptation v3 : on reproduit cette logique, mais avec plusieurs classes physiques (`wall`, `door`, `window`, `stairs`) au lieu de seulement `wall`.

## Pipeline v3

```text
DWG/DXF
  -> entities_df
  -> scores physiques wall/door/window/stairs
  -> hard drop annotations/hatch/dim/text/plafond/sol/mobilier
  -> suppression micro-entités
  -> suppression layers répétitifs bruités
  -> conservation du plus gros composant spatial
  -> rendu vectoriel monochrome
  -> nettoyage raster final
```

## Commande

```bash
python cli_clean_plan_v3.py data/input_plans/sample.dwg \
  --rules config/default_rules_clean_v3.yaml \
  --decisions data/work/layer_review/layer_decisions.yaml \
  --out output/rendered_clean_v3.png \
  --entities-csv output/entities_df_v3.csv \
  --clean-csv output/clean_physical_entities_v3.csv
```

## Streamlit

```bash
streamlit run app_layer_review_v3.py
```

## Réglages prioritaires

Si le plan reste trop sale :

```yaml
noise:
  keep_only_physical_classes: true
  min_length_for_unclassified: 18.0
  raster_remove_components_area_lt: 120
```

Si trop de portes/fenêtres disparaissent :

```yaml
classification:
  door_score_threshold: 0.42
  window_score_threshold: 0.40
noise:
  min_length_for_physical: 2.0
```

Si le crop supprime une aile séparée du bâtiment :

```yaml
noise:
  keep_largest_spatial_component: false
  raster_keep_largest_component: false
```
