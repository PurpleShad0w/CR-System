# Addendum v2 - Sujet III - plan propre + vue aérienne 2.5D

Cette v2 corrige deux limites de la première version :

1. Le rendu était trop pauvre car les portes, fenêtres et éléments utiles en blocs étaient souvent perdus.
2. La sortie 2.5D ressemblait trop à une extrusion 3D verticale, alors que l'objectif est une vue aérienne lisible.

## Changements principaux

- Extraction explicite des `INSERT` et de leurs `virtual_entities()` pour récupérer les objets contenus dans les blocs.
- Classification multi-classes : `wall`, `door`, `window`, `stairs`, `furniture`, `annotation`, `other`.
- Les layers qui contiennent portes/fenêtres/escalier ne sont plus supprimés automatiquement si le fichier de décisions ne les droppe pas explicitement.
- La sortie 2.5D devient une image top-down avec ombre légère, pas une scène Plotly 3D.
- Le rendu conserve maintenant les entités `other` gardées, ce qui évite de perdre des détails architecturaux quand le classifieur n'est pas sûr.

## Fichiers

```text
src/dwg_entities.py
src/feature_classifier.py
src/layer_classifier.py
src/render_clean_plan.py
src/render_25d.py
cli_clean_plan_v2.py
app_layer_review_v2.py
config/default_rules_aerial_25d.yaml
```

## CLI

```bash
python cli_clean_plan_v2.py data/input_plans/sample.dwg \
  --rules config/default_rules_aerial_25d.yaml \
  --decisions data/work/layer_review/layer_decisions.yaml \
  --out output/rendered_clean_v2.png \
  --aerial output/rendered_aerial_25d.png \
  --entities-csv output/entities_df_v2.csv
```

## Streamlit

```bash
streamlit run app_layer_review_v2.py
```

## Ajustements utiles

Si le rendu reste trop chargé :

```yaml
render:
  draw_other_kept: false
```

Si portes/fenêtres restent absentes, vérifier dans l'interface les colonnes `blocks`, `classes`, `door_score`, `window_score`. Les blocs de portes/fenêtres arrivent souvent sous des noms de bloc plutôt que sous des layers explicites.

Si la vue aérienne est encore trop volumétrique :

```yaml
aerial_25d:
  shadow_offset_px: 2
  shadow_alpha: 0.08
```
