# Addendum v6 - v1+ cleanup

Cette version repart du constat que `rendered_clean.png` v1 est la meilleure base : le plan est lisible, avec portes/fenêtres/détails, mais il reste des artefacts lointains qui agrandissent la feuille et ajoutent du bruit.

## Différence avec v3/v4/v5

Les versions précédentes essayaient de reclassifier le DWG en objets physiques. Ça a provoqué un effet de balancier :

- trop strict : disparition des portes/fenêtres/détails ;
- trop permissif : retour des cartouches, barres et artefacts éloignés.

La v1+ garde la logique v1 :

```text
DWG/DXF -> filtre simple layers/entity types -> rendu vectoriel détaillé
```

Puis ajoute un nettoyage post-rendu :

```text
rendered_clean.png -> composantes raster -> seed principal -> expansion locale -> suppression artefacts éloignés -> crop
```

## Utilisation complète depuis le DWG

```bash
python cli_clean_plan_v1plus.py data/input_plans/sample.dwg \
  --rules config/default_rules_v1plus.yaml \
  --decisions data/work/layer_review/layer_decisions.yaml \
  --out output/rendered_clean_v1plus.png \
  --entities-csv output/entities_df_v1plus.csv
```

## Nettoyer directement le meilleur rendu v1 existant

```bash
python cli_cleanup_existing_render.py output/rendered_clean.png \
  --rules config/default_rules_v1plus.yaml \
  --out output/rendered_clean_v1plus_from_existing.png
```

C'est l'option que je recommande en premier, car elle ne change pas la sélection DWG déjà satisfaisante.

## Réglages

Si des artefacts lointains restent :

```yaml
post_cleanup:
  main_bbox_expand_ratio: 0.08
  iterative_expansion_passes: 3
  remove_component_area_lt: 15
```

Si des détails utiles proches du plan disparaissent :

```yaml
post_cleanup:
  main_bbox_expand_ratio: 0.16
  iterative_expansion_passes: 5
  remove_component_area_lt: 4
```
