# Sujet III - v9 / nettoyage des lignes parasites + shadow-only 2.25D

La v8 est dans la bonne direction : précision du plan + shadow-only 2.25D. Le problème restant est surtout la présence de traits parasites longs et fins, visibles autour du bâti ou sous forme de lignes de projection/grille.

La v9 ajoute un nettoyage raster conservateur **entre le rendu 2D et le shadow pass** :

```text
rendered_clean_v8_2d.png
  -> suppression des longs hairlines isolés
  -> suppression très légère des petits composants isolés
  -> crop
  -> shadow-only 2.25D
```

## Commande recommandée

À lancer sur le rendu 2D, pas sur l'image déjà ombrée :

```bash
python cli_clean_parasites_then_shadow.py output/rendered_clean_v8_2d.png \
  --rules config/default_rules_shadow225_v9.yaml \
  --out-2d output/rendered_clean_v9_2d_cleaned.png \
  --out-225d output/rendered_clean_v9_shadow225.png
```

## Nettoyer seulement le 2D

```bash
python cli_clean_parasites_only.py output/rendered_clean_v8_2d.png \
  --rules config/default_rules_shadow225_v9.yaml \
  --out output/rendered_clean_v9_2d_cleaned.png
```

## Si les parasites restent

```yaml
parasite_cleanup:
  min_run_length_ratio: 0.012
  min_run_length_px: 24
  max_mean_perpendicular_support: 2.8
  min_component_area_px: 20
```

## Si le nettoyage supprime trop de petits détails

```yaml
parasite_cleanup:
  min_run_length_ratio: 0.025
  min_run_length_px: 48
  max_mean_perpendicular_support: 1.8
  min_dense_neighborhood_ink: 24
  min_component_area_px: 6
```

## Tuning de sélection DWG

Si tu veux réduire les parasites en amont dans `default_rules_shadow225_v8.yaml`, ajoute dans `selection.hard_drop_layer_keywords` :

```yaml
- AXE
- GRID
- GRILLE
- REPERE
- REPÈRE
- PROJECTION
- CONSTRUCTION
- MODULE
- MODULATION
- TRACE
- TRACÉ
- GUIDE
- ALIGNEMENT
```

Ne mets pas `CLOISON`, `PORTE`, `FENETRE`, `MENUI` dans les hard drops.
