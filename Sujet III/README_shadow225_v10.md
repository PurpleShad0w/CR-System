# Sujet III - v10 / nettoyage des lignes non-mur + shadow-only 2.25D

La v9 ne changeait presque rien car le filtre était trop conservateur. La v10 cible plus directement ce qui gêne visuellement : les longues lignes fines qui ne sont pas des murs.

## Principe

```text
rendu 2D v8
  -> détection des traits épais protégés = murs / contours forts
  -> détection des petits composants protégés = portes / fenêtres / détails
  -> suppression des longues lignes fines horizontales/verticales non protégées
  -> suppression des composants fins éloignés
  -> shadow-only 2.25D
```

## Commande recommandée

Toujours partir du rendu 2D, pas du rendu déjà ombré :

```bash
python cli_clean_nonwall_then_shadow.py output/rendered_clean_v8_2d.png \
  --rules config/default_rules_shadow225_v10.yaml \
  --out-2d output/rendered_clean_v10_2d_nonwall_cleaned.png \
  --out-225d output/rendered_clean_v10_shadow225.png
```

## Nettoyer seulement le 2D

```bash
python cli_clean_nonwall_only.py output/rendered_clean_v8_2d.png \
  --rules config/default_rules_shadow225_v10.yaml \
  --out output/rendered_clean_v10_2d_nonwall_cleaned.png
```

## Si les lignes non-mur restent trop visibles

```yaml
nonwall_cleanup:
  min_run_length_ratio: 0.006
  min_run_length_px: 14
  max_thin_support_px: 4
  dense_zone_min_ink: 55
```

## Si trop de détails disparaissent

```yaml
nonwall_cleanup:
  min_run_length_ratio: 0.018
  min_run_length_px: 34
  max_thin_support_px: 2
  preserve_component_area_lt: 220
  preserve_component_bbox_max_px: 56
```

## Variante moins risquée: fade au lieu de erase

```yaml
nonwall_cleanup:
  action: fade
  fade_gray: 230
```

Cette option garde l'information graphique mais la rend moins intrusive.
