# Pipeline de prédiction des consommations et températures bâtiment

## 1. Objectif du projet

Ce projet met en place une pipeline complète de traitement, d’enrichissement, de nettoyage, d’entraînement, d’évaluation, de prédiction et de reporting autour des données de sites et de zones bâtiment.

La pipeline manipule principalement :

- des historiques site ;
- des historiques zone ;
- des données météo site ;
- des logs de consommation détaillés par usage et par type de compteur ;
- des tables annexes de correspondance pour les usages et les types de compteurs.

L’objectif principal est de prédire différentes grandeurs quotidiennes, notamment :

- consommation électrique totale ;
- consommation électrique “accurate” reconstruite depuis les usages détaillés ;
- consommation électrique hors BVE ;
- consommations par usage ;
- consommation d’eau ;
- température intérieure.

La pipeline permet aussi de générer des graphes d’évaluation, des métriques par cible et par groupe, ainsi que des prédictions futures quand des données météo futures sont disponibles.

---

## 2. Organisation générale de la pipeline

La pipeline suit l’ordre logique suivant :

1. Enrichissement des historiques avec les consommations détaillées par usage.
2. Nettoyage des données.
3. Entraînement des modèles.
4. Évaluation des modèles.
5. Génération de prédictions futures.
6. Génération de rapports et graphes.

Ordre d’exécution recommandé :

```powershell
python .\run_enrich_from_consumptiondrift.py --db-dir "db"
python -m pipeline.run_clean --config config.yaml --level all
python -m pipeline.run_train --config config.yaml --level all --target all
python -m pipeline.run_evaluate --config config.yaml --level all --target all
python -m pipeline.run_predict --config config.yaml --level all --target all
python -m pipeline.run_report --config config.yaml --level all --target all --site all
```

---

## 3. Données d’entrée

### 3.1 Données principales

Les données principales du projet sont stockées dans le dossier `db/`.

Les nouveaux fichiers principaux sont :

- `site_history.csv`
- `zone_history.csv`
- `site_weather.csv`
- `consumptions_drifts.csv`

`site_history.csv` et `zone_history.csv` contiennent les historiques de consommation et de température. Ils incluent notamment :

- `siteId`
- `zoneId` côté zone
- `indoorTempDegC`
- `elecBveKwh`
- `elecCvcKwh`
- `elecForceKwh`
- `elecLightingKwh`
- `elecAggregatedKwh`
- `elecTotalKwh`
- `waterM3`
- `dtUpdate`

`site_weather.csv` contient les données météo quotidiennes par site. Il inclut notamment :

- `siteId`
- `tempMin`
- `tempMax`
- `chancePluie`
- `djC`
- `djF`
- `tempAmb`
- `humidity`
- `windSpeed`
- `dtUpdate`

`consumptions_drifts.csv` contient les consommations détaillées par usage et type de compteur. Il inclut notamment :

- `siteId`
- `zoneId`
- `perimeter`
- `date`
- `consumption`
- `usageId`
- `meterTypeId`

### 3.2 Données annexes

Les fichiers annexes restent nécessaires pour interpréter les consommations détaillées :

- `usages.csv`
- `metertypes.csv`
- éventuellement `Sites_Shyrka_Infos.xlsx`

`usages.csv` permet d’associer un `usageId` à un nom d’usage.

`metertypes.csv` permet d’associer un `meterTypeId` à une famille de compteur, par exemple électricité, eau, eau chaude ou eau froide.

`Sites_Shyrka_Infos.xlsx` peut être utilisé pour enrichir les historiques avec des informations statiques comme la surface, si la colonne correspondante est disponible.

---

## 4. Fichier de configuration

La pipeline est pilotée par `config.yaml`.

Exemple de configuration recommandée si l’enrichissement produit `sitehist_enriched.csv` et `zonehist_enriched.csv` :

```yaml
paths:
  db_dir: db
  out_dir: output_ai
  site_infos_file: Sites_Shyrka_Infos.xlsx

level_defaults:
  site:
    id_cols: [siteId]
    date_col: date
    hist_file: sitehist_enriched.csv
    pred_file: sitepred.csv
    weath_file: site_weather.csv

  zone:
    id_cols: [siteId, zoneId]
    date_col: date
    hist_file: zonehist_enriched.csv
    pred_file: zonepred.csv
    weath_file: site_weather.csv

features:
  lags: [1, 2, 7, 14]
  rolling_windows: [7, 14, 30]
  add_calendar: true
  add_site_id: true
  weather_cols: [tempAmb, humidity, djF, djC, tempMin, tempMax, chancePluie, windSpeed]
  static_cols: [surface_m2]

model:
  type: ridge
  ridge:
    alpha: 1.0
  hist_gbdt:
    max_depth: 8
    learning_rate: 0.08
    max_iter: 350
    l2_regularization: 0.0

training:
  valid_days: 60

prediction:
  days: null
```

Remarque importante : si `config.yaml` pointe directement vers `site_history.csv` et `zone_history.csv`, la pipeline n’exploite pas les colonnes enrichies issues de `consumptions_drifts.csv`. Pour exploiter les usages détaillés, il faut d’abord exécuter l’enrichissement puis faire pointer la config vers les fichiers enrichis.

---

## 5. Étape 1 : enrichissement des historiques

### 5.1 Objectif

L’enrichissement consiste à fusionner les historiques principaux avec les consommations détaillées issues de `consumptions_drifts.csv`.

Cette étape permet de créer de nouvelles colonnes de consommation par usage, par exemple :

- `elecPcKwh`
- `elecProcessKwh`
- `elecHeatingKwh`
- `elecEcsKwh`
- `elecGeneralKwh`
- `elecAirConditioningKwh`
- des colonnes liées à l’eau, l’eau chaude ou l’eau froide selon les types de compteurs présents.

Elle permet aussi de reconstruire un total électrique issu des consommations détaillées :

- `elecTotalFromDriftKwh`

Ce total peut ensuite être utilisé pour produire une cible dérivée :

- `elecTotalAccurateKwh`

### 5.2 Commande

```powershell
python .\run_enrich_from_consumptiondrift.py --db-dir "db"
```

### 5.3 Arguments

- `--db-dir` : chemin vers le dossier contenant les fichiers CSV.
- `--perimeter` : périmètre à utiliser dans `consumptions_drifts.csv`. La valeur par défaut est généralement `zone`.
- `--no-fill-existing-4` : option permettant de ne pas compléter les quatre usages électriques historiques depuis les colonnes drift.

### 5.4 Sorties attendues

L’étape d’enrichissement produit généralement :

- `db/sitehist_enriched.csv`
- `db/zonehist_enriched.csv`
- `db/drift_usage_mapping.csv`

`drift_usage_mapping.csv` sert à tracer la correspondance entre :

- `usageId`
- nom d’usage
- `meterTypeId`
- type de compteur
- nom de colonne créée dans les fichiers enrichis

### 5.5 Point d’attention

Les fichiers enrichis doivent être ceux utilisés par la pipeline principale. Donc `config.yaml` doit pointer vers les fichiers enrichis, pas directement vers les fichiers sources, si l’on veut exploiter les nouveaux usages.

---

## 6. Étape 2 : nettoyage des données

### 6.1 Objectif

Le nettoyage prépare les historiques pour l’entraînement.

`run_clean.py` charge les données via `load_level_tables`, normalise les dates, construit certains totaux dérivés, détecte les colonnes de consommation et applique plusieurs traitements de nettoyage.

Le nettoyage peut notamment :

- convertir `date` ou `dtUpdate` vers une date journalière ;
- construire ou reconstruire `elecTotalAccurateKwh` ;
- construire ou reconstruire `elecTotalNoBveKwh` ;
- détecter les colonnes de consommation ;
- gérer certaines valeurs nulles, négatives ou aberrantes ;
- produire des logs d’actions de nettoyage.

### 6.2 Commande

```powershell
python -m pipeline.run_clean --config config.yaml --level all
```

### 6.3 Arguments

- `--config` : chemin vers le fichier de configuration.
- `--level` :
  - `site` : nettoyer uniquement les historiques site ;
  - `zone` : nettoyer uniquement les historiques zone ;
  - `all` : nettoyer site et zone.

### 6.4 Sorties produites

Dans le dossier `output_ai/`, le nettoyage produit :

- `sitehist_cleaned.csv`
- `zonehist_cleaned.csv`

Il produit aussi des logs de nettoyage, par exemple :

- `cleanlog_site_local_spike_elecTotalKwh.csv`
- `cleanlog_site_local_spike_elecTotalFromDriftKwh.csv`
- `cleanlog_site_cap_elecTotalKwh.csv`
- `cleanlog_zone_cap_<target>.csv`

Les logs permettent d’auditer les valeurs supprimées, propagées ou capées.

### 6.5 Remarque sur les nouvelles données nettoyées en amont

Même si les nouveaux fichiers source sont déjà nettoyés avant ingestion, cette étape reste utile car elle garantit l’uniformité des formats, reconstruit les variables dérivées et protège la pipeline contre les valeurs extrêmes restantes.

---

## 7. Étape 3 : entraînement des modèles

### 7.1 Objectif

L’entraînement construit un modèle par couple :

- niveau : `site` ou `zone` ;
- cible : consommation, température ou usage.

La logique dynamique permet d’entraîner automatiquement les usages électriques présents dans les fichiers enrichis quand `--target all` est utilisé.

### 7.2 Commande

```powershell
python -m pipeline.run_train --config config.yaml --level all --target all
```

### 7.3 Arguments

- `--config` : fichier de configuration.
- `--level` :
  - `site`
  - `zone`
  - `all`
- `--target` :
  - `elecTotalKwh`
  - `elecTotalAccurateKwh`
  - `elecTotalNoBveKwh`
  - `waterM3`
  - `indoorTempDegC`
  - un usage précis comme `elecPcKwh`
  - `elecUses`
  - `all`

### 7.4 Features utilisées

Les features sont construites à partir :

- des identifiants (`siteId`, éventuellement `zoneId`) ;
- des variables météo configurées ;
- des features calendrier ;
- des lags ;
- des rolling windows ;
- des variables statiques disponibles comme `surface_m2`.

La configuration standard contient notamment :

- lags : `[1, 2, 7, 14]`
- rolling windows : `[7, 14, 30]`
- météo : `tempAmb`, `humidity`, `djF`, `djC`, `tempMin`, `tempMax`, `chancePluie`, `windSpeed`
- statique : `surface_m2`

### 7.5 Sorties produites

Les modèles sont écrits dans :

- `output_ai/models/`

Chaque cible produit :

- un fichier modèle `.joblib` ;
- un fichier metadata `.meta.json`.

Exemples :

- `output_ai/models/site_elecTotalKwh.joblib`
- `output_ai/models/site_elecTotalKwh.meta.json`
- `output_ai/models/zone_elecPcKwh.joblib`
- `output_ai/models/zone_elecPcKwh.meta.json`

Les metadata contiennent notamment :

- la cible ;
- le niveau ;
- les colonnes de features ;
- les colonnes catégorielles ;
- le type de transformation cible ;
- la taille de la fenêtre de validation.

---

## 8. Étape 4 : évaluation des modèles

### 8.1 Objectif

L’évaluation rejoue les modèles sur la fenêtre de validation et produit des métriques globales et parfois par groupe.

### 8.2 Commande

```powershell
python -m pipeline.run_evaluate --config config.yaml --level all --target all
```

### 8.3 Arguments

- `--config` : fichier de configuration.
- `--level` :
  - `site`
  - `zone`
  - `all`
- `--target` :
  - cible précise ;
  - `elecUses` ;
  - `all`.

### 8.4 Sorties produites

Les fichiers d’évaluation sont écrits dans `output_ai/`.

Exemples :

- `eval_site_elecTotalKwh.csv`
- `eval_site_elecTotalAccurateKwh.csv`
- `eval_zone_elecTotalKwh.csv`
- `eval_site_elecPcKwh.csv`
- `eval_preds_site_elecTotalKwh.csv`
- `eval_site_elecTotalKwh_by_group.csv`

### 8.5 Métriques produites

Les fichiers d’évaluation peuvent contenir :

- `rows`
- `MAE`
- `RMSE`
- `WAPE`
- `Bias`
- `MedAE`
- `sMAPE`
- `R2`
- `MAE_m2`
- `RMSE_m2`
- `WAPE_m2`

Les fichiers `eval_preds_*` contiennent les valeurs vraies et prédites, généralement sous forme :

- identifiants (`siteId`, `zoneId`) ;
- `date` ;
- `y_true` ;
- `y_pred`.

---

## 9. Étape 5 : prédiction future

### 9.1 Objectif

Le script de prédiction génère des prédictions futures à partir :

- de l’historique nettoyé ;
- des modèles entraînés ;
- des données météo disponibles dans le futur.

### 9.2 Commande

```powershell
python -m pipeline.run_predict --config config.yaml --level all --target all
```

### 9.3 Arguments

- `--config` : fichier de configuration.
- `--level` :
  - `site`
  - `zone`
  - `all`
- `--target` :
  - cible précise ;
  - `elecUses` ;
  - `all`
- `--days` :
  - nombre maximal de jours à prédire ;
  - si absent, la valeur vient de `prediction.days` dans `config.yaml`.

Si `prediction.days` vaut `null`, l’horizon est déterminé par les données météo disponibles.

### 9.4 Sorties produites

Les prédictions sont écrites dans `output_ai/`.

Exemples :

- `pred_site_elecTotalKwh.csv`
- `pred_site_elecTotalAccurateKwh.csv`
- `pred_site_elecPcKwh.csv`
- `pred_zone_elecTotalKwh.csv`

Chaque fichier contient généralement :

- les identifiants (`siteId`, éventuellement `zoneId`) ;
- la `date` ;
- la prédiction `yhat`.

### 9.5 Fonctionnement autoregressif

Pour chaque date future, le script :

1. construit une ligne de features ;
2. injecte les données météo du jour si disponibles ;
3. calcule les features calendrier ;
4. reconstruit les lags à partir de l’historique et des prédictions déjà générées ;
5. calcule les rolling windows ;
6. applique le modèle ;
7. ajoute la prédiction à l’état interne pour produire les jours suivants.

---

## 10. Étape 6 : reporting et graphes

### 10.1 Objectif

Le reporting génère des graphes d’évaluation permettant de comprendre :

- la qualité de prédiction ;
- les erreurs ;
- les séries temporelles train/validation ;
- les comparaisons entre méthodes de calcul du total électrique.

### 10.2 Commande

```powershell
python -m pipeline.run_report --config config.yaml --level all --target all --site all
```

### 10.3 Arguments

- `--config` : fichier de configuration.
- `--level` :
  - `site`
  - `zone`
  - `all`
- `--target` :
  - cible précise ;
  - `elecUses` ;
  - `all`
- `--site` :
  - identifiant d’un site ;
  - ou `all` pour générer les séries temporelles sur tous les sites disponibles.

### 10.4 Graphes produits

Les graphes sont écrits dans :

- `output_ai/figures/`

Le reporting produit généralement :

- parity plot linéaire capé p95 ;
- parity plot linéaire capé p99 ;
- parity plot en échelle log ;
- histogramme des résidus ;
- séries temporelles train/validation ;
- graphes de comparaison direct vs somme des usages ;
- scatter plots de comparaison ;
- hexbin density plots ;
- versions full et zoom pour éviter que les outliers écrasent l’échelle.

Exemples de noms de fichiers :

- `parity_site_elecTotalKwh_p95.png`
- `parity_site_elecTotalKwh_p99.png`
- `parity_site_elecTotalKwh_log.png`
- `resid_site_elecTotalKwh.png`
- `ts_site170_elecTotalKwh_train_valid.png`
- `compare_total_direct_vs_sumuses_site_wape_scatter_full.png`
- `compare_total_direct_vs_sumuses_site_rmse_hexbin_zoom.png`

### 10.5 Données de comparaison

Le reporting peut aussi produire des CSV dans :

- `output_ai/figures/compare/`

Exemples :

- `compare_total_direct_vs_sumuses_site_by_group.csv`
- `compare_total_direct_vs_sumuses_zone_by_group.csv`
- `outliers_site_RMSE_direct.csv`
- `outliers_zone_RMSE_sumUses.csv`

Ces fichiers servent à identifier :

- les sites ou zones où le modèle direct est meilleur ;
- les sites ou zones où la somme des usages est meilleure ;
- les groupes qui produisent des valeurs aberrantes ;
- les sites ou zones responsables des graphes illisibles.

---

## 11. Cibles principales

### 11.1 Cibles historiques

- `elecTotalKwh` : total électrique historique.
- `elecBveKwh` : consommation BVE.
- `elecCvcKwh` : consommation CVC.
- `elecForceKwh` : consommation force.
- `elecLightingKwh` : consommation éclairage.
- `waterM3` : consommation d’eau.
- `indoorTempDegC` : température intérieure.

### 11.2 Cibles dérivées

- `elecTotalFromDriftKwh` : somme des usages électriques reconstruits depuis `consumptions_drifts.csv`.
- `elecTotalAccurateKwh` : total électrique reconstruit ou priorisé depuis les données drift.
- `elecTotalNoBveKwh` : total électrique hors BVE.

### 11.3 Usages dynamiques

Lorsque les fichiers enrichis contiennent de nouveaux usages, la pipeline peut les détecter automatiquement.

Exemples possibles :

- `elecPcKwh`
- `elecProcessKwh`
- `elecHeatingKwh`
- `elecEcsKwh`
- `elecGeneralKwh`
- `elecAirConditioningKwh`

La logique `--target all` doit inclure les usages présents dans les fichiers enrichis, sous réserve que les scripts soient synchronisés avec cette logique dynamique.

---

## 12. Niveaux de granularité

La pipeline fonctionne sur deux niveaux.

### 12.1 Niveau site

Identifiant :

- `siteId`

Fichiers principaux :

- `sitehist_cleaned.csv`
- modèles `site_<target>.joblib`
- évaluations `eval_site_<target>.csv`
- prédictions `pred_site_<target>.csv`
- graphes `parity_site_<target>_*.png`

### 12.2 Niveau zone

Identifiants :

- `siteId`
- `zoneId`

Fichiers principaux :

- `zonehist_cleaned.csv`
- modèles `zone_<target>.joblib`
- évaluations `eval_zone_<target>.csv`
- prédictions `pred_zone_<target>.csv`
- graphes `parity_zone_<target>_*.png`

---

## 13. Dossiers de sortie

### 13.1 `db/`

Contient les données sources et enrichies :

- `site_history.csv`
- `zone_history.csv`
- `site_weather.csv`
- `consumptions_drifts.csv`
- `usages.csv`
- `metertypes.csv`
- `sitehist_enriched.csv`
- `zonehist_enriched.csv`
- `drift_usage_mapping.csv`

### 13.2 `output_ai/`

Contient les données produites par la pipeline :

- historiques nettoyés ;
- modèles ;
- évaluations ;
- prédictions ;
- logs de nettoyage.

Exemples :

- `sitehist_cleaned.csv`
- `zonehist_cleaned.csv`
- `eval_site_elecTotalKwh.csv`
- `pred_site_elecTotalKwh.csv`

### 13.3 `output_ai/models/`

Contient les modèles entraînés :

- `.joblib`
- `.meta.json`

### 13.4 `output_ai/figures/`

Contient les graphes générés par le reporting.

### 13.5 `output_ai/figures/compare/`

Contient les graphes et CSV de comparaison entre méthodes de calcul du total électrique.

---

## 14. Commandes usuelles

### 14.1 Enrichir les historiques

```powershell
python .\run_enrich_from_consumptiondrift.py --db-dir "db"
```

### 14.2 Nettoyer les données

```powershell
python -m pipeline.run_clean --config config.yaml --level all
```

### 14.3 Entraîner tous les modèles

```powershell
python -m pipeline.run_train --config config.yaml --level all --target all
```

### 14.4 Entraîner une cible seulement

Site uniquement :

```powershell
python -m pipeline.run_train --config config.yaml --level site --target elecTotalKwh
```

Zone uniquement :

```powershell
python -m pipeline.run_train --config config.yaml --level zone --target indoorTempDegC
```

Usage spécifique :

```powershell
python -m pipeline.run_train --config config.yaml --level site --target elecPcKwh
```

### 14.5 Évaluer tous les modèles

```powershell
python -m pipeline.run_evaluate --config config.yaml --level all --target all
```

### 14.6 Générer les prédictions

```powershell
python -m pipeline.run_predict --config config.yaml --level all --target all
```

Avec une limite explicite :

```powershell
python -m pipeline.run_predict --config config.yaml --level all --target all --days 14
```

### 14.7 Générer les rapports et graphes

```powershell
python -m pipeline.run_report --config config.yaml --level all --target all --site all
```

Pour un site précis :

```powershell
python -m pipeline.run_report --config config.yaml --level site --target elecTotalKwh --site 170
```

---

## 15. Interprétation rapide des métriques

### MAE

Erreur absolue moyenne. Plus elle est faible, meilleur est le modèle.

### RMSE

Racine de l’erreur quadratique moyenne. Elle pénalise fortement les grosses erreurs.

### WAPE

Erreur absolue pondérée par le volume réel. Utile pour comparer des consommations d’échelles différentes.

### Bias

Erreur moyenne signée.

- Bias positif : le modèle surestime en moyenne.
- Bias négatif : le modèle sous-estime en moyenne.

### R2

Indicateur de performance par rapport à une prédiction moyenne.

- proche de 1 : bon modèle ;
- proche de 0 : peu mieux qu’une moyenne ;
- négatif : moins bon qu’une baseline constante.

### Métriques au m²

Quand `surface_m2` est disponible, certaines métriques peuvent être normalisées par surface :

- `MAE_m2`
- `RMSE_m2`
- `WAPE_m2`

---

## 16. Bonnes pratiques d’utilisation

### 16.1 Toujours régénérer l’enrichissement après changement de drift

Si `consumptions_drifts.csv`, `usages.csv` ou `metertypes.csv` change, relancer :

```powershell
python .\run_enrich_from_consumptiondrift.py --db-dir "db"
```

### 16.2 Toujours relancer le nettoyage après enrichissement

Même si les nouvelles données sont supposées propres, le nettoyage reconstruit certaines colonnes dérivées et uniformise les formats.

```powershell
python -m pipeline.run_clean --config config.yaml --level all
```

### 16.3 Réentraîner après modification des données

Après enrichissement ou nettoyage, les modèles existants peuvent être obsolètes.

```powershell
python -m pipeline.run_train --config config.yaml --level all --target all
```

### 16.4 Lire les évaluations avant d’utiliser les prédictions

Avant de considérer une cible comme exploitable, consulter :

- `eval_<level>_<target>.csv`
- `eval_<level>_<target>_by_group.csv`
- les parity plots ;
- les séries temporelles.

### 16.5 Se méfier des usages rares

Certains usages peuvent être présents uniquement sur quelques sites ou zones. Ils peuvent produire des modèles instables si le nombre de points est trop faible.

---

## 17. Dépannage courant

### 17.1 La pipeline ne trouve pas un modèle

Message typique :

```text
[WARN] Missing meta/model for site_elecPcKwh. Skipping.
```

Cause probable :

- le modèle n’a pas été entraîné ;
- la cible a été ajoutée après le dernier entraînement ;
- la cible a été skip parce qu’il n’y avait pas assez de données.

Solution :

```powershell
python -m pipeline.run_train --config config.yaml --level all --target all
```

### 17.2 Un CSV est mal lu

Symptômes possibles :

- toutes les colonnes dans une seule colonne ;
- erreurs de parsing ;
- colonnes attendues absentes.

Cause probable :

- séparateur `,` vs `;` ;
- BOM UTF-8 ;
- valeurs `NULL` textuelles.

Solution :

- vérifier `io_utils.py` ;
- utiliser une lecture CSV robuste comma/semicolon.

### 17.3 `Input X contains infinity`

Cause probable :

- une valeur infinie ou trop grande s’est retrouvée dans les features de prédiction ;
- souvent via une prédiction autoregressive précédente.

Solution :

- nettoyer `X` avant `model.predict` ;
- remplacer `inf` par `NaN` ;
- caper les valeurs extrêmes ;
- laisser l’imputer gérer les NaN.

### 17.4 Tous les usages ne sont pas entraînés

Cause probable :

- la config pointe vers les fichiers non enrichis ;
- les scripts ne détectent pas dynamiquement les nouveaux usages ;
- certains usages n’ont pas assez de signal.

Solution :

1. Vérifier que `config.yaml` pointe vers les fichiers enrichis.
2. Vérifier `drift_usage_mapping.csv`.
3. Vérifier que les colonnes d’usage existent dans `sitehist_cleaned.csv` ou `zonehist_cleaned.csv`.
4. Relancer train avec `--target all`.

---

## 18. Ordre standard de travail recommandé

Pour repartir proprement depuis les données sources :

```powershell
python .\run_enrich_from_consumptiondrift.py --db-dir "db"
python -m pipeline.run_clean --config config.yaml --level all
python -m pipeline.run_train --config config.yaml --level all --target all
python -m pipeline.run_evaluate --config config.yaml --level all --target all
python -m pipeline.run_predict --config config.yaml --level all --target all
python -m pipeline.run_report --config config.yaml --level all --target all --site all
```

Pour travailler uniquement sur le niveau site :

```powershell
python -m pipeline.run_clean --config config.yaml --level site
python -m pipeline.run_train --config config.yaml --level site --target all
python -m pipeline.run_evaluate --config config.yaml --level site --target all
python -m pipeline.run_predict --config config.yaml --level site --target all
python -m pipeline.run_report --config config.yaml --level site --target all --site all
```

Pour travailler uniquement sur une cible :

```powershell
python -m pipeline.run_train --config config.yaml --level all --target elecTotalKwh
python -m pipeline.run_evaluate --config config.yaml --level all --target elecTotalKwh
python -m pipeline.run_predict --config config.yaml --level all --target elecTotalKwh
python -m pipeline.run_report --config config.yaml --level all --target elecTotalKwh --site all
```

---

## 19. Glossaire

### `site`

Niveau agrégé par bâtiment ou site.

### `zone`

Sous-niveau d’un site, identifié par `zoneId`.

### `history`

Historique journalier des mesures.

### `weather`

Données météo associées au site.

### `drift`

Table détaillée de consommations par usage, type de compteur et périmètre.

### `usage`

Type d’usage de consommation, par exemple éclairage, CVC, force, PC, process, ECS, chauffage.

### `meter type`

Famille de compteur, par exemple électricité, eau, eau chaude ou eau froide.

### `target`

Variable que le modèle cherche à prédire.

### `lag`

Valeur passée de la cible, par exemple J-1, J-2, J-7.

### `rolling window`

Statistiques calculées sur une fenêtre passée, par exemple moyenne ou médiane des 7 derniers jours.

### `valid_days`

Nombre de jours conservés pour la validation finale.

---

## 20. Notes importantes pour les futurs développeurs

- Ne pas pointer la config vers les fichiers non enrichis si l’objectif est d’exploiter les usages détaillés.
- Ne pas supprimer `drift_usage_mapping.csv` : il est utile pour auditer les colonnes créées.
- Ne pas interpréter `elecTotalAccurateKwh` comme forcément meilleur que `elecTotalKwh` sans regarder les métriques.
- Les usages rares doivent être interprétés avec prudence.
- Les graphes de comparaison doivent être lus avec les CSV d’outliers associés.
- Après toute modification de données source, relancer l’enrichissement, le nettoyage, puis l’entraînement.
