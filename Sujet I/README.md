# Générateur de rapports AMO Build4Use

Ce projet génère des rapports PowerPoint d'audit à partir de pages OneNote terrain. La version actuelle du générateur repose sur un workflow "Page Cards" : chaque page OneNote traitée devient une slide ou une source de slide, avec extraction du texte, des images, des légendes et des notes vocales transcrites.

Le pipeline complet est le suivant :

```text
Microsoft OneNote
    -> export local via Microsoft Graph
    -> conversion Markdown/HTML en JSON structurés
    -> construction d'un assemblage Page Cards
    -> humanisation et consolidation du texte par LLM
    -> rendu PowerPoint avec template
```

Le but de ce README est de permettre à une personne qui ne connaît pas le code de configurer l'environnement et de générer un rapport de bout en bout.

---

## 1. Arborescence attendue

La version nettoyée du projet doit ressembler à ceci :

```text
Sujet I/
  .env
  requirements.txt
  process_onenote.py
  render_report_pptx.py

  local_webapp_ui/
    app_dual.py
    start_dual.bat
    ui/
      onenote_cloud.py

  onenote_exporter/
    __init__.py
    __main__.py
    cli.py
    auth.py
    graph.py
    exporter.py
    markdown.py

  src/
    llm_client.py
    page_text.py
    page_images.py
    image_selection.py

    page_cards/
      run_page_cards.py
      build_page_cards_assembled.py
      humanize_page_cards.py

    legacy/
      legacy_runner.py
      section_names.py

  input/
    config/
      style_card.md
      presets.json              # optionnel
    templates/
      Templates Slides.pptx
      slide_types_template_slides.json
    onenote-exporter/
      output/
      cache/

  process/
    onenote/
    page_cards/

  output/
    reports/
```

Les dossiers `process/` et `output/` peuvent être vidés régulièrement : ils contiennent les artefacts générés. Les fichiers de code, les templates, `.env` et `requirements.txt` doivent être conservés.

---

## 2. Prérequis système

### 2.1 Python

Installer Python 3.11 ou plus récent. Le projet a été utilisé avec Python 3.14 dans certains tests locaux, mais une version 3.11+ est généralement suffisante.

Vérification :

```bash
python --version
```

### 2.2 FFmpeg

FFmpeg est nécessaire pour transcrire les notes vocales OneNote. Sans FFmpeg, le traitement audio échoue avec un statut de transcription en échec.

Vérification :

```bash
ffmpeg -version
```

Si la commande n'est pas reconnue, installer FFmpeg puis ajouter le dossier `bin` au `PATH` Windows.

### 2.3 PowerPoint non requis pour générer le PPTX

Le rendu PowerPoint est fait par Python via `python-pptx`. PowerPoint n'a pas besoin d'être ouvert pour générer le fichier. En revanche, si le fichier de sortie est déjà ouvert dans PowerPoint, l'écriture peut échouer ou produire un fichier alternatif.

---

## 3. Installer l'environnement Python

Depuis la racine du projet :

```bash
cd "C:\Users\OctaveCINQUANTA\CR SYSTEM\PROJETS Build 4 Use - _octave\Sujet I"
```

Créer un environnement virtuel :

```bash
python -m venv .venv
```

Activer l'environnement :

```bash
.venv\Scripts\activate
```

Installer les dépendances :

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Le fichier `requirements.txt` doit contenir au minimum :

```text
streamlit
msal
requests
python-dotenv
beautifulsoup4
python-pptx
Pillow
numpy
pyyaml
```

Pour la transcription audio locale, ajouter selon le moteur utilisé :

```text
faster-whisper
```

ou :

```text
openai-whisper
```

Si `faster-whisper` est utilisé, il est recommandé de tester la transcription sur une petite page OneNote avant de lancer un gros notebook.

---

## 4. Configurer le fichier `.env`

Créer un fichier `.env` à la racine du projet.

Exemple :

```env
# Microsoft Graph / OneNote
MS_TENANT_ID=common
MS_CLIENT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
GRAPH_SCOPES=Notes.Read offline_access openid profile

# Hugging Face
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
HF_MODEL=mistralai/Mistral-7B-Instruct-v0.3

# Optionnel : réduire les risques de throttling Microsoft Graph
GRAPH_MIN_DELAY=0.3
```

Adapter les noms exacts des variables si `llm_client.py` ou `onenote_exporter` utilisent déjà d'autres clés. L'idée importante est de ne jamais écrire les secrets directement dans le code.

Ajouter `.env` dans `.gitignore` si le projet est versionné :

```text
.env
input/onenote-exporter/cache/
```

---

## 5. Configurer Hugging Face pour l'humanisation LLM

L'humanisation des slides se fait via un modèle Hugging Face configuré dans `llm_client.py` et appelé depuis `src/page_cards/humanize_page_cards.py`.

### 5.1 Créer un token Hugging Face

1. Se connecter à Hugging Face.
2. Aller dans les paramètres du compte.
3. Créer un token d'accès.
4. Copier le token dans `.env` :

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 5.2 Choisir un modèle

Dans `.env`, définir le modèle voulu :

```env
HF_MODEL=mistralai/Mistral-7B-Instruct-v0.3
```

Le nom exact du modèle dépend de ce qui est supporté par `llm_client.py` et de l'accès disponible sur Hugging Face.

### 5.3 Vérifier que le LLM fonctionne

Lancer une humanisation sur un `assembled_page_cards.json` existant :

```bash
python src/page_cards/humanize_page_cards.py ^
  --assembled process/page_cards/assembled_page_cards.json ^
  --out process/page_cards/assembled_page_cards_humanized.json
```

Si tout fonctionne, le fichier de sortie doit contenir des champs comme :

```json
"source_fact_count": 4,
"final_bullet_count": 4,
"coverage_score": 0.75
```

Ces champs servent à vérifier que la génération ne perd pas trop d'information.

---

## 6. Configurer Microsoft Graph pour OneNote

Le téléchargement OneNote utilise Microsoft Graph via le package local `onenote_exporter` et l'interface `local_webapp_ui/ui/onenote_cloud.py`.

### 6.1 Créer une application Azure / Entra ID

Dans le portail Azure / Entra ID :

1. Créer une nouvelle application.
2. Noter le `Client ID`.
3. Définir le tenant à utiliser ; si l'application doit fonctionner en device flow sur plusieurs comptes, `common` peut être utilisé selon la configuration de l'organisation.
4. Activer le flux public client / device code si nécessaire.
5. Ajouter les permissions Microsoft Graph nécessaires à la lecture OneNote.

Pour un accès aux notebooks accessibles par l'utilisateur connecté, utiliser des permissions déléguées telles que :

```text
Notes.Read
Notes.Read.All
offline_access
openid
profile
```

Selon la politique de l'organisation, certaines permissions peuvent nécessiter un consentement administrateur. Utiliser le principe du moindre privilège : commencer par `Notes.Read`, puis augmenter uniquement si nécessaire.

### 6.2 Renseigner `.env`

Exemple :

```env
MS_TENANT_ID=common
MS_CLIENT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
GRAPH_SCOPES=Notes.Read offline_access openid profile
```

Si l'export échoue pour des raisons de scopes, vérifier la sortie console : le token affichera souvent les scopes réellement accordés.

---

## 7. Lancer l'interface web

Depuis la racine du projet ou depuis `local_webapp_ui/` selon ton organisation :

```bash
cd local_webapp_ui
start_dual.bat
```

Ou directement :

```bash
python -m streamlit run app_dual.py
```

L'interface contient les zones principales suivantes :

```text
Page Cards
OneNote Microsoft Graph
Legacy éventuel / ancien pipeline
```

La version courante doit utiliser principalement :

```text
Page Cards
OneNote Microsoft Graph
```

---

## 8. Télécharger un notebook OneNote

Dans l'interface web :

1. Ouvrir l'onglet OneNote Microsoft Graph.
2. Cliquer sur la liste des notebooks.
3. Se connecter via le device code Microsoft si demandé.
4. Sélectionner ou saisir le notebook à exporter.
5. Lancer l'export.

Le notebook exporté doit apparaître dans :

```text
input/onenote-exporter/output/<NomDuNotebook>/
```

Chaque page OneNote est exportée sous forme de Markdown, avec les assets associés :

```text
input/onenote-exporter/output/<NomDuNotebook>/pages/*.md
input/onenote-exporter/output/<NomDuNotebook>/<page_id>/res-*.jpg
input/onenote-exporter/output/<NomDuNotebook>/<page_id>/aud-*.m4a
```

Le nom du dossier notebook est important : c'est ce nom qui devra ensuite être utilisé dans le traitement et la génération.

---

## 9. Traiter l'export OneNote

Le traitement convertit les Markdown OneNote en JSON exploitables par le générateur.

### 9.1 Via l'interface web

Dans l'interface :

1. Aller dans la section de traitement OneNote.
2. Renseigner le notebook à traiter. Exemple :

```text
Savills2
```

3. Vérifier l'input root :

```text
input/onenote-exporter/output
```

4. Vérifier l'output root :

```text
process/onenote
```

5. Cocher `Transcrire audio` si les notes vocales doivent être transcrites.
6. Lancer le traitement.

### 9.2 En ligne de commande

```bash
python process_onenote.py Savills2 ^
  --input input/onenote-exporter/output ^
  --out process/onenote ^
  --transcribe
```

Optionnel, si tu veux copier les assets dans le dossier process :

```bash
python process_onenote.py Savills2 ^
  --input input/onenote-exporter/output ^
  --out process/onenote ^
  --transcribe ^
  --copy-assets
```

### 9.3 Résultat attendu

Le traitement doit créer :

```text
process/onenote/Savills2/manifest.json
process/onenote/Savills2/pages/*.json
process/onenote/Savills2/errors.jsonl
```

Chaque page JSON contient notamment :

```json
{
  "title": "Présentation du site",
  "blocks": [
    {"type": "heading", "text": "Présentation du site"},
    {"type": "audio", "path": "...m4a", "transcript": "..."},
    {"type": "image", "path": "...jpg"}
  ],
  "assets": {
    "images": [],
    "audio": []
  }
}
```

Si une note vocale est bien transcrite, le bloc audio doit contenir :

```json
"transcript_meta": {
  "engine": "faster-whisper",
  "language": "fr",
  "ffmpeg": true,
  "status": "ok"
}
```

---

## 10. Règles de rédaction dans OneNote

Pour obtenir un bon rapport, les pages OneNote doivent être structurées proprement.

### 10.1 Titre de page

Chaque page doit avoir un titre clair. Exemple :

```text
Présentation du site
Contexte de l'opération
Aérothermes
Supervision GTB
```

### 10.2 Texte de contenu

Le texte normal devient du contenu de slide.

Exemple :

```text
La supervision GTB repose sur une interface web accessible localement.
Les historiques de température sont disponibles pour les principales zones.
```

### 10.3 Notes vocales

Les notes vocales sont traitées comme du texte après transcription. Elles peuvent donc suffire à créer une slide, même sans image.

### 10.4 Légendes photo

Pour éviter qu'une légende devienne une puce de slide, utiliser un marqueur explicite sous l'image :

```text
Légende: Armoire GTB principale
```

Ou :

```text
Caption: Vue supervision zone extension
```

Règle actuelle :

```text
Image
Légende: texte de légende
```

La légende doit être placée juste sous l'image. Le texte qui ne commence pas par `Légende:` reste du contenu de slide.

---

## 11. Générer les Page Cards

La génération Page Cards transforme les pages JSON en assemblage exploitable par le renderer PPTX.

### 11.1 Via l'interface web

Dans l'onglet Page Cards :

1. Choisir le notebook local.
2. Renseigner le Case ID.
3. Renseigner la section OneNote à utiliser.
4. Régler `Max images`.
5. Régler `Max bullets`.
6. Cocher ou non le retraitement OneNote.
7. Cocher ou non la transcription audio.
8. Lancer Page Cards.

Exemple :

```text
Notebook: Savills2
Case ID: Savills2
Section OneNote: Visite de site
Max images: 6
Max bullets: 10
```

### 11.2 En ligne de commande : assemblage brut

```bash
python src/page_cards/build_page_cards_assembled.py ^
  --pages-index process/onenote/Savills2/manifest.json ^
  --out process/page_cards/assembled_page_cards.json ^
  --case-id Savills2 ^
  --section-name "Visite de site" ^
  --max-images 6 ^
  --max-bullets 10
```

Le fichier généré contient des slides au format JSON :

```text
process/page_cards/assembled_page_cards.json
```

Chaque slide contient typiquement :

```json
{
  "type": "CONTENT_TEXT_IMAGES",
  "title": "Aérothermes",
  "bullets": "- ...",
  "images": [
    {"path": "...jpg", "caption": "..."}
  ],
  "raw_text": "source complète avant humanisation"
}
```

---

## 12. Humaniser et sécuriser le texte

L'humanisation transforme les notes brutes en puces professionnelles.

La version actuelle cherche à :

```text
- préserver l'information source,
- refuser les sorties trop pauvres,
- éviter les "Rien à signaler" abusifs,
- améliorer la cohérence intra-slide,
- conserver des métriques qualité.
```

Commande :

```bash
python src/page_cards/humanize_page_cards.py ^
  --assembled process/page_cards/assembled_page_cards.json ^
  --out process/page_cards/assembled_page_cards_humanized.json
```

Paramètres utiles :

```bash
--temperature 0.2
--max-tokens 700
--top-p 1.0
--sleep 0.0
```

Exemple complet :

```bash
python src/page_cards/humanize_page_cards.py ^
  --assembled process/page_cards/assembled_page_cards.json ^
  --out process/page_cards/assembled_page_cards_humanized.json ^
  --temperature 0.2 ^
  --max-tokens 700
```

### 12.1 Champs de contrôle qualité

Après humanisation, chaque slide peut contenir :

```json
"source_fact_count": 5,
"final_bullet_count": 5,
"coverage_score": 0.6,
"llm_rejected_reason": "coverage_or_insufficient_output"
```

Interprétation :

```text
source_fact_count   nombre de faits détectés dans la source
final_bullet_count  nombre de puces finales
coverage_score      score indicatif de couverture entre source et sortie
llm_rejected_reason présent si le LLM a produit une sortie trop pauvre
```

Si `llm_rejected_reason` apparaît, ce n'est pas forcément une erreur : cela signifie que le garde-fou a remplacé une sortie LLM insuffisante par un fallback plus fidèle aux sources.

---

## 13. Générer le PowerPoint

Le rendu PPTX utilise :

```text
render_report_pptx.py
input/templates/Templates Slides.pptx
input/templates/slide_types_template_slides.json
process/page_cards/assembled_page_cards_humanized.json
```

Commande :

```bash
python render_report_pptx.py ^
  --template "input/templates/Templates Slides.pptx" ^
  --assembled process/page_cards/assembled_page_cards_humanized.json ^
  --out output/reports/Savills2/Rapport_Audit.pptx ^
  --slide-types input/templates/slide_types_template_slides.json
```

Le fichier final est créé ici :

```text
output/reports/Savills2/Rapport_Audit.pptx
```

Si le fichier existe déjà et est ouvert dans PowerPoint, fermer PowerPoint puis relancer la commande.

---

## 14. Workflow complet recommandé

### 14.1 Workflow interface web

1. Lancer l'interface :

```bash
cd local_webapp_ui
start_dual.bat
```

2. Dans OneNote Microsoft Graph :
   - se connecter,
   - lister les notebooks,
   - exporter le notebook.

3. Traiter l'export :
   - notebook : `Savills2`,
   - input root : `input/onenote-exporter/output`,
   - output root : `process/onenote`,
   - transcription audio : activée.

4. Dans Page Cards :
   - notebook : `Savills2`,
   - case ID : `Savills2`,
   - section : `Visite de site`,
   - max images : `6`,
   - max bullets : `10`,
   - lancer la génération.

5. Vérifier le PPTX dans :

```text
output/reports/<case_id>/
```

### 14.2 Workflow CLI complet

```bash
python -m onenote_exporter.cli ^
  --config .env ^
  --notebook "Savills2" ^
  --output-dir input/onenote-exporter/output ^
  --formats md
```

```bash
python process_onenote.py Savills2 ^
  --input input/onenote-exporter/output ^
  --out process/onenote ^
  --transcribe
```

```bash
python src/page_cards/build_page_cards_assembled.py ^
  --pages-index process/onenote/Savills2/manifest.json ^
  --out process/page_cards/assembled_page_cards.json ^
  --case-id Savills2 ^
  --section-name "Visite de site" ^
  --max-images 6 ^
  --max-bullets 10
```

```bash
python src/page_cards/humanize_page_cards.py ^
  --assembled process/page_cards/assembled_page_cards.json ^
  --out process/page_cards/assembled_page_cards_humanized.json
```

```bash
python render_report_pptx.py ^
  --template "input/templates/Templates Slides.pptx" ^
  --assembled process/page_cards/assembled_page_cards_humanized.json ^
  --out output/reports/Savills2/Rapport_Audit.pptx ^
  --slide-types input/templates/slide_types_template_slides.json
```

---

## 15. Dépannage

### 15.1 `Notebook folder not found`

Erreur typique :

```text
Notebook folder not found: input/onenote-exporter/output/test
```

Cause : le nom du notebook demandé ne correspond pas au nom du dossier exporté.

Solution : vérifier les dossiers disponibles :

```bash
dir input\onenote-exporter\output
```

Puis relancer avec le bon notebook :

```bash
python process_onenote.py Savills2 --input input/onenote-exporter/output --out process/onenote --transcribe
```

### 15.2 Les audios ne sont pas transcrits

Vérifier :

```bash
ffmpeg -version
```

Puis ouvrir un fichier page JSON dans :

```text
process/onenote/<Notebook>/pages/*.json
```

Un bloc audio correctement transcrit doit contenir :

```json
"type": "audio",
"transcript": "...",
"transcript_meta": {
  "status": "ok"
}
```

Si `status` vaut `failed`, regarder `error`.

### 15.3 Le rapport indique `Rien à signaler (notes insuffisantes)` alors que les notes existent

Vérifier dans l'assembled humanisé :

```json
"raw_text"
"raw_bullets"
"source_fact_count"
"coverage_score"
"llm_rejected_reason"
```

Si `raw_text` ou `raw_bullets` contient du texte mais `source_fact_count` vaut `0`, le problème vient de l'extraction de faits.

Si `source_fact_count` est positif mais `bullets` est pauvre, vérifier `llm_rejected_reason` et `llm_raw_output`.

### 15.4 Les images n'apparaissent pas

Vérifier que les chemins image référencés dans `assembled_page_cards_humanized.json` existent réellement.

Exemple :

```json
"images": [
  {"path": ".../res-xxxx.jpg", "caption": "..."}
]
```

Si les chemins sont relatifs, le renderer tente de les résoudre depuis `process/`, `input/` et le dossier de l'assembled.

### 15.5 Les légendes sont mauvaises

Règle recommandée dans OneNote :

```text
Légende: texte de légende
```

La légende doit être située immédiatement sous l'image. Un texte normal non préfixé peut être traité comme contenu de slide.

### 15.6 Microsoft Graph renvoie une erreur 429

Cela signifie que trop de requêtes ont été envoyées en peu de temps.

Actions possibles :

```env
GRAPH_MIN_DELAY=0.3
```

Puis relancer l'export. Si le notebook est très gros, utiliser un notebook ou une section plus ciblée.

### 15.7 Hugging Face refuse la requête

Vérifier :

```env
HF_TOKEN=hf_xxx
HF_MODEL=...
```

Puis vérifier que le token a accès au modèle sélectionné.

---

## 16. Bonnes pratiques d'utilisation terrain

### 16.1 Une page OneNote = un sujet clair

Bon exemple :

```text
Aérothermes
Supervision GTB
Présentation du site
Contexte de l'opération
```

Mauvais exemple :

```text
Divers
Photos
Notes audit
```

### 16.2 Mettre les informations importantes en texte ou en audio clair

Le générateur sait traiter :

```text
- texte écrit,
- audio transcrit,
- images,
- légendes explicites.
```

Mais il ne peut pas inventer les informations absentes.

### 16.3 Préférer plusieurs faits courts à une note trop vague

Bon :

```text
La supervision ne présente pas de synoptique technique.
Les historiques de température sont disponibles sur novembre 2025.
Les plans d'implantation ne sont pas accessibles depuis l'interface.
```

Moins bon :

```text
Supervision pas ouf, voir photos.
```

### 16.4 Vérifier les slides audio-only

Les slides composées uniquement d'audio doivent être surveillées avec :

```json
"source_fact_count"
"coverage_score"
```

Si `source_fact_count` est élevé mais le texte final est pauvre, la sortie LLM a probablement été rejetée ou le fallback doit être amélioré.

---

## 17. Fichiers à ne pas supprimer

Pour la version actuelle, conserver :

```text
local_webapp_ui/app_dual.py
local_webapp_ui/start_dual.bat
local_webapp_ui/ui/onenote_cloud.py
onenote_exporter/*
process_onenote.py
render_report_pptx.py
src/llm_client.py
src/page_text.py
src/page_images.py
src/image_selection.py
src/page_cards/run_page_cards.py
src/page_cards/build_page_cards_assembled.py
src/page_cards/humanize_page_cards.py
src/legacy/legacy_runner.py
src/legacy/section_names.py
input/config/style_card.md
input/templates/Templates Slides.pptx
input/templates/slide_types_template_slides.json
requirements.txt
.env
```

Les deux fichiers `src/legacy/legacy_runner.py` et `src/legacy/section_names.py` peuvent être supprimés uniquement si `app_dual.py` est modifié pour ne plus les importer.

---

## 18. Résumé rapide pour générer un rapport

Version courte :

```text
1. Activer l'environnement Python.
2. Lancer local_webapp_ui/start_dual.bat.
3. Exporter le notebook via Microsoft Graph.
4. Traiter le notebook avec transcription audio.
5. Générer Page Cards avec le bon notebook, case ID et section.
6. Vérifier assembled_page_cards_humanized.json.
7. Générer le PPTX.
8. Ouvrir le rapport dans output/reports/<case_id>/.
```

Si une étape échoue, regarder d'abord :

```text
- nom exact du notebook,
- présence de ffmpeg,
- validité du token Microsoft Graph,
- validité du token Hugging Face,
- contenu de raw_text/raw_bullets,
- coverage_score des slides générées.
```
