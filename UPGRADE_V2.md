# 🆕 Nouveau Système de Classification (v2.0)

## 🎯 Changements majeurs

**Avant:** Clustering automatique → catégories générées → souvent incohérentes  
**Maintenant:** Découverte supervisée → classification individuelle → précis et cohérent

---

## 🚀 Workflow en 2 étapes

### Étape 1: Découvrir les catégories

```bash
python -m src.main --discover
```

- Analyse 50 notes aléatoires de votre vault
- Le LLM propose 8 catégories adaptées à vos thèmes
- Génère `categories.yaml` à la racine du projet

**Options:**
```bash
# Proposer 10 catégories au lieu de 8
python -m src.main --discover --count 10
```

### Étape 2: Classifier les notes

```bash
# Dry-run (prévisualisation)
python -m src.main

# Appliquer les changements
python -m src.main --apply
```

- Chaque note est classifiée **individuellement** par le LLM
- Les classifications sont mises en cache (rapide sur les relances)
- Dry-run par défaut, `--apply` pour écrire dans les fichiers

**Options:**
```bash
# Effacer le cache et tout reclassifier
python -m src.main --clear-cache

# Générer un rapport JSON
python -m src.main --report-format json
```

---

## 📁 Fichier `categories.yaml`

Exemple de fichier généré:

```yaml
categories:
  - name: "Tech - Backup"
    description: "Sauvegarde de données, Plakar, S3, rsync, snapshots"
  
  - name: "Dev - Programmation"
    description: "Python, JavaScript, frameworks, code, tutoriels"
  
  - name: "Maison - Travaux"
    description: "Rénovation, carrelage, salle de bain, matériaux"
  
  - name: "IA - Outils"
    description: "Claude, GPT, prompts, MCP, automatisation IA"
  
  - name: "Finance"
    description: "Investissement, trading, broker, retraite"
  
  - name: "Perso"
    description: "Notes personnelles, rendez-vous, messages"
  
  - name: "Divers"
    description: "Tout ce qui ne rentre pas ailleurs"
```

**Vous pouvez:**
- ✅ Ajouter des catégories
- ✅ Supprimer des catégories
- ✅ Modifier les noms et descriptions
- ✅ Réorganiser l'ordre

Relancez simplement `python -m src.main` après modification.

---

## 💾 Système de cache

Le fichier `output/classification_cache.json` stocke les classifications déjà effectuées.

**Avantages:**
- ⚡ Relances ultra-rapides
- 💰 Économise des appels LLM
- 🔄 Seules les nouvelles notes sont classifiées

**Gestion:**
```bash
# Reclassifier toutes les notes
python -m src.main --clear-cache

# Le cache est automatiquement invalidé si:
#   - Le contenu d'une note change
#   - categories.yaml est modifié
```

---

## 📊 Exemple de session complète

```bash
# 1. Découvrir les catégories (une seule fois)
$ python -m src.main --discover --count 8
# → génère categories.yaml

# 2. (Optionnel) Éditer categories.yaml selon vos préférences

# 3. Classifier les notes (dry-run)
$ python -m src.main
# → génère output/dry_run_report_YYYYMMDD_HHMMSS.md

# 4. Vérifier le rapport, puis appliquer
$ python -m src.main --apply
# ✅ 75 notes organisées

# 5. Plus tard: ajouter de nouvelles notes
# Les nouvelles notes seront classifiées, les anciennes récupérées du cache
$ python -m src.main
```

---

## 🆚 Comparaison Ancien vs Nouveau

| Critère | Ancien (Clustering) | Nouveau (Classification) |
|---------|---------------------|---------------------------|
| **Précision** | ❌ Clusters hétérogènes | ✅ Chaque note analysée individuellement |
| **Contrôle** | ❌ Catégories auto-générées | ✅ Vous validez les catégories |
| **Cohérence** | ❌ "Torrent + Carrelage + Finance" | ✅ Chaque note dans SA catégorie |
| **Vitesse 1ère fois** | ⚡ ~2 min | 🐢 ~5-10 min (75 notes) |
| **Vitesse relance** | 🐢 Toujours pareil | ⚡ Instantané (cache) |
| **Nouvelles notes** | 🔄 Tout recalculer | ✅ Seules les nouvelles |

---

## 🔧 Dépannage

### Erreur: "Fichier categories.yaml introuvable"
→ Lancez d'abord `python -m src.main --discover`

### Le LLM propose des catégories bizarres
→ Relancez `--discover` avec `--count` différent, ou éditez `categories.yaml` manuellement

### Classification lente
→ C'est normal la première fois (1 appel LLM par note). Ensuite le cache accélère tout.

### Une note est mal classée
→ Éditez `categories.yaml` pour clarifier la description, puis `--clear-cache` et relancez

---

## 📖 Ancienne documentation

L'ancien système (clustering HDBSCAN) est documenté dans [README_OLD.md](README_OLD.md).

---

**Questions ? Ouvrez une issue sur GitHub !** 🚀
