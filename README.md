# 🗂️ Obsidian Organizer

**Organisation automatique de notes Obsidian avec IA locale (MLX + CUDA)**

Cet outil analyse vos notes Obsidian (.md) avec des embeddings IA, les regroupe par similarité, génère automatiquement des catégories pertinentes via un LLM, et met à jour le frontmatter YAML pour une utilisation avec le plugin [abstract-folder](https://github.com/RahmaniErfan/abstract-folder).

---

## ✨ Fonctionnalités

- 🧠 **Embeddings sémantiques** : [EmbeddingGemma-300M](https://huggingface.co/google/embeddinggemma-300m) pour capturer le sens de vos notes
- 🎯 **Clustering intelligent** : UMAP + HDBSCAN pour grouper les notes similaires
- 🤖 **Catégorisation par LLM** : [Ministral-3B](https://huggingface.co/mistralai/Ministral-3b-instruct-2412) génère des noms de catégories pertinents
- ⚡ **Multi-plateforme** : Support automatique MLX (Apple Silicon M1/M2/M3/M4) et CUDA (NVIDIA GPU)
- � **NPU Apple Silicon** : Embeddings MLX natifs pour exploiter le NPU du M4
- 👷 **Parallélisation** : Scan des fichiers et appels LLM parallélisés (configurable via `--workers`)
- 🔐 **Cache thread-safe** : Verrouillage fichier pour les accès concurrents
- �🔍 **Dry-run obligatoire** : Prévisualisez les changements avant application
- 📊 **Rapports détaillés** : Markdown et JSON

---

## 📋 Prérequis

### Général
- Python 3.10+
- Compte [Hugging Face](https://huggingface.co/) avec token d'API
- Plugin Obsidian [abstract-folder](https://github.com/RahmaniErfan/abstract-folder) installé

### Matériel

#### Option 1: Mac Apple Silicon (M1/M2/M3/M4)
- macOS 13+ recommandé
- 8 GB RAM minimum (16 GB recommandé)

#### Option 2: Windows/Linux avec GPU NVIDIA
- CUDA 12.1+
- 8 GB VRAM minimum
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) installé

---

## 🚀 Installation

### 1. Cloner/Télécharger le projet

```bash
cd /Users/jordanmirmand/code/tire_obsidian
```

### 2. Créer un environnement virtuel

```bash
python3 -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

### 3. Installer les dépendances

#### Sur Mac Apple Silicon (MLX):
```bash
pip install -r requirements-mlx.txt
```

#### Sur Windows/Linux avec CUDA:
```bash
# D'abord installer PyTorch avec CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Puis les autres dépendances
pip install -r requirements-cuda.txt
```

### 4. Configuration

Copiez le fichier d'exemple et configurez:

```bash
cp .env.example .env
nano .env  # ou votre éditeur préféré
```

**Variables essentielles dans `.env`:**

```env
# Token Hugging Face (obligatoire)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx

# Seuil minimum de notes pour créer une catégorie (configurable)
MIN_CLUSTER_SIZE=5

# Chemin vers vos notes (par défaut: ./notes)
VAULT_PATH=./notes
```

Pour obtenir un token Hugging Face:
1. Créez un compte sur https://huggingface.co/
2. Allez dans **Settings → Access Tokens**
3. Créez un token avec permission **Read**

### 5. Copier vos notes Obsidian

```bash
# Copiez vos notes .md dans le dossier notes/
cp -r /chemin/vers/votre/vault/*.md ./notes/
```

Ou modifiez `VAULT_PATH` dans `.env` pour pointer vers votre vault existant.

---

## 📖 Utilisation

### Mode Dry-Run (Preview uniquement)

**Par défaut, aucun fichier n'est modifié.** Le script génère un rapport des changements prévus:

```bash
python -m src.main
```

Cela va:
1. Scanner vos notes dans `notes/`
2. Générer les embeddings avec EmbeddingGemma-300M
3. Regrouper les notes similaires (clustering)
4. Générer des noms de catégories via Ministral-3B
5. Créer un rapport dans `output/dry_run_report_YYYYMMDD_HHMMSS.md`

### Appliquer les changements

Une fois le rapport vérifié:

```bash
python -m src.main --apply
```

⚠️ **Faites une sauvegarde de votre vault avant !**

### Options avancées

```bash
# Réutiliser les embeddings déjà calculés (gain de temps)
python -m src.main --skip-embeddings

# Générer un rapport JSON au lieu de Markdown
python -m src.main --report-format json

# Générer les deux formats
python -m src.main --report-format both

# Contrôler le nombre de workers (0 = auto-détection)
python -m src.main --workers 4

# Effacer le cache et tout reclassifier
python -m src.main --clear-cache
```

---

## 📊 Exemple de rapport

Après un dry-run, vous obtenez un rapport comme celui-ci:

```markdown
# 📋 Rapport Dry-Run - Organisation Obsidian

**Date**: 14/12/2025 15:30:42
**Mode**: DRY-RUN (aucun fichier modifié)

---

## 📊 Statistiques

- **Total de notes**: 502
- **Nouvelles catégories**: 487
- **Catégories mises à jour**: 15
- **Nombre de catégories uniques**: 23

---

## 📁 Changements par catégorie

### Tech - Python (45 notes)

- ✨ **django-deployment.md** - Nouvelle catégorie
- ✨ **pandas-tips.md** - Nouvelle catégorie
- 🔄 **flask-api.md** - `Dev` → `Tech - Python`
...

### Citations - Philosophie (18 notes)

- ✨ **nietzsche-quote.md** - Nouvelle catégorie
- ✨ **stoicism.md** - Nouvelle catégorie
...
```

---

## ⚙️ Configuration avancée

### Ajuster les paramètres de clustering

Dans `.env`:

```env
# Clustering
MIN_CLUSTER_SIZE=5          # Minimum 5 notes pour créer une catégorie
HDBSCAN_MIN_SAMPLES=3       # Densité minimale

# UMAP (réduction dimensionnelle)
UMAP_N_COMPONENTS=15        # Dimensions finales
UMAP_N_NEIGHBORS=15         # Voisinage local
UMAP_MIN_DIST=0.1           # Distance minimale

# LLM
LLM_SAMPLES_PER_CLUSTER=5   # Échantillons envoyés au LLM par cluster
```

### Changer de modèles

```env
# Utiliser un autre modèle d'embeddings
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2

# Utiliser un autre LLM (si disponible en MLX ou via Transformers)
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3
```

---

## 🔧 Résolution de problèmes

### Erreur: `HF_TOKEN manquant`

→ Créez un fichier `.env` à partir de `.env.example` et ajoutez votre token Hugging Face

### Erreur: `Aucun fichier .md trouvé`

→ Copiez vos notes dans le dossier `notes/` ou configurez `VAULT_PATH` dans `.env`

### Erreur: `MLX non installé` (sur Mac)

→ Installez MLX: `pip install -r requirements-mlx.txt`

### Erreur: `CUDA not available` (sur Windows/Linux)

→ Vérifiez que:
1. Vous avez une GPU NVIDIA
2. CUDA Toolkit est installé
3. PyTorch est installé avec CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

### Le LLM génère des catégories bizarres

→ Essayez:
- Augmenter `LLM_SAMPLES_PER_CLUSTER` pour donner plus de contexte
- Réduire `MIN_CLUSTER_SIZE` si trop de notes vont dans "Divers"
- Vérifier que vos notes ont du contenu significatif

### Mémoire insuffisante

#### Sur Mac:
```bash
# Augmenter la limite de mémoire wired (nécessite sudo)
sudo sysctl iogpu.wired_limit_mb=16384
```

#### Sur Windows/Linux:
→ Réduire la quantization (déjà en 4-bit) ou utiliser un modèle plus petit

---

## 📁 Structure du projet

```
tire_obsidian/
├── notes/                    # Vos notes Obsidian (.md)
├── output/                   # Rapports et embeddings
│   ├── dry_run_report_*.md
│   ├── dry_run_report_*.json
│   ├── classification_cache.json  # Cache des classifications
│   └── embeddings.npy
├── src/
│   ├── __init__.py
│   ├── main.py              # Point d'entrée
│   ├── config.py            # Configuration + détection backend + workers
│   ├── embeddings.py        # EmbeddingGenerator + MLXEmbeddingGenerator (NPU)
│   ├── llm.py               # Ministral wrapper (MLX/CUDA)
│   ├── clustering.py        # UMAP + HDBSCAN
│   ├── categorizer.py       # Classification LLM parallélisée
│   ├── cache.py             # Cache thread-safe avec FileLock
│   ├── frontmatter_manager.py  # Lecture/écriture YAML parallélisée
│   └── dry_run.py           # Rapports
├── categories.yaml          # Catégories personnalisables
├── .env                     # Configuration (créer depuis .env.example)
├── .env.example
├── requirements-mlx.txt     # Dépendances Mac (inclut psutil, filelock)
├── requirements-cuda.txt    # Dépendances Windows/Linux
└── README.md
```

---

## 🔮 Utilisation future (nouvelles notes)

Pour intégrer de nouvelles notes:

1. Copiez les nouvelles notes dans `notes/`
2. Relancez l'analyse:
   ```bash
   python -m src.main
   ```
3. Le script détectera automatiquement si de nouvelles catégories doivent être créées
4. Vérifiez le rapport, puis appliquez avec `--apply`

Pour éviter de tout recalculer:
```bash
python -m src.main --skip-embeddings
```
(Les nouvelles notes seront ajoutées au clustering existant)

---

## 🎯 Plugin abstract-folder

Une fois les catégories ajoutées au frontmatter, utilisez [abstract-folder](https://github.com/RahmaniErfan/abstract-folder) dans Obsidian:

### Format du frontmatter généré:

```yaml
---
parent: "[[Tech - Python]]"
---
```

### Visualisation dans Obsidian:

Le plugin abstract-folder créera une vue hiérarchique où:
- Les catégories deviennent des "dossiers virtuels"
- Les notes apparaissent dans leur catégorie respective
- Une note peut avoir plusieurs parents (multi-parentage)

---

## 🤝 Contribution

Ce projet est personnel mais les suggestions sont bienvenues !

---

## 📝 Licence

MIT License - Voir LICENSE (à créer si besoin)

---

## 🙏 Crédits

- [EmbeddingGemma-300M](https://huggingface.co/google/embeddinggemma-300m) by Google
- [Ministral-3B](https://huggingface.co/mistralai/Ministral-3b-instruct-2412) by Mistral AI
- [abstract-folder](https://github.com/RahmaniErfan/abstract-folder) by Erfan Rahmani
- [MLX](https://github.com/ml-explore/mlx) by Apple
- [sentence-transformers](https://www.sbert.net/), [UMAP](https://umap-learn.readthedocs.io/), [HDBSCAN](https://hdbscan.readthedocs.io/)

---

## 📧 Contact

Jordan Mirmand - [@jordanmirmand](https://github.com/jordanmirmand)

---

**Bon classement ! 🎉**
