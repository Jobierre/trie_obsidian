# 🤖 Modèles recommandés

## Pour Mac Apple Silicon (MLX)

### LLM recommandés (du plus léger au plus puissant)

| Modèle | Taille | VRAM | Qualité | ID Hugging Face |
|--------|--------|------|---------|----------------|
| **Llama 3.2 3B** | 3B | ~4 GB | ⭐⭐⭐ | `mlx-community/Llama-3.2-3B-Instruct-4bit` |
| **Mistral 7B** | 7B | ~6 GB | ⭐⭐⭐⭐ | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` |
| **Qwen 2.5 7B** | 7B | ~6 GB | ⭐⭐⭐⭐ | `mlx-community/Qwen2.5-7B-Instruct-4bit` |
| **Llama 3.1 8B** | 8B | ~7 GB | ⭐⭐⭐⭐⭐ | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` |

**Recommandation pour Mac M4 (16 GB RAM)**: `mlx-community/Mistral-7B-Instruct-v0.3-4bit`

### Embeddings

| Modèle | Taille | Dimension | ID Hugging Face |
|--------|--------|-----------|----------------|
| **EmbeddingGemma** (recommandé) | 300M | 768 | `google/embeddinggemma-300m` |
| **all-MiniLM-L6-v2** | 22M | 384 | `sentence-transformers/all-MiniLM-L6-v2` |
| **all-mpnet-base-v2** | 110M | 768 | `sentence-transformers/all-mpnet-base-v2` |

### Embeddings MLX natifs (NPU Apple Silicon)

Pour exploiter le NPU du M4 et obtenir des performances optimales :

| Modèle | Taille | Dimension | ID Hugging Face |
|--------|--------|-----------|----------------|
| **BGE-small-en MLX** (recommandé) | 33M | 384 | `mlx-community/bge-small-en-v1.5-mlx` |
| **Nomic Embed MLX** | 137M | 768 | `mlx-community/nomic-embed-text-v1.5-mlx` |

**Configuration dans `.env` :**
```env
# Activer les embeddings MLX natifs (NPU)
EMBEDDING_MODEL_MLX=mlx-community/bge-small-en-v1.5-mlx
```

---

## Pour Windows/Linux avec CUDA

### LLM recommandés

| Modèle | Taille | VRAM (4-bit) | Qualité | ID Hugging Face |
|--------|--------|--------------|---------|----------------|
| **Ministral 3B** | 3B | ~4 GB | ⭐⭐⭐ | `mistralai/Ministral-3b-instruct-2412` |
| **Mistral 7B** | 7B | ~6 GB | ⭐⭐⭐⭐ | `mistralai/Mistral-7B-Instruct-v0.3` |
| **Llama 3.1 8B** | 8B | ~7 GB | ⭐⭐⭐⭐⭐ | `meta-llama/Llama-3.1-8B-Instruct` |
| **Qwen 2.5 14B** | 14B | ~12 GB | ⭐⭐⭐⭐⭐ | `Qwen/Qwen2.5-14B-Instruct` |

**Recommandation pour RTX 5070ti (16 GB VRAM)**: `mistralai/Mistral-7B-Instruct-v0.3`

---

## Configuration dans `.env`

```env
# Pour Mac M4 (avec NPU)
LLM_MODEL=mlx-community/Mistral-7B-Instruct-v0.3-4bit
EMBEDDING_MODEL=google/embeddinggemma-300m
EMBEDDING_MODEL_MLX=mlx-community/bge-small-en-v1.5-mlx  # Optionnel: embeddings NPU

# Pour Windows RTX 5070ti
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3
EMBEDDING_MODEL=google/embeddinggemma-300m
```

---

## Modèles MLX disponibles

Recherchez sur: https://huggingface.co/mlx-community

Filtres populaires:
- `mlx-community/Mistral-*`
- `mlx-community/Llama-*`
- `mlx-community/Qwen*`
- `mlx-community/*-4bit` (quantization 4-bit = moins de mémoire)

---

## Dépannage

### Erreur "TokenizersBackend does not exist"

Le modèle utilise un tokenizer custom non compatible avec MLX. Solutions:

1. **Utilisez un modèle MLX pré-converti** (recommandé):
   ```env
   LLM_MODEL=mlx-community/Mistral-7B-Instruct-v0.3-4bit
   ```

2. **Mettez à jour les dépendances**:
   ```bash
   pip install --upgrade transformers mlx-lm tokenizers
   ```

### Mémoire insuffisante

Si vous manquez de RAM/VRAM:

1. Utilisez un modèle plus petit (3B au lieu de 7B)
2. Augmentez la limite (Mac):
   ```bash
   sudo sysctl iogpu.wired_limit_mb=16384
   ```
3. Réduisez `LLM_SAMPLES_PER_CLUSTER` dans `.env`:
   ```env
   LLM_SAMPLES_PER_CLUSTER=3
   ```

---

## Performance vs Qualité

| Taille | Vitesse | Qualité catégorisation | RAM/VRAM |
|--------|---------|------------------------|----------|
| 3B | 🚀🚀🚀 | ⭐⭐⭐ | 4 GB |
| 7B | 🚀🚀 | ⭐⭐⭐⭐ | 6-8 GB |
| 8B | 🚀🚀 | ⭐⭐⭐⭐⭐ | 7-9 GB |
| 14B+ | 🚀 | ⭐⭐⭐⭐⭐ | 12+ GB |

**Pour 502 notes**, un modèle 7B est le meilleur compromis qualité/vitesse.
