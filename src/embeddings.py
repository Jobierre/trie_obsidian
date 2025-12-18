"""
Génération d'embeddings avec support MLX natif et CUDA/CPU fallback
"""
from typing import List, Dict
import numpy as np
from tqdm import tqdm
from pathlib import Path

from .config import config


class MLXEmbeddingGenerator:
    """
    Génère des embeddings en utilisant MLX natif pour exploiter le NPU Apple Silicon.
    Optimisé pour les Mac M1/M2/M3/M4.
    """
    
    def __init__(self):
        import mlx.core as mx
        from mlx_lm import load
        
        self.mx = mx
        model_name = config.embedding_model_mlx
        print(f"📦 Chargement du modèle MLX natif: {model_name}")
        
        # Charger le modèle MLX
        self.model, self.tokenizer = load(model_name)
        
        # Dimension d'embedding (à ajuster selon le modèle)
        self._embedding_dim = 384  # bge-small-en-v1.5 = 384
        
        print(f"✅ Modèle MLX chargé (NPU activé)")
    
    def _mean_pooling(self, hidden_states, attention_mask):
        """Mean pooling sur les hidden states"""
        mx = self.mx
        # Masquer les tokens de padding
        mask_expanded = mx.expand_dims(attention_mask, -1)
        sum_embeddings = mx.sum(hidden_states * mask_expanded, axis=1)
        sum_mask = mx.clip(mx.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask
    
    def generate_embeddings(self, documents: List[Dict[str, str]]) -> np.ndarray:
        """
        Génère les embeddings pour une liste de documents avec MLX.
        
        Args:
            documents: Liste de dicts avec 'path' et 'content'
        
        Returns:
            Array numpy de shape (n_documents, embedding_dim)
        """
        mx = self.mx
        
        if not documents:
            raise ValueError("Aucun document à embedder")
        
        print(f"\n🔢 Génération des embeddings MLX pour {len(documents)} notes...")
        
        embeddings = []
        
        for doc in tqdm(documents, desc="Embeddings MLX"):
            content = doc['content'][:2000]
            prompt = f"task: clustering | query: {content}"
            
            # Tokeniser
            inputs = self.tokenizer(prompt, return_tensors="np", truncation=True, max_length=512)
            input_ids = mx.array(inputs['input_ids'])
            attention_mask = mx.array(inputs['attention_mask'])
            
            # Générer les embeddings
            outputs = self.model(input_ids)
            
            # Mean pooling
            if hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            else:
                hidden_states = outputs
            
            embedding = self._mean_pooling(hidden_states, attention_mask)
            
            # Normaliser
            embedding = embedding / mx.linalg.norm(embedding, axis=-1, keepdims=True)
            
            embeddings.append(np.array(embedding[0]))
        
        embeddings = np.array(embeddings)
        print(f"✅ Embeddings MLX générés: shape {embeddings.shape}")
        
        return embeddings
    
    def save_embeddings(self, embeddings: np.ndarray, output_path: Path):
        """Sauvegarde les embeddings sur disque"""
        np.save(output_path, embeddings)
        print(f"💾 Embeddings sauvegardés: {output_path}")
    
    def load_embeddings(self, input_path: Path) -> np.ndarray:
        """Charge les embeddings depuis le disque"""
        embeddings = np.load(input_path)
        print(f"📂 Embeddings chargés: {input_path} (shape: {embeddings.shape})")
        return embeddings


class EmbeddingGenerator:
    """
    Génère des embeddings pour les notes Obsidian.
    Utilise sentence-transformers (CUDA/CPU fallback).
    """
    
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        
        print(f"📦 Chargement du modèle d'embeddings: {config.embedding_model}")
        
        # Note: EmbeddingGemma ne supporte pas float16
        # On force float32 ou bfloat16
        self.model = SentenceTransformer(
            config.embedding_model,
            trust_remote_code=True,
            token=config.hf_token,
            device="mps" if config.backend == "mlx" else None  # Metal Performance Shaders pour Mac
        )
        
        print(f"✅ Modèle chargé (dimension: {self.model.get_sentence_embedding_dimension()})")
    
    def generate_embeddings(self, documents: List[Dict[str, str]]) -> np.ndarray:
        """
        Génère les embeddings pour une liste de documents.
        
        Args:
            documents: Liste de dicts avec 'path' et 'content'
        
        Returns:
            Array numpy de shape (n_documents, embedding_dim)
        """
        if not documents:
            raise ValueError("Aucun document à embedder")
        
        print(f"\n🔢 Génération des embeddings pour {len(documents)} notes...")
        
        # Préparer les textes avec le prompt optimisé pour clustering
        texts = []
        for doc in documents:
            # Utiliser le prompt recommandé pour clustering
            # Format: "task: clustering | query: {content}"
            content = doc['content'][:2000]  # Limiter à 2000 chars pour éviter de dépasser le contexte
            prompt = f"task: clustering | query: {content}"
            texts.append(prompt)
        
        # Générer les embeddings avec barre de progression
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True  # Normalisation pour la similarité cosinus
        )
        
        print(f"✅ Embeddings générés: shape {embeddings.shape}")
        
        return embeddings
    
    def save_embeddings(self, embeddings: np.ndarray, output_path: Path):
        """Sauvegarde les embeddings sur disque"""
        np.save(output_path, embeddings)
        print(f"💾 Embeddings sauvegardés: {output_path}")
    
    def load_embeddings(self, input_path: Path) -> np.ndarray:
        """Charge les embeddings depuis le disque"""
        embeddings = np.load(input_path)
        print(f"📂 Embeddings chargés: {input_path} (shape: {embeddings.shape})")
        return embeddings


def get_embedding_generator():
    """
    Retourne le générateur d'embeddings approprié selon le backend configuré.
    
    Returns:
        MLXEmbeddingGenerator pour Apple Silicon avec MLX natif,
        EmbeddingGenerator pour CUDA/CPU fallback
    """
    if config.backend == "mlx" and config.embedding_model_mlx:
        try:
            return MLXEmbeddingGenerator()
        except ImportError as e:
            print(f"⚠️  MLX non disponible, fallback sur sentence-transformers: {e}")
            return EmbeddingGenerator()
    else:
        return EmbeddingGenerator()
