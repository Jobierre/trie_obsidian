"""
Génération d'embeddings avec EmbeddingGemma-300M
"""
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pathlib import Path

from .config import config


class EmbeddingGenerator:
    """Génère des embeddings pour les notes Obsidian"""
    
    def __init__(self):
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
