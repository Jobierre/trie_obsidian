"""
Clustering des notes avec UMAP + HDBSCAN
"""
from typing import List, Dict, Tuple
import numpy as np
from sklearn.preprocessing import normalize
import umap
import hdbscan

from .config import config


class NoteClustering:
    """Regroupe les notes similaires par clustering"""
    
    def __init__(self):
        self.umap_reducer = None
        self.clusterer = None
    
    def cluster_notes(self, embeddings: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Regroupe les notes par similarité.
        
        Args:
            embeddings: Array numpy de shape (n_notes, embedding_dim)
        
        Returns:
            Tuple (labels, stats) où:
            - labels: array de cluster IDs pour chaque note (-1 = noise/non classé)
            - stats: dict avec statistiques du clustering
        """
        print(f"\n🔬 Clustering de {len(embeddings)} notes...")
        
        # Étape 1: Réduction dimensionnelle avec UMAP
        print(f"📉 Réduction dimensionnelle (UMAP): {embeddings.shape[1]} → {config.umap_n_components} dimensions")
        
        self.umap_reducer = umap.UMAP(
            n_components=config.umap_n_components,
            n_neighbors=config.umap_n_neighbors,
            min_dist=config.umap_min_dist,
            metric='cosine',
            random_state=42
        )
        
        embeddings_reduced = self.umap_reducer.fit_transform(embeddings)
        print(f"✅ Réduction terminée: shape {embeddings_reduced.shape}")
        
        # Étape 2: Clustering hiérarchique avec HDBSCAN
        print(f"🎯 Clustering hiérarchique (HDBSCAN)...")
        
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=config.min_cluster_size,
            min_samples=config.hdbscan_min_samples,
            metric='euclidean',
            cluster_selection_method='eom',  # Excess of Mass
            prediction_data=True
        )
        
        labels = self.clusterer.fit_predict(embeddings_reduced)
        
        # Statistiques
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        stats = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_percentage': (n_noise / len(labels)) * 100,
            'cluster_sizes': {}
        }
        
        # Taille de chaque cluster
        for cluster_id in range(n_clusters):
            cluster_size = list(labels).count(cluster_id)
            stats['cluster_sizes'][cluster_id] = cluster_size
        
        # Afficher les résultats
        print(f"\n📊 Résultats du clustering:")
        print(f"   • {n_clusters} clusters trouvés")
        print(f"   • {n_noise} notes non classées ({stats['noise_percentage']:.1f}%)")
        
        if n_clusters > 0:
            print(f"\n📈 Distribution des clusters:")
            for cluster_id in sorted(stats['cluster_sizes'].keys()):
                size = stats['cluster_sizes'][cluster_id]
                bar = "█" * (size // 5)  # Barre visuelle
                print(f"   Cluster {cluster_id:2d}: {size:3d} notes {bar}")
        
        return labels, stats
    
    def get_cluster_samples(
        self, 
        documents: List[Dict[str, str]], 
        labels: np.ndarray, 
        n_samples: int = 5
    ) -> Dict[int, List[Dict[str, str]]]:
        """
        Extrait des échantillons représentatifs de chaque cluster.
        
        Args:
            documents: Liste des documents originaux
            labels: Labels de cluster pour chaque document
            n_samples: Nombre d'échantillons à extraire par cluster
        
        Returns:
            Dict mapping cluster_id -> liste d'échantillons de notes
        """
        cluster_samples = {}
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        for cluster_id in range(n_clusters):
            # Indices des notes dans ce cluster
            indices = np.where(labels == cluster_id)[0]
            
            # Prendre n_samples aléatoires (ou tous si moins de n_samples)
            if len(indices) <= n_samples:
                sample_indices = indices
            else:
                sample_indices = np.random.choice(indices, n_samples, replace=False)
            
            # Extraire les documents
            samples = [documents[i] for i in sample_indices]
            cluster_samples[cluster_id] = samples
        
        return cluster_samples
    
    def get_unclustered_notes(
        self, 
        documents: List[Dict[str, str]], 
        labels: np.ndarray
    ) -> List[Dict[str, str]]:
        """
        Retourne les notes non classées (label = -1).
        
        Args:
            documents: Liste des documents originaux
            labels: Labels de cluster pour chaque document
        
        Returns:
            Liste des notes non classées
        """
        unclustered_indices = np.where(labels == -1)[0]
        return [documents[i] for i in unclustered_indices]
