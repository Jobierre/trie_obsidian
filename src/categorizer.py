"""
Catégorisation des clusters avec le LLM
"""
from typing import List, Dict
import numpy as np
from tqdm import tqdm

from .config import config
from .llm import LLMGenerator


class ClusterCategorizer:
    """Génère des noms de catégories pour les clusters avec le LLM"""
    
    def __init__(self, llm: LLMGenerator):
        self.llm = llm
    
    def categorize_clusters(
        self, 
        cluster_samples: Dict[int, List[Dict[str, str]]]
    ) -> Dict[int, str]:
        """
        Génère un nom de catégorie pour chaque cluster.
        
        Args:
            cluster_samples: Dict mapping cluster_id -> liste d'échantillons de notes
        
        Returns:
            Dict mapping cluster_id -> nom de catégorie
        """
        print(f"\n🏷️  Génération des noms de catégories pour {len(cluster_samples)} clusters...")
        
        categories = {}
        
        for cluster_id, samples in tqdm(cluster_samples.items(), desc="Catégorisation"):
            # Extraire juste le contenu des notes
            sample_contents = [doc['content'] for doc in samples]
            
            # Demander au LLM de générer un nom de catégorie
            category_name = self.llm.categorize_cluster(sample_contents)
            
            categories[cluster_id] = category_name
            
            # Afficher pour feedback
            tqdm.write(f"   Cluster {cluster_id:2d} → {category_name}")
        
        print(f"✅ Catégorisation terminée\n")
        
        return categories
    
    def handle_unclustered_notes(
        self, 
        unclustered_notes: List[Dict[str, str]]
    ) -> str:
        """
        Décide quoi faire avec les notes non classées.
        
        Pour l'instant, on les met toutes dans une catégorie "Divers".
        
        Args:
            unclustered_notes: Liste des notes non classées
        
        Returns:
            Nom de catégorie pour les notes non classées
        """
        if not unclustered_notes:
            return None
        
        print(f"\n📦 {len(unclustered_notes)} notes non classées")
        
        # Si trop peu de notes, les mettre dans "Divers"
        if len(unclustered_notes) < config.min_cluster_size:
            print(f"   → Catégorie: Divers (moins de {config.min_cluster_size} notes)")
            return "Divers"
        
        # Si suffisamment de notes, on peut essayer de les sous-catégoriser
        # Pour l'instant, on les met aussi dans "Divers"
        # TODO: Implémenter un second niveau de clustering si besoin
        print(f"   → Catégorie: Divers")
        return "Divers"
    
    def map_notes_to_categories(
        self,
        documents: List[Dict[str, str]],
        labels: np.ndarray,
        categories: Dict[int, str]
    ) -> Dict[str, str]:
        """
        Crée un mapping note_path -> category_name.
        
        Args:
            documents: Liste des documents
            labels: Labels de cluster pour chaque document
            categories: Mapping cluster_id -> category_name
        
        Returns:
            Dict mapping str(file_path) -> category_name
        """
        note_to_category = {}
        
        for i, doc in enumerate(documents):
            cluster_id = int(labels[i])
            
            if cluster_id == -1:
                # Note non classée
                category = "Divers"
            else:
                category = categories.get(cluster_id, "Divers")
            
            note_to_category[str(doc['path'])] = category
        
        return note_to_category
