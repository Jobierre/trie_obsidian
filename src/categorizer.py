"""
Classification des notes avec le LLM (mode individuel avec parallélisation)
"""
from typing import List, Dict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

from .config import config, get_max_workers
from .llm import LLMGenerator
from .cache import ClassificationCache


class NoteClassifier:
    """Classe chaque note individuellement dans des catégories prédéfinies (parallélisé si possible)"""
    
    def __init__(self, llm: LLMGenerator, use_cache: bool = True, num_workers: int = 0):
        self.llm = llm
        self.cache = ClassificationCache() if use_cache else None
        
        # MLX n'est PAS thread-safe - forcer 1 worker pour éviter les segfaults
        if config.backend == "mlx":
            self.num_workers = 1
            self._parallel_enabled = False
        else:
            # Limiter les workers LLM à la moitié pour la mémoire (CUDA/CPU)
            self.num_workers = max(1, get_max_workers(num_workers) // 2)
            self._parallel_enabled = self.num_workers > 1
        
        self._lock = threading.Lock()
    
    def _classify_single_note(
        self,
        doc: Dict,
        categories: List[Dict[str, str]]
    ) -> tuple:
        """
        Classifie une note unique (thread-safe).
        
        Returns:
            Tuple (file_path, category, from_cache)
        """
        file_path = doc['path']
        content = doc['content']
        
        # Extraire le titre
        title = file_path.stem if hasattr(file_path, 'stem') else str(file_path).split('/')[-1].replace('.md', '')
        
        # Vérifier le cache
        cached_category = None
        if self.cache:
            cached_category = self.cache.get_cached_category(file_path, content)
        
        if cached_category:
            return (str(file_path), cached_category, True)
        
        # Classifier avec le LLM
        category = self.llm.choose_category(title, content, categories)
        
        # Mettre en cache (thread-safe via FileLock dans cache.py)
        if self.cache:
            self.cache.cache_category(file_path, content, category)
        
        return (str(file_path), category, False)
    
    def classify_notes(
        self,
        documents: List[Dict[str, str]],
        categories: List[Dict[str, str]]
    ) -> Dict[str, str]:
        """
        Classe chaque note individuellement (parallélisé).
        
        Args:
            documents: Liste de dicts avec 'path', 'metadata', 'content'
            categories: Liste de catégories disponibles avec 'name' et 'description'
        
        Returns:
            Dict mapping str(file_path) -> category_name
        """
        print(f"\n🏷️  Classification de {len(documents)} notes...")
        print(f"📁 {len(categories)} catégories disponibles")
        if self._parallel_enabled:
            print(f"👷 {self.num_workers} workers LLM")
        else:
            print(f"⚡ Mode séquentiel (MLX non thread-safe)")
        
        if self.cache:
            cache_stats = self.cache.get_stats()
            print(f"💾 Cache: {cache_stats['total_entries']} entrées")
        print()
        
        note_to_category = {}
        cached_count = 0
        classified_count = 0
        
        # Mode séquentiel pour MLX (pas thread-safe)
        if not self._parallel_enabled:
            for doc in tqdm(documents, desc="Classification"):
                file_path, category, from_cache = self._classify_single_note(doc, categories)
                note_to_category[file_path] = category
                if from_cache:
                    cached_count += 1
                else:
                    classified_count += 1
        else:
            # Parallélisation des appels LLM (CUDA/CPU uniquement)
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # Soumettre toutes les tâches
                futures = {
                    executor.submit(self._classify_single_note, doc, categories): doc
                    for doc in documents
                }
                
                # Récupérer les résultats avec barre de progression
                for future in tqdm(as_completed(futures), total=len(documents), desc="Classification"):
                    file_path, category, from_cache = future.result()
                    note_to_category[file_path] = category
                    
                    if from_cache:
                        cached_count += 1
                    else:
                        classified_count += 1
        
        print(f"\n✅ Classification terminée:")
        print(f"   • {classified_count} notes classifiées par le LLM")
        if self.cache:
            print(f"   • {cached_count} notes récupérées du cache")
        print()
        
        return note_to_category

