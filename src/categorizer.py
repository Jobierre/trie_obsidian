"""
Classification des notes avec le LLM (mode individuel)
"""
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm

from .config import config
from .llm import LLMGenerator
from .cache import ClassificationCache


class NoteClassifier:
    """Classe chaque note individuellement dans des catégories prédéfinies"""
    
    def __init__(self, llm: LLMGenerator, use_cache: bool = True):
        self.llm = llm
        self.cache = ClassificationCache() if use_cache else None
    
    def classify_notes(
        self,
        documents: List[Dict[str, str]],
        categories: List[Dict[str, str]]
    ) -> Dict[str, str]:
        """
        Classe chaque note individuellement.
        
        Args:
            documents: Liste de dicts avec 'path', 'metadata', 'content'
            categories: Liste de catégories disponibles avec 'name' et 'description'
        
        Returns:
            Dict mapping str(file_path) -> category_name
        """
        print(f"\n🏷️  Classification de {len(documents)} notes...")
        print(f"📁 {len(categories)} catégories disponibles")
        
        if self.cache:
            cache_stats = self.cache.get_stats()
            print(f"💾 Cache: {cache_stats['total_entries']} entrées")
        print()
        
        note_to_category = {}
        cached_count = 0
        classified_count = 0
        
        for doc in tqdm(documents, desc="Classification"):
            file_path = doc['path']
            content = doc['content']
            
            # Extraire le titre
            title = file_path.stem if hasattr(file_path, 'stem') else str(file_path).split('/')[-1].replace('.md', '')
            
            # Vérifier le cache
            cached_category = None
            if self.cache:
                cached_category = self.cache.get_cached_category(file_path, content)
            
            if cached_category:
                category = cached_category
                cached_count += 1
            else:
                # Classifier avec le LLM
                category = self.llm.choose_category(title, content, categories)
                classified_count += 1
                
                # Mettre en cache
                if self.cache:
                    self.cache.cache_category(file_path, content, category)
            
            note_to_category[str(file_path)] = category
        
        print(f"\n✅ Classification terminée:")
        print(f"   • {classified_count} notes classifiées par le LLM")
        if self.cache:
            print(f"   • {cached_count} notes récupérées du cache")
        print()
        
        return note_to_category

