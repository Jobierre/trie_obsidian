"""
Système de cache pour éviter de reclassifier les notes déjà traitées
"""
import json
import hashlib
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

from .config import config


class ClassificationCache:
    """Cache les classifications de notes pour éviter les reclassifications"""
    
    def __init__(self):
        self.cache_file = config.output_dir / "classification_cache.json"
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """Charge le cache depuis le disque"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Erreur lecture cache: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Sauvegarde le cache sur le disque"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  Erreur sauvegarde cache: {e}")
    
    def _get_note_hash(self, note_path: Path, content: str) -> str:
        """Génère un hash unique pour une note (basé sur le chemin + contenu)"""
        note_id = f"{note_path}|{content[:500]}"
        return hashlib.md5(note_id.encode()).hexdigest()
    
    def get_cached_category(self, note_path: Path, content: str) -> Optional[str]:
        """
        Récupère la catégorie depuis le cache si elle existe et est à jour.
        
        Args:
            note_path: Chemin de la note
            content: Contenu de la note
        
        Returns:
            Nom de la catégorie ou None si pas en cache
        """
        note_hash = self._get_note_hash(note_path, content)
        
        if note_hash in self.cache:
            cached_entry = self.cache[note_hash]
            return cached_entry.get('category')
        
        return None
    
    def cache_category(self, note_path: Path, content: str, category: str):
        """
        Met en cache la classification d'une note.
        
        Args:
            note_path: Chemin de la note
            content: Contenu de la note
            category: Catégorie assignée
        """
        note_hash = self._get_note_hash(note_path, content)
        
        self.cache[note_hash] = {
            'path': str(note_path),
            'category': category,
            'timestamp': datetime.now().isoformat()
        }
        
        self._save_cache()
    
    def clear_cache(self):
        """Efface complètement le cache"""
        self.cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        print("🗑️  Cache effacé")
    
    def get_stats(self) -> Dict:
        """Retourne des statistiques sur le cache"""
        return {
            'total_entries': len(self.cache),
            'cache_file': str(self.cache_file),
            'exists': self.cache_file.exists()
        }
