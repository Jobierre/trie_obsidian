"""
Système de cache thread-safe pour éviter de reclassifier les notes déjà traitées
"""
import json
import hashlib
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

try:
    from filelock import FileLock
    HAS_FILELOCK = True
except ImportError:
    HAS_FILELOCK = False

from .config import config


class ClassificationCache:
    """Cache les classifications de notes pour éviter les reclassifications (thread-safe)"""
    
    def __init__(self):
        self.cache_file = config.output_dir / "classification_cache.json"
        self.lock_file = config.output_dir / "classification_cache.json.lock"
        self._lock = FileLock(str(self.lock_file)) if HAS_FILELOCK else None
        self.cache = self._load_cache()
    
    def _acquire_lock(self):
        """Acquiert le verrou pour les accès concurrents"""
        if self._lock:
            self._lock.acquire()
    
    def _release_lock(self):
        """Libère le verrou"""
        if self._lock:
            self._lock.release()
    
    def _load_cache(self) -> Dict:
        """Charge le cache depuis le disque (thread-safe)"""
        self._acquire_lock()
        try:
            if self.cache_file.exists():
                try:
                    with open(self.cache_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"⚠️  Erreur lecture cache: {e}")
                    return {}
            return {}
        finally:
            self._release_lock()
    
    def _save_cache(self):
        """Sauvegarde le cache sur le disque (thread-safe)"""
        self._acquire_lock()
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  Erreur sauvegarde cache: {e}")
        finally:
            self._release_lock()
    
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
