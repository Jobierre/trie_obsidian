"""
Gestion du fichier categories.yaml
"""
from pathlib import Path
from typing import List, Dict, Optional
import yaml

from .config import config


class CategoryManager:
    """Gère les catégories prédéfinies pour la classification"""
    
    def __init__(self):
        self.categories_file = config.project_root / "categories.yaml"
        self.categories = []
    
    def load_categories(self) -> List[Dict[str, str]]:
        """
        Charge les catégories depuis categories.yaml.
        
        Returns:
            Liste de dicts avec 'name' et 'description'
        """
        if not self.categories_file.exists():
            raise FileNotFoundError(
                f"❌ Fichier {self.categories_file} introuvable.\n"
                f"Exécutez d'abord: python -m src.main --discover"
            )
        
        with open(self.categories_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        self.categories = data.get('categories', [])
        
        if not self.categories:
            raise ValueError("❌ Aucune catégorie trouvée dans categories.yaml")
        
        print(f"✅ {len(self.categories)} catégories chargées depuis {self.categories_file.name}")
        for cat in self.categories:
            print(f"   • {cat['name']}")
        print()
        
        return self.categories
    
    def save_categories(self, categories: List[Dict[str, str]]):
        """
        Sauvegarde les catégories dans categories.yaml.
        
        Args:
            categories: Liste de dicts avec 'name' et 'description'
        """
        # Toujours ajouter la catégorie Divers si pas présente
        if not any(cat['name'] == 'Divers' for cat in categories):
            categories.append({
                'name': 'Divers',
                'description': 'Notes qui ne correspondent à aucune catégorie spécifique, ou en attente de classification'
            })
        
        data = {
            'categories': categories
        }
        
        # Créer le fichier avec commentaires
        yaml_content = f"""# Catégories pour organisation Obsidian
# Générées automatiquement - modifiez selon vos besoins
# 
# Instructions:
#   - Ajoutez/supprimez des catégories librement
#   - Format: "Domaine - Sous-thème" recommandé
#   - Des descriptions claires aident le LLM à mieux classifier
#   - Relancez 'python -m src.main' après modification

"""
        
        with open(self.categories_file, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        
        print(f"💾 Catégories sauvegardées: {self.categories_file}")
        print(f"   → {len(categories)} catégories")
        print(f"\n📝 Éditez le fichier pour ajuster les catégories, puis relancez le programme.\n")
    
    def get_category_names(self) -> List[str]:
        """Retourne la liste des noms de catégories"""
        return [cat['name'] for cat in self.categories]
    
    def validate_category(self, category_name: str) -> bool:
        """Vérifie si une catégorie existe"""
        return category_name in self.get_category_names()
    
    def create_default_categories(self) -> List[Dict[str, str]]:
        """Crée un set de catégories par défaut si aucune découverte n'est faite"""
        return [
            {
                'name': 'Tech - Infrastructure',
                'description': 'Serveurs, conteneurs (Docker, LXC), configuration système, réseau, stockage'
            },
            {
                'name': 'Dev - Programmation',
                'description': 'Code, langages (Python, JavaScript, etc.), frameworks, bibliothèques, tutoriels'
            },
            {
                'name': 'IA - Outils',
                'description': 'Intelligence artificielle, LLM, prompts, Claude, GPT, automatisation IA'
            },
            {
                'name': 'Finance',
                'description': 'Investissement, trading, comptabilité, retraite, économie'
            },
            {
                'name': 'Maison',
                'description': 'Bricolage, rénovation, achats, aménagement, travaux'
            },
            {
                'name': 'Perso',
                'description': 'Notes personnelles, rendez-vous, messages, idées diverses'
            },
            {
                'name': 'Divers',
                'description': 'Tout ce qui ne rentre pas dans les autres catégories'
            }
        ]
