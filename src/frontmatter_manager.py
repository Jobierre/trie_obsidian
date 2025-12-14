"""
Gestion du frontmatter YAML des notes Obsidian
"""
from pathlib import Path
from typing import Optional, List
import frontmatter


class FrontmatterManager:
    """Lit et écrit le frontmatter YAML des notes Obsidian"""
    
    @staticmethod
    def read_note(file_path: Path) -> dict:
        """
        Lit une note Obsidian et extrait le frontmatter + contenu.
        
        Args:
            file_path: Chemin vers la note .md
        
        Returns:
            Dict avec 'metadata' (dict du frontmatter) et 'content' (str du body)
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            post = frontmatter.load(f)
        
        return {
            'metadata': dict(post.metadata),
            'content': post.content
        }
    
    @staticmethod
    def write_note(file_path: Path, content: str, metadata: dict):
        """
        Écrit une note Obsidian avec frontmatter.
        
        Args:
            file_path: Chemin vers la note .md
            content: Contenu de la note (sans frontmatter)
            metadata: Dict du frontmatter à écrire
        """
        post = frontmatter.Post(content, **metadata)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(frontmatter.dumps(post))
    
    @staticmethod
    def update_parent(file_path: Path, parent_category: str, dry_run: bool = True) -> dict:
        """
        Met à jour le champ 'parent' dans le frontmatter d'une note.
        
        Args:
            file_path: Chemin vers la note .md
            parent_category: Nom de la catégorie parent (ex: "Tech - Python")
            dry_run: Si True, ne modifie pas le fichier (mode preview)
        
        Returns:
            Dict avec les infos du changement effectué
        """
        # Lire la note
        note_data = FrontmatterManager.read_note(file_path)
        metadata = note_data['metadata']
        content = note_data['content']
        
        # Récupérer l'ancien parent (si existe)
        old_parent = metadata.get('parent', None)
        
        # Format Obsidian link: [[Category]]
        new_parent = f"[[{parent_category}]]"
        
        # Mettre à jour le metadata
        metadata['parent'] = new_parent
        
        # Préparer le résumé du changement
        change_info = {
            'file': str(file_path),
            'old_parent': old_parent,
            'new_parent': new_parent,
            'action': 'update' if old_parent else 'add'
        }
        
        # Écrire si pas en dry-run
        if not dry_run:
            FrontmatterManager.write_note(file_path, content, metadata)
        
        return change_info
    
    @staticmethod
    def get_existing_categories(vault_path: Path) -> List[str]:
        """
        Scanne le vault et retourne toutes les catégories existantes.
        
        Args:
            vault_path: Chemin vers le vault Obsidian
        
        Returns:
            Liste des noms de catégories uniques trouvées
        """
        categories = set()
        
        for md_file in vault_path.rglob("*.md"):
            try:
                note_data = FrontmatterManager.read_note(md_file)
                parent = note_data['metadata'].get('parent')
                
                if parent:
                    # Enlever les [[ ]]
                    if isinstance(parent, str):
                        category = parent.strip('[]').strip()
                        categories.add(category)
                    elif isinstance(parent, list):
                        for p in parent:
                            category = p.strip('[]').strip()
                            categories.add(category)
            except Exception as e:
                print(f"⚠️  Erreur lecture {md_file.name}: {e}")
                continue
        
        return sorted(list(categories))
    
    @staticmethod
    def scan_vault(vault_path: Path) -> List[dict]:
        """
        Scanne tous les fichiers .md dans le vault.
        
        Args:
            vault_path: Chemin vers le vault Obsidian
        
        Returns:
            Liste de dicts avec 'path', 'metadata' et 'content'
        """
        documents = []
        
        print(f"📂 Scan du vault: {vault_path}")
        
        md_files = list(vault_path.rglob("*.md"))
        
        if not md_files:
            raise ValueError(f"❌ Aucun fichier .md trouvé dans {vault_path}")
        
        print(f"📄 {len(md_files)} notes trouvées")
        
        for md_file in md_files:
            try:
                note_data = FrontmatterManager.read_note(md_file)
                documents.append({
                    'path': md_file,
                    'metadata': note_data['metadata'],
                    'content': note_data['content']
                })
            except Exception as e:
                print(f"⚠️  Erreur lecture {md_file.name}: {e}")
                continue
        
        print(f"✅ {len(documents)} notes chargées avec succès\n")
        
        return documents
