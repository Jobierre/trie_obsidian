"""
Gestion du mode dry-run et génération de rapports
"""
import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime

from .config import config


class DryRunManager:
    """Gère le mode preview et génère des rapports de changements"""
    
    def __init__(self):
        self.changes = []
    
    def add_change(self, change_info: dict):
        """
        Ajoute un changement au rapport.
        
        Args:
            change_info: Dict avec les infos du changement (de FrontmatterManager.update_parent)
        """
        self.changes.append(change_info)
    
    def generate_report(self, output_format: str = 'markdown') -> str:
        """
        Génère un rapport des changements prévus.
        
        Args:
            output_format: 'markdown' ou 'json'
        
        Returns:
            Chemin du fichier de rapport généré
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_format == 'markdown':
            return self._generate_markdown_report(timestamp)
        elif output_format == 'json':
            return self._generate_json_report(timestamp)
        else:
            raise ValueError(f"Format non supporté: {output_format}")
    
    def _generate_markdown_report(self, timestamp: str) -> str:
        """Génère un rapport Markdown"""
        output_path = config.output_dir / f"dry_run_report_{timestamp}.md"
        
        # Statistiques
        n_total = len(self.changes)
        n_add = sum(1 for c in self.changes if c['action'] == 'add')
        n_update = sum(1 for c in self.changes if c['action'] == 'update')
        
        # Grouper par catégorie
        by_category = {}
        for change in self.changes:
            category = change['new_parent'].strip('[]').strip()
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(change)
        
        # Générer le rapport
        lines = [
            "# 📋 Rapport Dry-Run - Organisation Obsidian",
            "",
            f"**Date**: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
            f"**Mode**: DRY-RUN (aucun fichier modifié)",
            "",
            "---",
            "",
            "## 📊 Statistiques",
            "",
            f"- **Total de notes**: {n_total}",
            f"- **Nouvelles catégories**: {n_add}",
            f"- **Catégories mises à jour**: {n_update}",
            f"- **Nombre de catégories uniques**: {len(by_category)}",
            "",
            "---",
            "",
            "## 📁 Changements par catégorie",
            ""
        ]
        
        for category in sorted(by_category.keys()):
            changes_in_cat = by_category[category]
            lines.append(f"### {category} ({len(changes_in_cat)} notes)")
            lines.append("")
            
            for change in changes_in_cat[:10]:  # Limiter à 10 pour lisibilité
                file_name = Path(change['file']).name
                
                if change['action'] == 'add':
                    lines.append(f"- ✨ **{file_name}** - Nouvelle catégorie")
                else:
                    old = change['old_parent'].strip('[]').strip() if change['old_parent'] else 'Aucune'
                    lines.append(f"- 🔄 **{file_name}** - `{old}` → `{category}`")
            
            if len(changes_in_cat) > 10:
                lines.append(f"- ... et {len(changes_in_cat) - 10} autres notes")
            
            lines.append("")
        
        lines.extend([
            "---",
            "",
            "## ⚙️ Prochaines étapes",
            "",
            "Pour appliquer ces changements:",
            "```bash",
            "python -m src.main --apply",
            "```",
            "",
            "⚠️ **Attention**: Faites une sauvegarde de votre vault avant d'appliquer les changements !",
            ""
        ])
        
        # Écrire le fichier
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        return str(output_path)
    
    def _generate_json_report(self, timestamp: str) -> str:
        """Génère un rapport JSON"""
        output_path = config.output_dir / f"dry_run_report_{timestamp}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'mode': 'dry_run',
            'statistics': {
                'total_notes': len(self.changes),
                'new_categories': sum(1 for c in self.changes if c['action'] == 'add'),
                'updated_categories': sum(1 for c in self.changes if c['action'] == 'update')
            },
            'changes': self.changes
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return str(output_path)
    
    def print_summary(self):
        """Affiche un résumé concis des changements"""
        n_total = len(self.changes)
        n_add = sum(1 for c in self.changes if c['action'] == 'add')
        n_update = sum(1 for c in self.changes if c['action'] == 'update')
        
        # Catégories uniques
        categories = set(c['new_parent'].strip('[]').strip() for c in self.changes)
        
        print("\n" + "=" * 60)
        print("📋 RÉSUMÉ DES CHANGEMENTS (DRY-RUN)")
        print("=" * 60)
        print(f"✅ {n_total} notes seront modifiées:")
        print(f"   • {n_add} recevront une nouvelle catégorie")
        print(f"   • {n_update} verront leur catégorie mise à jour")
        print(f"\n📁 {len(categories)} catégories uniques:")
        for cat in sorted(categories):
            count = sum(1 for c in self.changes if c['new_parent'].strip('[]').strip() == cat)
            print(f"   • {cat}: {count} notes")
        print("=" * 60)
        print()
