"""
Script principal d'orchestration
"""
import argparse
import random
from pathlib import Path

from .config import config
from .frontmatter_manager import FrontmatterManager
from .llm import LLMGenerator
from .categorizer import NoteClassifier
from .category_manager import CategoryManager
from .dry_run import DryRunManager


def discover_mode(llm: LLMGenerator, documents: list, count: int, sample_size: int = 50):
    """Mode découverte: analyse les notes et propose des catégories"""
    print("\n" + "=" * 60)
    print("🔍 MODE DÉCOUVERTE DES CATÉGORIES")
    print("=" * 60)
    print(f"Analyse de {len(documents)} notes pour proposer {count} catégories...\n")
    
    # Prendre un échantillon représentatif
    sample_size = min(sample_size, len(documents))
    sample = random.sample(documents, sample_size)
    
    # Préparer les données pour le LLM
    sample_data = []
    for doc in sample:
        title = doc['path'].stem if hasattr(doc['path'], 'stem') else str(doc['path']).split('/')[-1].replace('.md', '')
        sample_data.append({
            'title': title,
            'content': doc['content']
        })
    
    print(f"📊 Échantillon: {sample_size} notes analysées")
    print("🤖 Génération des catégories...")
    
    # Demander au LLM de proposer des catégories
    categories = llm.discover_categories(sample_data, count=count)
    
    print(f"\n✅ {len(categories)} catégories proposées:\n")
    for i, cat in enumerate(categories, 1):
        print(f"{i}. {cat['name']}")
        print(f"   {cat['description']}\n")
    
    # Sauvegarder dans categories.yaml
    cat_manager = CategoryManager()
    cat_manager.save_categories(categories)
    
    print("=" * 60)
    print("✨ DÉCOUVERTE TERMINÉE")
    print("=" * 60)
    print(f"Fichier généré: categories.yaml")
    print("\n💡 Prochaines étapes:")
    print("   1. Éditez categories.yaml pour ajuster les catégories")
    print("   2. Relancez: python -m src.main")
    print("   3. Vérifiez le rapport de classification")
    print("   4. Appliquez: python -m src.main --apply\n")


def classify_mode(args):
    """Mode classification: classe les notes dans des catégories prédéfinies"""
    print("\n" + "=" * 60)
    print("🗂️  OBSIDIAN ORGANIZER - CLASSIFICATION")
    print("=" * 60)
    
    if args.apply:
        print("⚠️  MODE: APPLICATION DES CHANGEMENTS")
        response = input("Êtes-vous sûr ? Avez-vous fait une sauvegarde ? (oui/non): ")
        if response.lower() not in ['oui', 'yes', 'y']:
            print("❌ Opération annulée")
            return
    else:
        print("ℹ️  MODE: DRY-RUN (preview uniquement)")
    
    print("\n")
    
    # Étape 1: Scanner le vault
    print("=" * 60)
    print("ÉTAPE 1/4: Scan du vault")
    print("=" * 60)
    
    documents = FrontmatterManager.scan_vault(config.vault_path)
    
    if len(documents) == 0:
        print("❌ Aucune note trouvée. Copiez vos notes .md dans le dossier:")
        print(f"   {config.vault_path}")
        return
    
    # Étape 2: Charger les catégories
    print("=" * 60)
    print("ÉTAPE 2/4: Chargement des catégories")
    print("=" * 60)
    
    cat_manager = CategoryManager()
    try:
        categories = cat_manager.load_categories()
    except FileNotFoundError as e:
        print(str(e))
        print("\n💡 Lancez d'abord la découverte:")
        print("   python -m src.main --discover")
        return
    
    # Étape 3: Charger le LLM et classifier
    print("=" * 60)
    print("ÉTAPE 3/4: Classification des notes")
    print("=" * 60)
    
    llm = LLMGenerator()
    classifier = NoteClassifier(llm, use_cache=not args.clear_cache)
    
    note_to_category = classifier.classify_notes(documents, categories)
    
    # Étape 4: Appliquer les changements (ou dry-run)
    print("=" * 60)
    print("ÉTAPE 4/4: Application des changements")
    print("=" * 60)
    
    dry_run_manager = DryRunManager()
    
    for doc in documents:
        file_path = doc['path']
        category = note_to_category[str(file_path)]
        
        # Mettre à jour le frontmatter
        change_info = FrontmatterManager.update_parent(
            file_path,
            category,
            dry_run=not args.apply
        )
        
        dry_run_manager.add_change(change_info)
    
    # Afficher le résumé
    dry_run_manager.print_summary()
    
    # Générer le rapport
    if not args.apply:
        print("📄 Génération du rapport...")
        
        if args.report_format in ['markdown', 'both']:
            md_path = dry_run_manager.generate_report('markdown')
            print(f"   ✅ Rapport Markdown: {md_path}")
        
        if args.report_format in ['json', 'both']:
            json_path = dry_run_manager.generate_report('json')
            print(f"   ✅ Rapport JSON: {json_path}")
        
        print("\n💡 Pour appliquer les changements, relancez avec --apply")
    else:
        print("✅ Changements appliqués avec succès !")
        print(f"📊 {len(documents)} notes ont été organisées")
    
    print("\n" + "=" * 60)
    print("✨ TERMINÉ")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="🗂️  Obsidian Organizer - Classement automatique de notes avec IA"
    )
    
    # Mode découverte
    parser.add_argument(
        '--discover',
        action='store_true',
        help='Mode découverte: analyse les notes et propose des catégories'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=8,
        help='Nombre de catégories à découvrir (défaut: 8)'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=50,
        help='Nombre de notes à analyser pour la découverte (défaut: 50)'
    )
    
    # Mode classification
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Applique les changements (par défaut: dry-run)'
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Efface le cache et reclassifie toutes les notes'
    )
    parser.add_argument(
        '--report-format',
        choices=['markdown', 'json', 'both'],
        default='markdown',
        help='Format du rapport de dry-run'
    )
    
    args = parser.parse_args()
    
    # Scanner le vault dans tous les cas
    documents = FrontmatterManager.scan_vault(config.vault_path)
    
    if len(documents) == 0:
        print("❌ Aucune note trouvée. Copiez vos notes .md dans:")
        print(f"   {config.vault_path}")
        return
    
    # Mode découverte
    if args.discover:
        llm = LLMGenerator()
        discover_mode(llm, documents, args.count, args.sample)
    else:
        # Mode classification normale
        classify_mode(args)


if __name__ == "__main__":
    main()

