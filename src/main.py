"""
Script principal d'orchestration
"""
import argparse
from pathlib import Path

from .config import config
from .frontmatter_manager import FrontmatterManager
from .embeddings import EmbeddingGenerator
from .clustering import NoteClustering
from .llm import LLMGenerator
from .categorizer import ClusterCategorizer
from .dry_run import DryRunManager


def main():
    parser = argparse.ArgumentParser(
        description="🗂️  Obsidian Organizer - Classement automatique de notes avec IA"
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Applique les changements (par défaut: dry-run)'
    )
    parser.add_argument(
        '--skip-embeddings',
        action='store_true',
        help='Réutilise les embeddings existants (dans output/embeddings.npy)'
    )
    parser.add_argument(
        '--report-format',
        choices=['markdown', 'json', 'both'],
        default='markdown',
        help='Format du rapport de dry-run'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("🗂️  OBSIDIAN ORGANIZER")
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
    print("ÉTAPE 1/6: Scan du vault")
    print("=" * 60)
    
    documents = FrontmatterManager.scan_vault(config.vault_path)
    
    if len(documents) == 0:
        print("❌ Aucune note trouvée. Copiez vos notes .md dans le dossier:")
        print(f"   {config.vault_path}")
        return
    
    # Étape 2: Générer les embeddings
    print("=" * 60)
    print("ÉTAPE 2/6: Génération des embeddings")
    print("=" * 60)
    
    embeddings_path = config.output_dir / "embeddings.npy"
    
    if args.skip_embeddings and embeddings_path.exists():
        print(f"⏩ Réutilisation des embeddings existants: {embeddings_path}")
        embedding_gen = EmbeddingGenerator()
        embeddings = embedding_gen.load_embeddings(embeddings_path)
    else:
        embedding_gen = EmbeddingGenerator()
        embeddings = embedding_gen.generate_embeddings(documents)
        embedding_gen.save_embeddings(embeddings, embeddings_path)
    
    # Étape 3: Clustering
    print("\n" + "=" * 60)
    print("ÉTAPE 3/6: Clustering des notes")
    print("=" * 60)
    
    clusterer = NoteClustering()
    labels, stats = clusterer.cluster_notes(embeddings)
    
    # Extraire des échantillons
    cluster_samples = clusterer.get_cluster_samples(
        documents, 
        labels, 
        n_samples=config.llm_samples_per_cluster
    )
    
    unclustered_notes = clusterer.get_unclustered_notes(documents, labels)
    
    # Étape 4: Charger le LLM
    print("\n" + "=" * 60)
    print("ÉTAPE 4/6: Chargement du LLM")
    print("=" * 60)
    
    llm = LLMGenerator()
    
    # Étape 5: Catégorisation
    print("\n" + "=" * 60)
    print("ÉTAPE 5/6: Génération des catégories")
    print("=" * 60)
    
    categorizer = ClusterCategorizer(llm)
    categories = categorizer.categorize_clusters(cluster_samples)
    
    # Mapper toutes les notes vers leur catégorie
    note_to_category = categorizer.map_notes_to_categories(
        documents, 
        labels, 
        categories
    )
    
    # Étape 6: Appliquer les changements (ou dry-run)
    print("\n" + "=" * 60)
    print("ÉTAPE 6/6: Application des changements")
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


if __name__ == "__main__":
    main()
