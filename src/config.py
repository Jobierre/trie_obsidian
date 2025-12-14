"""
Configuration et détection automatique du backend (MLX ou CUDA)
"""
import os
import platform
from pathlib import Path
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()


class Config:
    """Configuration globale de l'application"""
    
    def __init__(self):
        # Détection du backend
        self.backend = self._detect_backend()
        
        # Hugging Face
        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token or self.hf_token == "your_huggingface_token_here":
            raise ValueError(
                "❌ HF_TOKEN manquant dans le fichier .env\n"
                "Créez un token sur https://huggingface.co/settings/tokens\n"
                "Puis copiez .env.example vers .env et ajoutez votre token"
            )
        
        # Chemins
        self.project_root = Path(__file__).parent.parent
        self.vault_path = Path(os.getenv("VAULT_PATH", "./notes"))
        if not self.vault_path.is_absolute():
            self.vault_path = self.project_root / self.vault_path
        self.output_dir = self.project_root / "output"
        self.output_dir.mkdir(exist_ok=True)
        
        # Paramètres de clustering
        self.min_cluster_size = int(os.getenv("MIN_CLUSTER_SIZE", "5"))
        self.hdbscan_min_samples = int(os.getenv("HDBSCAN_MIN_SAMPLES", "3"))
        self.umap_n_components = int(os.getenv("UMAP_N_COMPONENTS", "15"))
        self.umap_n_neighbors = int(os.getenv("UMAP_N_NEIGHBORS", "15"))
        self.umap_min_dist = float(os.getenv("UMAP_MIN_DIST", "0.1"))
        
        # Paramètres LLM
        self.llm_samples_per_cluster = int(os.getenv("LLM_SAMPLES_PER_CLUSTER", "5"))
        
        # Modèles
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "google/embeddinggemma-300m")
        self.llm_model = os.getenv("LLM_MODEL", "mistralai/Ministral-3b-instruct-2412")
        
        # Afficher la configuration
        self._print_config()
    
    def _detect_backend(self) -> str:
        """
        Détecte automatiquement le backend à utiliser.
        
        Returns:
            "mlx" pour Apple Silicon, "cuda" pour NVIDIA GPU, "cpu" sinon
        """
        system = platform.system()
        processor = platform.processor()
        
        # Détection Apple Silicon (M1/M2/M3/M4)
        if system == "Darwin" and "arm" in processor.lower():
            try:
                import mlx
                return "mlx"
            except ImportError:
                print("⚠️  Apple Silicon détecté mais MLX non installé")
                print("Installation: pip install -r requirements-mlx.txt")
                return "cpu"
        
        # Détection CUDA
        if system in ["Linux", "Windows"]:
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                print("⚠️  PyTorch non installé")
                print("Installation: pip install -r requirements-cuda.txt")
                return "cpu"
        
        print("⚠️  Aucun accélérateur matériel détecté, utilisation du CPU")
        return "cpu"
    
    def _print_config(self):
        """Affiche la configuration détectée"""
        print("=" * 60)
        print("🔧 Configuration Obsidian Organizer")
        print("=" * 60)
        print(f"🖥️  Backend détecté    : {self.backend.upper()}")
        print(f"📁 Vault Obsidian     : {self.vault_path}")
        print(f"📊 Cluster min size   : {self.min_cluster_size} notes")
        print(f"🤖 Embedding model    : {self.embedding_model}")
        print(f"💬 LLM model          : {self.llm_model}")
        print("=" * 60)
        print()


# Instance globale
config = Config()
