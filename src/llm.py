"""
Wrapper pour le LLM (Ministral-3B) avec support MLX et CUDA
"""
from typing import List, Dict
from .config import config


class LLMGenerator:
    """Génère du texte avec Ministral-3B"""
    
    def __init__(self):
        self.backend = config.backend
        print(f"🤖 Chargement du LLM: {config.llm_model} (backend: {self.backend})")
        
        if self.backend == "mlx":
            self._load_mlx()
        elif self.backend == "cuda":
            self._load_cuda()
        else:
            raise ValueError(f"Backend {self.backend} non supporté pour le LLM")
        
        print("✅ LLM chargé avec succès")
    
    def _load_mlx(self):
        """Charge le modèle avec MLX (Apple Silicon)"""
        try:
            from mlx_lm import load, generate
            from transformers import AutoTokenizer
            
            # Si le modèle commence déjà par mlx-community/, le charger directement
            mlx_model_name = config.llm_model
            
            if mlx_model_name.startswith("mlx-community/"):
                print(f"🔍 Chargement du modèle MLX: {mlx_model_name}")
                try:
                    self.model, self.tokenizer = load(
                        mlx_model_name,
                        tokenizer_config={"trust_remote_code": True}
                    )
                    print(f"✅ Modèle MLX chargé")
                    self.generate_fn = generate
                    return
                except Exception as e:
                    print(f"❌ Erreur de chargement: {e}")
                    raise
            
            # Sinon, essayer de trouver une version MLX pré-convertie
            model_name_short = mlx_model_name.split('/')[-1]
            
            # Essayer les versions quantifiées en premier (plus légères)
            mlx_versions = [
                f"mlx-community/{model_name_short}-4bit",
                f"mlx-community/{model_name_short}-8bit",
                f"mlx-community/{model_name_short}"
            ]
            
            print(f"🔍 Recherche d'une version MLX optimisée...")
            
            for mlx_version in mlx_versions:
                try:
                    print(f"   → Test: {mlx_version}")
                    # Test rapide de l'existence du modèle
                    from huggingface_hub import model_info
                    model_info(mlx_version, token=config.hf_token)
                    
                    # Si on arrive ici, le modèle existe, on le charge
                    print(f"✅ Modèle trouvé, chargement en cours...")
                    self.model, self.tokenizer = load(
                        mlx_version,
                        tokenizer_config={"trust_remote_code": True}
                    )
                    print(f"✅ Modèle chargé: {mlx_version}")
                    self.generate_fn = generate
                    return
                except Exception:
                    continue
            
            print(f"⚠️  Aucune version MLX pré-convertie trouvée")
            print(f"\n💡 Solutions alternatives:")
            print(f"   1. Choisissez un modèle MLX pré-converti disponible:")
            print(f"      • mlx-community/Mistral-7B-Instruct-v0.3-4bit")
            print(f"      • mlx-community/Llama-3.2-3B-Instruct-4bit")
            print(f"      • mlx-community/Qwen2.5-7B-Instruct-4bit")
            print(f"\n   2. Modifiez votre .env:")
            print(f"      LLM_MODEL=mlx-community/Mistral-7B-Instruct-v0.3-4bit")
            print(f"\n   3. Consultez les modèles disponibles:")
            print(f"      https://huggingface.co/mlx-community")
            
            raise ValueError(
                f"Aucun modèle MLX compatible trouvé pour {mlx_model_name}. "
                f"Utilisez un modèle mlx-community/ dans votre .env"
            )
            
            self.generate_fn = generate
            
        except ImportError as e:
            raise ImportError(
                "MLX non installé. Installation: pip install mlx mlx-lm"
            ) from e
    
    def _load_cuda(self):
        """Charge le modèle avec PyTorch + CUDA (quantization 4-bit)"""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            
            # Configuration 4-bit pour économiser la VRAM
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Charger le tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.llm_model,
                token=config.hf_token,
                trust_remote_code=True
            )
            
            # Charger le modèle avec quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                config.llm_model,
                device_map="auto",
                quantization_config=quantization_config,
                token=config.hf_token,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
            
            self.model.eval()  # Mode évaluation
            
        except ImportError as e:
            raise ImportError(
                "PyTorch/Transformers non installé. Installation: pip install -r requirements-cuda.txt"
            ) from e
    
    def generate(self, prompt: str, max_tokens: int = 150, temperature: float = 0.3) -> str:
        """
        Génère une réponse à partir d'un prompt.
        
        Args:
            prompt: Le prompt d'instruction
            max_tokens: Nombre maximum de tokens à générer
            temperature: Température (0.0 = déterministe, 1.0 = créatif)
        
        Returns:
            Le texte généré
        """
        if self.backend == "mlx":
            return self._generate_mlx(prompt, max_tokens, temperature)
        elif self.backend == "cuda":
            return self._generate_cuda(prompt, max_tokens, temperature)
    
    def _generate_mlx(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Génération avec MLX"""
        # mlx_lm.generate retourne directement le texte complet
        response = self.generate_fn(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False
        )
        return response
    
    def _generate_cuda(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Génération avec CUDA"""
        import torch
        
        # Encoder le prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Générer
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Décoder (en enlevant le prompt)
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def categorize_cluster(self, sample_notes: List[str]) -> str:
        """
        Génère un nom de catégorie à partir d'échantillons de notes.
        
        Args:
            sample_notes: Liste de contenus de notes représentatives du cluster
        
        Returns:
            Nom de catégorie proposé (ex: "Tech - Python")
        """
        # Construire le prompt avec plus de contenu (800 chars au lieu de 500)
        samples_text = "\n\n---\n\n".join([f"Note {i+1}:\n{note[:800]}" for i, note in enumerate(sample_notes)])
        
        prompt = f"""
Tu es un expert en Knowledge Management (PKM) pour Obsidian. Ta mission est de classer le groupe de notes suivant dans une hiérarchie stricte.

CONTEXTE :
Tu dois analyser {len(sample_notes)} notes et déterminer leur point commun unique pour générer un titre de catégorie.

RÈGLES ABSOLUES DE FORMATTAGE :
1. Réponds UNIQUEMENT avec le titre. (Pas de markdown, pas de gras, pas d'intro).
2. Format : "DOMAINE - SOUS-THÈME"
   - Le DOMAINE doit être choisi dans la liste imposée ci-dessous.
   - Le SOUS-THÈME doit être court (1 à 3 mots), précis et technique.

MÉTHODOLOGIE POUR LE SOUS-THÈME :
1. Sois "Chirurgical" : Utilise le vocabulaire technique présent dans les notes.
2. Évite l'abstraction : Préfère "Numpy Array" à "Programmation Python".
3. Évite le générique : Si les notes parlent de factures, écris "Comptabilité" ou "Factures", surtout pas "Administratif Divers".
4. Si le contenu ne rentre pas parfaitement, choisis le Domaine le plus proche logiquement.

EXEMPLES (Inputs -> Outputs attendus) :
- Notes sur le Jardinage -> "Maison - Jardin" (Si Maison est autorisé)
- Notes sur Docker/LXC -> "Tech - Conteneurs" ou "Dev - Ops"
- Notes sur un Bilan sanguin -> "Santé - Analyses"

NOTES À CLASSER :
{samples_text}

RÉPONSE (Strictement "Domaine - Sous-thème") : [/INST]"""
        
        # Générer la réponse
        category = self.generate(prompt, max_tokens=20, temperature=0.1)
        
        # Nettoyer la réponse aggressivement
        category = category.strip()
        
        # Prendre seulement la première ligne
        category = category.split('\n')[0].strip()
        
        # Enlever guillemets, tirets en début, points, etc.
        category = category.strip('"').strip("'").strip('- ').strip('.').strip()
        
        # Si contient "Catégorie:", "Nom:", etc., extraire juste après
        if ':' in category:
            category = category.split(':', 1)[1].strip()
        
        # Limiter à 50 caractères max
        if len(category) > 50:
            category = category[:50].rsplit(' ', 1)[0]
        
        return category
