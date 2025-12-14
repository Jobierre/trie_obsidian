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
            
            # Chercher d'abord une version MLX pré-convertie
            mlx_model_name = config.llm_model
            
            # Si le modèle n'est pas sur mlx-community, essayer de charger l'original
            # MLX peut convertir à la volée (mais c'est plus lent au premier lancement)
            try:
                self.model, self.tokenizer = load(
                    f"mlx-community/{mlx_model_name.split('/')[-1]}",
                    tokenizer_config={"trust_remote_code": True}
                )
                print(f"✅ Modèle MLX pré-converti trouvé")
            except:
                print(f"⚠️  Version MLX non trouvée, conversion du modèle original...")
                self.model, self.tokenizer = load(
                    mlx_model_name,
                    tokenizer_config={"trust_remote_code": True, "token": config.hf_token}
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
        response = self.generate_fn(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature,
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
            Nom de catégorie proposé (ex: "Tech - Machine Learning")
        """
        # Construire le prompt
        samples_text = "\n\n---\n\n".join([f"Note {i+1}:\n{note[:500]}" for i, note in enumerate(sample_notes)])
        
        prompt = f"""<s>[INST] Tu es un assistant qui aide à organiser des notes Obsidian.

Voici {len(sample_notes)} notes similaires. Analyse leur contenu et propose UN SEUL nom de catégorie descriptif et concis (maximum 3 mots).

Format attendu: "Domaine - Sous-thème" (ex: "Tech - Python", "Santé - Nutrition", "Citations - Philosophie")

Notes:
{samples_text}

Nom de catégorie: [/INST]"""
        
        # Générer la réponse
        category = self.generate(prompt, max_tokens=50, temperature=0.3)
        
        # Nettoyer la réponse (enlever les sauts de ligne, etc.)
        category = category.strip().split('\n')[0].strip()
        
        # Enlever les guillemets si présents
        category = category.strip('"').strip("'")
        
        return category
