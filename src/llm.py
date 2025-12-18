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
        elif self.backend == "cpu":
            self._load_cpu()
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
            
            # Vérifier si c'est Ministral 3 (nécessite API spécifique)
            is_ministral3 = "Ministral-3" in config.llm_model or "ministral3" in config.llm_model.lower()
            
            if is_ministral3:
                print(f"🔧 Modèle Ministral 3 détecté : {config.llm_model}")
                from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend, BitsAndBytesConfig
                
                # Configuration 4-bit
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                print("📦 Chargement du tokenizer MistralCommonBackend...")
                self.tokenizer = MistralCommonBackend.from_pretrained(config.llm_model)
                
                print("🚀 Chargement du modèle Mistral3 (peut prendre quelques minutes)...")
                self.model = Mistral3ForConditionalGeneration.from_pretrained(
                    config.llm_model,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    quantization_config=quantization_config,
                    token=config.hf_token
                )
                
                self._is_ministral3 = True
                print("✅ Modèle Ministral 3 chargé avec succès")
                
            else:
                # Modèles classiques (Mistral, Llama, etc.)
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    config.llm_model,
                    token=config.hf_token,
                    trust_remote_code=True
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.llm_model,
                    device_map="auto",
                    quantization_config=quantization_config,
                    token=config.hf_token,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16
                )
                
                self._is_ministral3 = False
            
            self.model.eval()
            
        except ImportError as e:
            if "Mistral3ForConditionalGeneration" in str(e):
                raise ImportError(
                    "\n❌ Mistral3ForConditionalGeneration non trouvé.\n"
                    "Installez transformers depuis git:\n"
                    "   pip uninstall transformers -y\n"
                    "   pip install git+https://github.com/huggingface/transformers.git\n"
                    "   pip install mistral-common>=1.8.6\n"
                ) from e
            else:
                raise ImportError(
                    "PyTorch/Transformers non installé. Installation: pip install -r requirements-cuda.txt"
                ) from e
    
    def _load_cpu(self):
        """Charge le modèle en mode CPU (sans CUDA, sans quantization lourde)"""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print("⚠️  Mode CPU: performances limitées, considérez un modèle plus petit")
            
            # Charger le tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.llm_model,
                token=config.hf_token,
                trust_remote_code=True
            )
            
            # Charger le modèle en float32 sur CPU
            self.model = AutoModelForCausalLM.from_pretrained(
                config.llm_model,
                token=config.hf_token,
                trust_remote_code=True,
                dtype="auto",
                device_map="cpu",
                low_cpu_mem_usage=True
            )
            
            self.model.eval()
            
        except ImportError as e:
            raise ImportError(
                "PyTorch/Transformers non installé. Installation: pip install -r requirements-cuda.txt"
            ) from e
        except KeyError as e:
            raise ValueError(
                f"\n❌ Le modèle {config.llm_model} n'est pas encore supporté par transformers.\n"
                f"\n💡 Utilisez un modèle compatible dans .env:\n"
                f"   LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3\n"
                f"   LLM_MODEL=mistralai/Ministral-8B-Instruct-2410\n"
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
        elif self.backend in ("cuda", "cpu"):
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
        
        # Ministral 3 utilise une API différente
        if hasattr(self, '_is_ministral3') and self._is_ministral3:
            # Format pour Ministral 3 (texte seul)
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ]
            
            # Tokeniser avec MistralCommonBackend
            tokenized = self.tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True)
            tokenized["input_ids"] = tokenized["input_ids"].to(device="cuda")
            
            # Générer
            with torch.no_grad():
                output = self.model.generate(
                    **tokenized,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0
                )[0]
            
            # Décoder seulement la partie générée
            response = self.tokenizer.decode(output[len(tokenized["input_ids"][0]):])
            return response.strip()
        
        else:
            # Modèles classiques (AutoModelForCausalLM)
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
    
    def categorize_cluster(self, sample_notes: List[Dict[str, str]]) -> str:
        """
        Génère un nom de catégorie à partir d'échantillons de notes.
        
        Args:
            sample_notes: Liste de dicts avec 'title' et 'content'
        
        Returns:
            Nom de catégorie proposé ou "Divers" si hétérogène
        """
        # Construire le prompt avec titres + contenus
        samples_text = "\n\n---\n\n".join([
            f"Note {i+1}:\nTitre: {note['title']}\nContenu: {note['content'][:600]}" 
            for i, note in enumerate(sample_notes)
        ])
        
        prompt = f"""<s>[INST] Tu es un expert en organisation de notes Obsidian.

ANALYSE {len(sample_notes)} NOTES CI-DESSOUS:

{samples_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TÂCHE: Trouve le thème commun précis de ces notes.

RÈGLES STRICTES:
1. Si les notes sont COHÉRENTES (même thème), réponds: "Domaine - Sous-thème"
   Exemples:
   • Notes sur Plakar/S3/backup → "Tech - Backup"
   • Notes sur carrelage/sol SDB → "Maison - Travaux SDB"
   • Notes sur JavaScript/Python → "Dev - Langages"
   • Notes sur trading/finance → "Finance - Investissement"

2. Si les notes sont HÉTÉROGÈNES (thèmes différents), réponds: "DIVERS"
   Exemple: mélange torrent + carrelage + finance → DIVERS

3. Utilise le vocabulaire des TITRES (pas d'invention)
4. Maximum 4 mots, format "X - Y"
5. Pas d'explication, juste la catégorie

RÉPONSE:[/INST]"""
        
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
    
    def discover_categories(self, sample_notes: List[Dict[str, str]], count: int = 8) -> List[Dict[str, str]]:
        """
        Analyse un échantillon de notes et propose des catégories globales.
        
        Args:
            sample_notes: Liste de dicts avec 'title' et 'content' (échantillon représentatif)
            count: Nombre de catégories à proposer
        
        Returns:
            Liste de dicts avec 'name' et 'description'
        """
        # Préparer un résumé des notes
        notes_summary = "\n\n".join([
            f"- {note['title']}: {note['content'][:200]}..."
            for note in sample_notes[:30]  # Max 30 notes pour le prompt
        ])
        
        prompt = f"""<s>[INST] Tu es un expert en organisation de connaissances (PKM).

MISSION: Analyse ces notes Obsidian et propose {count} CATÉGORIES PRINCIPALES pour organiser tout le vault.

ÉCHANTILLON DE NOTES ({len(sample_notes[:30])} notes):
{notes_summary}

RÈGLES:
1. Propose {count} catégories qui couvrent les THÈMES PRINCIPAUX
2. Format: "Domaine - Sous-thème" (ex: "Tech - Infrastructure", "Dev - Python")
3. Chaque catégorie doit avoir:
   - Un nom court et descriptif (2-4 mots)
   - Une description claire (1 phrase)
4. Ajoute toujours "Divers" en dernière catégorie

EXEMPLES:
Tech - Backup: Sauvegarde de données, outils comme Plakar, S3, rsync
Dev - Programmation: Code, langages, frameworks, tutoriels
Maison - Travaux: Rénovation, bricolage, matériaux, achats
Finance: Investissement, trading, comptabilité

RÉPONDS AU FORMAT JSON UNIQUEMENT:
[
  {{"name": "Tech - Infrastructure", "description": "..."}},
  {{"name": "Dev - Programmation", "description": "..."}},
  ...
][/INST]"""
        
        # Générer avec plus de tokens pour avoir toutes les catégories
        response = self.generate(prompt, max_tokens=800, temperature=0.3)
        
        print(f"🔍 Réponse LLM ({len(response)} caractères)")
        
        # Parser le JSON
        import json
        import re
        
        try:
            # Méthode 1: Extraire le JSON entre crochets
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                
                # Nettoyer le JSON (enlever les virgules traînantes, etc.)
                json_str = re.sub(r',\s*]', ']', json_str)  # Fix trailing comma
                json_str = re.sub(r',\s*}', '}', json_str)  # Fix trailing comma in objects
                
                categories = json.loads(json_str)
                
                # Valider la structure
                if isinstance(categories, list) and len(categories) > 0:
                    valid_cats = []
                    for c in categories:
                        if isinstance(c, dict) and 'name' in c:
                            valid_cats.append({
                                'name': c.get('name', '').strip(),
                                'description': c.get('description', '').strip()
                            })
                    if valid_cats:
                        print(f"✅ {len(valid_cats)} catégories parsées avec succès")
                        return valid_cats
        except json.JSONDecodeError as e:
            print(f"⚠️  Erreur JSON: {e}")
        except Exception as e:
            print(f"⚠️  Erreur parsing: {e}")
        
        # Méthode 2: Regex pour extraire les catégories
        print("⚠️  JSON invalide, extraction par regex...")
        categories = []
        
        # Pattern pour trouver "name": "..." et "description": "..."
        pattern = r'"name"\s*:\s*"([^"]+)".*?"description"\s*:\s*"([^"]+)"'
        matches = re.findall(pattern, response, re.DOTALL)
        
        for name, desc in matches:
            categories.append({'name': name.strip(), 'description': desc.strip()})
        
        if categories:
            print(f"✅ {len(categories)} catégories extraites par regex")
            return categories
        
        # Méthode 3: Fallback - parser ligne par ligne
        print("⚠️  Regex échoué, parsing ligne par ligne...")
        for line in response.split('\n'):
            line = line.strip()
            # Pattern: "Catégorie: description" ou "- Catégorie: description"
            if ':' in line and len(line) > 5:
                clean = line.lstrip('- •*"\'{[')
                if clean.startswith('name') or clean.startswith('description'):
                    continue
                parts = clean.split(':', 1)
                if len(parts) == 2:
                    name = parts[0].strip().strip('"\'')
                    desc = parts[1].strip().strip('"\']},.')
                    if name and len(name) < 50:
                        categories.append({'name': name, 'description': desc})
        
        if categories:
            print(f"✅ {len(categories)} catégories extraites manuellement")
            return categories
        
        print("❌ Impossible de parser les catégories, utilisation des catégories par défaut")
        from .category_manager import CategoryManager
        return CategoryManager().create_default_categories()
    
    def choose_category(
        self, 
        note_title: str, 
        note_content: str, 
        available_categories: List[Dict[str, str]]
    ) -> str:
        """
        Choisit la meilleure catégorie pour une note parmi une liste prédéfinie.
        
        Args:
            note_title: Titre de la note
            note_content: Contenu de la note
            available_categories: Liste des catégories disponibles
        
        Returns:
            Nom exact de la catégorie choisie
        """
        # Construire la liste des catégories pour le prompt
        categories_text = "\n".join([
            f"{i+1}. {cat['name']}: {cat['description']}"
            for i, cat in enumerate(available_categories)
        ])
        
        # Limiter le contenu
        content_excerpt = note_content[:800]
        
        prompt = f"""<s>[INST] Classe cette note Obsidian dans UNE catégorie.

NOTE:
Titre: {note_title}
Contenu: {content_excerpt}

CATÉGORIES DISPONIBLES:
{categories_text}

RÈGLES:
1. Choisis la catégorie LA PLUS PERTINENTE
2. Réponds UNIQUEMENT avec le nom exact de la catégorie
3. Si vraiment aucune ne convient, choisis "Divers"

Catégorie:[/INST]"""
        
        response = self.generate(prompt, max_tokens=20, temperature=0.1)
        
        # Nettoyer la réponse
        category = response.strip().strip('"').strip("'").strip()
        
        # Prendre la première ligne
        category = category.split('\n')[0].strip()
        
        # Enlever numérotation si présente (ex: "1. Tech - Backup" → "Tech - Backup")
        if category and category[0].isdigit():
            category = category.split('.', 1)[-1].strip()
        
        # Vérifier que la catégorie existe
        category_names = [cat['name'] for cat in available_categories]
        
        if category not in category_names:
            # Essayer de trouver une correspondance partielle
            for cat_name in category_names:
                if cat_name.lower() in category.lower() or category.lower() in cat_name.lower():
                    return cat_name
            # Fallback sur Divers
            return "Divers" if "Divers" in category_names else category_names[0]
        
        return category

