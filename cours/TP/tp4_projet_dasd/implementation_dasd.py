import json
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

class DASFilteringPipeline:
    def __init__(self, student_model_id="unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"):
        """
        Initialise le pipeline de filtrage DAS avec le Student (Local 4-bit).
        """
        self.student_model_id = student_model_id
        print(f"Chargement du modèle étudiant : {self.student_model_id}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.student_model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # On force tout le modèle sur le premier GPU (device 0) pour éviter le split CPU
        self.model = AutoModelForCausalLM.from_pretrained(
            self.student_model_id,
            device_map={"": 0}, 
            trust_remote_code=True,
            dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        self.model.eval()

    def get_student_stats(self, prompt: str, response: str) -> dict:
        """
        Calcule les log-probabilités de la réponse (Student).
        """
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)

        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        input_ids = inputs.input_ids

        # Masquage du prompt
        prompt_messages = [{"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        prompt_tokens = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids
        response_start_idx = prompt_tokens.shape[1]

        labels = input_ids.clone()
        labels[:, :response_start_idx] = -100

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            token_losses = loss_fct(shift_logits.transpose(1, 2), shift_labels)
            token_logprobs = -token_losses

            valid_mask = shift_labels != -100
            valid_logprobs = token_logprobs[valid_mask].cpu().numpy()

        # Moyenne géométrique des probabilités (comme dans DAS)
        # exp(mean(logprob))
        mean_prob = np.exp(np.mean(valid_logprobs)) if len(valid_logprobs) > 0 else 0.0
        
        return {
            "mean_prob": float(mean_prob),
            "num_tokens": int(len(valid_logprobs))
        }

    def process_dataset(self, input_path, output_dir, threshold=0.1, max_entries=None):
        if not os.path.exists(input_path):
            print(f"Erreur : Le fichier {input_path} n'existe pas.")
            return

        with open(input_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        stage1_data = []
        stage2_data = []

        if max_entries is not None:
            dataset = dataset[:max_entries]

        for entry in tqdm(dataset):
            instruction = entry.get("input", "")
            system_prompt = entry.get("system_prompt", "")

            # Traitement Stage 1 (Basse température)
            if "stage1_response" in entry:
                s1 = entry["stage1_response"]
                p_teacher = s1.get("confidence", 0.0) / 100.0
                resp_content = s1.get("content", "")
                
                try:
                    stats = self.get_student_stats(instruction, resp_content)
                    p_student = stats["mean_prob"]
                    divergence = p_teacher - p_student
                    
                    # Logique DAS : On garde si la divergence est positive (Teacher >> Student)
                    # ou si on veut simplement tout le stage 1 pour la base.
                    # Ici on applique un filtrage léger pour le stage 1
                    if divergence > -0.05: # On tolère que le student soit un peu meilleur
                        stage1_data.append(self.format_sharegpt(instruction, resp_content, system_prompt))
                except Exception as e:
                    print(f"Erreur Stage 1 : {e}")

            # Traitement Stage 2 (Haute température)
            if "stage2_response" in entry:
                s2 = entry["stage2_response"]
                p_teacher = s2.get("confidence", 0.0) / 100.0
                resp_content = s2.get("content", "")

                try:
                    stats = self.get_student_stats(instruction, resp_content)
                    p_student = stats["mean_prob"]
                    divergence = p_teacher - p_student

                    # DAS strict pour Stage 2 (Haute diversité)
                    if divergence > threshold:
                        stage2_data.append(self.format_sharegpt(instruction, resp_content, system_prompt))
                except Exception as e:
                    print(f"Erreur Stage 2 : {e}")

        # Sauvegarde
        os.makedirs(output_dir, exist_ok=True)
        s1_path = os.path.join(output_dir, "train_stage1_filtered.json")
        s2_path = os.path.join(output_dir, "train_stage2_filtered.json")

        with open(s1_path, 'w', encoding='utf-8') as f:
            json.dump(stage1_data, f, ensure_ascii=False, indent=2)
        
        with open(s2_path, 'w', encoding='utf-8') as f:
            json.dump(stage2_data, f, ensure_ascii=False, indent=2)

        print(f"Filtrage terminé.")
        print(f"Stage 1: {len(stage1_data)} exemples retenus.")
        print(f"Stage 2: {len(stage2_data)} exemples retenus.")

    def format_sharegpt(self, instruction, response, system_prompt):
        return {
            "conversations": [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": response}
            ],
            "system": system_prompt
        }

if __name__ == "__main__":
    # Paramètres
    DATASET_PATH = "synthetic_dataset.json"
    OUTPUT_DIR = "."
    STUDENT_MODEL = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
    
    # Seuil de divergence (P_T - P_S)
    # Plus il est élevé, plus on est sélectif sur les "Teacher Sentences"
    THRESHOLD = 0.1 

    pipeline = DASFilteringPipeline(student_model_id=STUDENT_MODEL)
    pipeline.process_dataset(DATASET_PATH, OUTPUT_DIR, threshold=THRESHOLD)
