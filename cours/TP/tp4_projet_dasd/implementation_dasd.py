import json
import os
import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------
# Config 4-bit
# -----------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# -----------------------
# Pipeline DAS Filtering
# -----------------------
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

        self.model = AutoModelForCausalLM.from_pretrained(
            self.student_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        self.model.eval()

    # -----------------------
    # Utilities
    # -----------------------
    @staticmethod
    def split_into_sentences(text: str):
        sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
        return [s.strip() for s in sentences if s.strip()]

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
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
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

        mean_prob = np.exp(np.mean(valid_logprobs)) if len(valid_logprobs) > 0 else 0.0

        return {
            "mean_prob": float(mean_prob),
            "num_tokens": int(len(valid_logprobs))
        }

    # -----------------------
    # DAS phrase-level
    # -----------------------
    def compute_das_for_response(self, response_content: str, teacher_confidence: float):
        """
        Calcule la divergence phrase par phrase et la densité de Teacher Sentences.
        """
        sentences = self.split_into_sentences(response_content)
        das_sentences = []

        for sent in sentences:
            stats = self.get_student_stats("", sent)
            p_student = stats["mean_prob"]
            p_teacher = teacher_confidence / 100.0
            divergence = p_teacher - p_student

            if divergence > 0.05:
                label = "Teacher"
            elif abs(divergence) <= 0.05:
                label = "Shared"
            else:
                label = "Student"

            das_sentences.append({
                "sentence": sent,
                "p_teacher": p_teacher,
                "p_student": p_student,
                "divergence": divergence,
                "label": label
            })

        num_teacher = sum(1 for s in das_sentences if s["label"] == "Teacher")
        density = num_teacher / max(1, len(das_sentences))

        return das_sentences, density

    @staticmethod
    def keep_response(density, min_density=0.3):
        return density >= min_density

    @staticmethod
    def format_sharegpt(instruction, response, system_prompt):
        return {
            "conversations": [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": response}
            ],
            "system": system_prompt
        }

    # -----------------------
    # Pipeline principal
    # -----------------------
    def process_dataset(self, input_path, output_dir, stage1_density=0.1, stage2_density=0.3, max_entries=None):
        if not os.path.exists(input_path):
            print(f"Erreur : Le fichier {input_path} n'existe pas.")
            return

        with open(input_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        stage1_data = []
        stage2_data = []
        stage1_densities = []
        stage2_densities = []

        if max_entries is not None:
            dataset = dataset[:max_entries]

        for entry in tqdm(dataset):
            instruction = entry.get("input", "")
            system_prompt = entry.get("system_prompt", "")

            # ---------------- Stage 1 ----------------
            if "stage1_response" in entry:
                s1 = entry["stage1_response"]
                resp_content = s1.get("content", "")
                teacher_conf = s1.get("confidence", 0.0)

                try:
                    _, density = self.compute_das_for_response(resp_content, teacher_conf)
                    stage1_densities.append(density)
                    if self.keep_response(density, min_density=stage1_density):
                        stage1_data.append(self.format_sharegpt(instruction, resp_content, system_prompt))
                except Exception as e:
                    print(f"Erreur Stage 1 : {e}")

            # ---------------- Stage 2 ----------------
            if "stage2_response" in entry:
                s2 = entry["stage2_response"]
                resp_content = s2.get("content", "")
                teacher_conf = s2.get("confidence", 0.0)

                try:
                    _, density = self.compute_das_for_response(resp_content, teacher_conf)
                    stage2_densities.append(density)
                    if self.keep_response(density, min_density=stage2_density):
                        stage2_data.append(self.format_sharegpt(instruction, resp_content, system_prompt))
                except Exception as e:
                    print(f"Erreur Stage 2 : {e}")

        # ---------------- Sauvegarde ----------------
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

        # ---------------- Histogramme DAS ----------------
        plt.figure(figsize=(10, 6))
        plt.hist(stage1_densities, bins=20, alpha=0.6, label="Stage 1")
        plt.hist(stage2_densities, bins=20, alpha=0.6, label="Stage 2")
        plt.xlabel("Densité de Teacher Sentences")
        plt.ylabel("Nombre de réponses")
        plt.title("Distribution DAS par Stage")
        plt.legend()

        # --- sauvegarde ---
        plt.savefig(os.path.join(output_dir, "histogram_das.png"), dpi=300)
        # --- affichage ---
        plt.show()


# -----------------------
# MAIN EXECUTION
# -----------------------
if __name__ == "__main__":
    DATASET_PATH = "synthetic_dataset.json"
    OUTPUT_DIR = "."
    STUDENT_MODEL = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"

    pipeline = DASFilteringPipeline(student_model_id=STUDENT_MODEL)
    pipeline.process_dataset(DATASET_PATH, OUTPUT_DIR)
