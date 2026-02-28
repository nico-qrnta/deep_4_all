import argparse
import json
from client import DraftAssistantClient
from datasets.helpers import confidence_score
from generate_synthetic_dataset import main as generate_main


def run_single_prompt():
    user_prompt = (
        "Action demandée : full_draft\n"
    )
    client = DraftAssistantClient()
    response = client.ask(user_prompt=user_prompt)
    parsed = json.loads(response.choices[0].message.content)
    logprobs_obj = response.choices[0].logprobs
    confidence = confidence_score(logprobs_obj.content)
    print("=== JSON STRUCTURÉ ===")
    print(json.dumps(parsed, indent=2, ensure_ascii=False))
    print("\n=== CONFIANCE MODÈLE ===")
    print(f"{confidence} / 100")


def main():
    parser = argparse.ArgumentParser(description="LoL Draft Assistant")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic draft dataset")
    args = parser.parse_args()
    if args.generate:
        generate_main()
    else:
        run_single_prompt()


if __name__ == "__main__":
    main()

