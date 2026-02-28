import os
from dotenv import load_dotenv
from openai import OpenAI
from prompts import SYSTEM_PROMPT, RESPONSE_FORMAT

load_dotenv()

class DraftAssistantClient:
    def __init__(self):
        api_key = os.getenv("INFOMANIAK_API_KEY")
        product_id = os.getenv("INFOMANIAK_PRODUCT_ID")
        self.system_prompt = SYSTEM_PROMPT
        self.response_format = RESPONSE_FORMAT

        if not api_key:
            raise RuntimeError("INFOMANIAK_API_KEY manquante")
        if not product_id:
            raise RuntimeError("INFOMANIAK_PRODUCT_ID manquant")

        self.client = OpenAI(
            api_key=api_key,
            base_url=f"https://api.infomaniak.com/2/ai/{product_id}/openai/v1"
        )

        self.model = "openai/gpt-oss-120b"

    def ask(self, user_prompt: str, temperature: float = 0.2, logprobs: bool = True, top_logprobs: int = 1):
        return self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=2500,
            response_format=self.response_format,
            logprobs=logprobs,
            top_logprobs=top_logprobs
        )
