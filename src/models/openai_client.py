from __future__ import annotations

from openai import OpenAI


class OpenAIModelClient:
    def __init__(self) -> None:
        self.client = OpenAI()

    def embed_texts(self, texts: list[str], model: str, batch_size: int = 64) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = [text.replace("\n", " ") for text in texts[start : start + batch_size]]
            response = self.client.embeddings.create(model=model, input=batch)
            embeddings.extend(item.embedding for item in response.data)
        return embeddings

    def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.2,
    ) -> str:
        response = self.client.responses.create(
            model=model,
            input=prompt,
            temperature=temperature,
        )
        return response.output_text
