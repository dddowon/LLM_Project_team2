from __future__ import annotations

from openai import BadRequestError, OpenAI


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
        temperature: float | None = 0.2,
    ) -> str:
        request = {
            "model": model,
            "input": prompt,
        }
        if temperature is not None and _supports_temperature(model):
            request["temperature"] = temperature

        try:
            response = self.client.responses.create(**request)
        except BadRequestError as exc:
            if "temperature" not in request or "temperature" not in str(exc):
                raise
            request.pop("temperature")
            response = self.client.responses.create(**request)
        return response.output_text


def _supports_temperature(model: str) -> bool:
    return not model.lower().startswith("gpt-5")
