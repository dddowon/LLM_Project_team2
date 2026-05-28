from __future__ import annotations

from openai import BadRequestError, OpenAI

from src.utils.embedding_text import truncate_text_for_embedding, truncate_text_for_embedding_retry


class OpenAIModelClient:
    def __init__(self) -> None:
        self.client = OpenAI()

    def _embed_one(self, text: str, model: str) -> list[float]:
        last_error: BadRequestError | None = None
        for prepared in truncate_text_for_embedding_retry(text, model=model):
            try:
                response = self.client.embeddings.create(model=model, input=[prepared])
                return list(response.data[0].embedding)
            except BadRequestError as exc:
                if "maximum input length" not in str(exc).lower():
                    raise
                last_error = exc
        if last_error is not None:
            raise last_error
        raise RuntimeError("Failed to prepare embedding input within token limits")

    def embed_texts(self, texts: list[str], model: str, batch_size: int = 64) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch_raw = texts[start : start + batch_size]
            batch = []
            for text in batch_raw:
                prepared, _ = truncate_text_for_embedding(text, model=model)
                batch.append(prepared)
            try:
                response = self.client.embeddings.create(model=model, input=batch)
                embeddings.extend(item.embedding for item in response.data)
            except BadRequestError as exc:
                if "maximum input length" not in str(exc).lower():
                    raise
                for text in batch_raw:
                    embeddings.append(self._embed_one(text, model))
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
        if temperature is not None and supports_chat_temperature(model):
            request["temperature"] = temperature

        try:
            response = self.client.responses.create(**request)
        except BadRequestError as exc:
            if "temperature" not in request or "temperature" not in str(exc):
                raise
            request.pop("temperature")
            response = self.client.responses.create(**request)
        return response.output_text


def supports_chat_temperature(model: str) -> bool:
    """GPT-5 계열은 chat/responses API에서 temperature 미지원인 경우가 많음."""
    return not model.lower().startswith("gpt-5")
