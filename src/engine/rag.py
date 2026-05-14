from __future__ import annotations

from src.config import AppConfig
from src.engine.prompts import build_rag_prompt, format_context
from src.engine.vector_store import FaissVectorStore
from src.models.openai_client import OpenAIModelClient


class RagEngine:
    def __init__(
        self,
        config: AppConfig,
        vector_store: FaissVectorStore,
        model_client: OpenAIModelClient | None = None,
    ) -> None:
        self.config = config
        self.vector_store = vector_store
        self.model_client = model_client or OpenAIModelClient()

    def retrieve(self, question: str) -> list[tuple[str, float, dict[str, str], str]]:
        embedding = self.model_client.embed_texts([question], self.config.openai.embedding_model)[0]
        results = self.vector_store.search(embedding, self.config.retrieval.top_k)
        filtered = [
            (chunk.chunk_id, score, chunk.metadata, chunk.text)
            for chunk, score in results
            if score >= self.config.retrieval.score_threshold
        ]
        return filtered

    def answer(self, question: str, chat_history: list[dict[str, str]] | None = None) -> dict:
        embedding = self.model_client.embed_texts([question], self.config.openai.embedding_model)[0]
        results = [
            item
            for item in self.vector_store.search(embedding, self.config.retrieval.top_k)
            if item[1] >= self.config.retrieval.score_threshold
        ]
        context = format_context(results, self.config.generation.max_context_chars)
        prompt = build_rag_prompt(question, context, chat_history)
        answer = self.model_client.generate(
            prompt=prompt,
            model=self.config.openai.generation_model,
            temperature=self.config.generation.temperature,
        )
        return {
            "question": question,
            "answer": answer,
            "sources": [
                {"chunk_id": chunk.chunk_id, "score": score, "metadata": chunk.metadata}
                for chunk, score in results
            ],
        }
