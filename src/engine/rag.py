from __future__ import annotations

from src.config import AppConfig
from src.engine.prompts import build_rag_prompt, format_context
from src.engine.vector_store import ChromaVectorStore
from src.models.openai_client import OpenAIModelClient


class RagEngine:
    def __init__(
        self,
        config: AppConfig,
        vector_store: ChromaVectorStore,
        model_client: OpenAIModelClient | None = None,
    ) -> None:
        self.config = config
        self.vector_store = vector_store
        self.model_client = model_client or OpenAIModelClient()

    def retrieve(
        self,
        question: str,
        *,
        doc_id: str | None = None,
    ) -> list[tuple[str, float, dict[str, str], str]]:
        embedding = self.model_client.embed_texts([question], self.config.openai.embedding_model)[0]
        results = self.vector_store.search(
            embedding,
            self.config.retrieval.top_k,
            doc_id=doc_id,
        )
        filtered = [
            (chunk.chunk_id, score, chunk.metadata, chunk.text)
            for chunk, score in results
            if score >= self.config.retrieval.score_threshold
        ]
        return filtered

    def answer(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
        *,
        include_source_text: bool = False,
        doc_id: str | None = None,
        question_type: str | None = None,
    ) -> dict:
        embedding = self.model_client.embed_texts([question], self.config.openai.embedding_model)[0]
        results = [
            item
            for item in self.vector_store.search(
                embedding,
                self.config.retrieval.top_k,
                doc_id=doc_id,
            )
            if item[1] >= self.config.retrieval.score_threshold
        ]
        context = format_context(results, self.config.generation.max_context_chars)
        prompt = build_rag_prompt(question, context, chat_history, question_type=question_type)
        answer = self.model_client.generate(
            prompt=prompt,
            model=self.config.openai.generation_model,
            temperature=self.config.generation.temperature,
        )
        sources: list[dict] = []
        for chunk, score in results:
            row: dict = {"chunk_id": chunk.chunk_id, "score": score, "metadata": chunk.metadata}
            if include_source_text:
                row["text"] = chunk.text
            sources.append(row)
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "doc_id": doc_id,
            "resolved_doc_id": self.vector_store.resolve_doc_id(doc_id),
        }
