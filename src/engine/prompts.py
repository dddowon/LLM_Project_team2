from __future__ import annotations

from src.dataset.schema import Chunk


SYSTEM_POLICY = """당신은 공공입찰 RFP 분석을 돕는 입찰메이트 사내 RAG 어시스턴트입니다.
반드시 제공된 문서 컨텍스트에 근거해서만 답변하세요.
문서에서 확인할 수 없는 내용은 추측하지 말고 "문서에서 확인되지 않습니다"라고 답하세요.
사업명, 발주기관, 예산, 제출 방식, 요구사항처럼 중요한 정보는 가능하면 항목별로 정리하세요.
답변 마지막에는 근거로 사용한 source_id를 나열하세요."""


def format_context(results: list[tuple[Chunk, float]], max_context_chars: int) -> str:
    blocks: list[str] = []
    total = 0
    for chunk, score in results:
        source_id = chunk.chunk_id
        title = chunk.metadata.get("사업명") or chunk.metadata.get("title") or chunk.metadata.get("file_name", "")
        body = (
            f"[source_id: {source_id} | score: {score:.4f} | title: {title}]\n"
            f"{chunk.text}"
        )
        if total + len(body) > max_context_chars:
            break
        blocks.append(body)
        total += len(body)
    return "\n\n---\n\n".join(blocks)


def build_rag_prompt(question: str, context: str, chat_history: list[dict[str, str]] | None = None) -> str:
    history_text = ""
    if chat_history:
        turns = [f"{item['role']}: {item['content']}" for item in chat_history[-6:]]
        history_text = "\n".join(turns)

    return f"""{SYSTEM_POLICY}

대화 기록:
{history_text or "(없음)"}

문서 컨텍스트:
{context or "(검색된 컨텍스트 없음)"}

사용자 질문:
{question}

답변:"""
