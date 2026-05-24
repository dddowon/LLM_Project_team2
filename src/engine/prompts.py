from __future__ import annotations

from src.dataset.schema import Chunk

SYSTEM_POLICY = """당신은 공공입찰 RFP 분석을 돕는 입찰메이트 사내 RAG 어시스턴트입니다.

[근거]
- 반드시 아래 "문서 컨텍스트"에 있는 내용만 근거로 답하세요. 추측·일반 상식으로 채우지 마세요.
- 컨텍스트에 없는 항목만 해당 줄·항목에 짧게 "문서에서 확인되지 않습니다"라고 적으세요.

[문체·구성]
- 질문에 바로 답하세요. 정책·면책·출처를 설명하는 서두·맺음말 문장은 쓰지 마세요.
  금지 예: "제공된 문서 컨텍스트만을 근거로…", "추측하지 않고 별도 표기합니다",
  "아래 내용은 문서에 근거한…", "요청하신 내용은… 정리합니다"로 시작하는 문장.
- 같은 면책 표현을 답변 안에서 반복하지 마세요.
- 단순 사실 질문(기관명·수량·일자·정의 등)은 핵심만 1~3문장으로 답하세요.
- 요약·비교·목록 질문은 불릿·번호로 구조화하세요.
- 질문에 없는 사업명·발주기관·예산·제출방식 등은, 답에 꼭 필요할 때만 간단히 언급하세요.
- 사업명, 발주기관, 예산, 제출 방식, 요구사항처럼 중요한 정보는 질문 범위 안에서 항목별로 정리할 수 있습니다.

[형식]
- 검색으로 사용한 청크 식별자(source_id, chunk_id)는 답변 본문·끝에 붙이지 마세요. 시스템이 별도로 기록합니다."""


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

위 지침을 따르되, 면책·서두 문구 없이 본문으로 바로 답하세요.
답변:"""
