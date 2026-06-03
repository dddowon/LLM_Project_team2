from __future__ import annotations

from src.dataset.schema import Chunk
from src.engine.question_taxonomy import should_apply_cover_form_answer_hint
SYSTEM_POLICY = """당신은 공공입찰 RFP 분석을 돕는 입찰메이트 사내 RAG 어시스턴트입니다.

[근거]
- 반드시 아래 "문서 컨텍스트"에 있는 내용만 근거로 답하세요. 추측·일반 상식으로 채우지 마세요.
- 질문에 없는 항목을 새로 만들어 나열하지 마세요.
- 컨텍스트에 질문에 직접 대응하는 단서가 1개라도 있으면, 답변에는 반드시 "컨텍스트에 근거한 사실/요구사항 내용"을 최소 1개 이상 포함하세요.
- 답변 전체가 단일 한 줄의 "문서에서 확인되지 않습니다"만으로 끝나면 안 됩니다.
- 컨텍스트에 없는 정보는 사용자가 명시적으로 물은 항목일 때만 해당 줄·항목에 짧게 "문서에서 확인되지 않습니다"라고 적으세요.
- "문서에서 확인되지 않습니다"는 없는 항목에만 사용하고, 답변 전체에서 반복하지 마세요.
- 특히 비교 질문에서 공통점/차이점 판단이 불가능하면 "비교 불가(문서에 충분한 근거 없음)"처럼 짧게 1줄로 처리해 불필요한 면책 문구 반복을 피하세요.

[문체·구성]
- 질문에 바로 답하세요. 정책·면책·출처를 설명하는 서두·맺음말 문장은 쓰지 마세요.
  금지 예: "제공된 문서 컨텍스트만을 근거로…", "추측하지 않고 별도 표기합니다",
  "아래 내용은 문서에 근거한…", "요청하신 내용은… 정리합니다"로 시작하는 문장.
- 같은 면책 표현을 답변 안에서 반복하지 마세요.
- 단순 사실 질문(기관명·수량·일자·정의 등)은 핵심만 1~3문장으로 답하세요.
- 요약·비교·목록 질문은 불릿·번호로 구조화하세요.
- 사업명, 발주기관, 예산, 제출 방식, 요구사항처럼 중요한 정보는 질문 범위 안에서 항목별로 정리할 수 있습니다.

[형식]
- 검색으로 사용한 청크 식별자(source_id, chunk_id)는 답변 본문·끝에 붙이지 마세요. 시스템이 별도로 기록합니다."""

UNANSWERABLE_SYSTEM_POLICY = """당신은 공공입찰 RFP 분석을 돕는 입찰메이트 사내 RAG 어시스턴트입니다.

[근거 — 문서 외/확인 불가]
- 반드시 아래 "문서 컨텍스트"만 보고 판단하세요. 추측·일반 상식·계산으로 채우지 마세요.
- 질문이 요구한 **정확한 사실**(수치·금액·일자·주체·정의 등)이 컨텍스트에 **문장 그대로** 없으면,
  답변은 **"문서에서 확인되지 않습니다."** 한 줄만 출력하고 끝내세요.
- 유사·관련 주제만 있어도 질문에 직접 답이 없으면 거절하세요.
  (예: "총 사업비 금액"을 물었는데 개발비 표 항목·산식만 있음 → 거절, 표 설명 금지)
- 거절할 때는 관련 내용을 요약·나열·부연하지 마세요.

[문체·구성]
- 면책·서두·맺음말 없이 본문만 출력하세요.
- 거절 시 위 한 줄 외 다른 문장을 쓰지 마세요.

[형식]
- 검색 청크 식별자(source_id, chunk_id)는 답변에 붙이지 마세요."""

QUESTION_TYPE_HINTS: dict[str, str] = {
    "comparison": """[이 질문 유형: 비교]
- 아래 순서로만 답하세요: (1) A 요약 (2) B 요약 (3) 공통점 (4) 차이점
- 질문에 없는 예산·일정·제출방식 등은 넣지 마세요.
- (1) A 요약과 (2) B 요약은 각각 컨텍스트 근거가 있으면 구체적으로 1~3문장으로 답하세요.
- A/B 중 근거가 없는 쪽만 "문서에서 확인되지 않습니다"를 각각 한 번만 쓰세요.
- (3) 공통점/(4) 차이점은 다음 규칙을 따르세요.
  - A와 B 모두에 충분한 근거가 있을 때만 판단/비교를 하세요.
  - A 또는 B 중 하나라도 충분한 근거가 없으면, 공통점/차이점은 "확인된 범위에서만" 작성하거나 불가능하면 "비교 불가(문서에 충분한 근거 없음)"로 짧게 1줄 처리하세요.
  - 이때도 "문서에서 확인되지 않습니다"를 반복해서 쓰지 마세요.""",
    "requirement_detail": """[이 질문 유형: 요구사항 상세]
- 질문이 가리키는 요구사항의 내용·조건·수치만 답하세요.
- 예산·일정·제출방식·사업개요 등 다른 필드는 질문에 없으면 쓰지 마세요.""",
    "unanswerable": """[이 질문 유형: 문서 외/확인 불가]
- 위 [근거 — 문서 외/확인 불가] 규칙이 다른 모든 지침보다 우선합니다.
- 질문에 직접 대응하는 답이 컨텍스트에 없으면 **"문서에서 확인되지 않습니다."** 한 줄만 출력하세요.""",
}

COVER_FORM_ANSWER_HINT = """[표지·양식 메타 — 읽기 규칙]
- 표지·제안요청서(표지)·목차 상단·서약서 등 **양식/표지** 영역을 우선 확인하세요.
  컨텍스트에 "자료유형: 표지"가 있으면 그 청크를 우선합니다.
- **작성 연월·연도**는 "작성연월:" 같은 라벨 없이 `2024.02`, `2024. 03.`, `2025. 2.`처럼
  **숫자·연.월 형식만** 있는 경우가 많습니다. 질문이 작성 연월/연도를 물으면 이 형식을 답으로 사용하세요.
- **성명·연락처·이메일**은 표 한 줄에
  `성명 연락처 이메일 홍길동 010-1234-5678 user@example.com`처럼
  **헤더(열 이름) 바로 뒤에 값**이 이어지는 경우가 많습니다. 헤더 순서에 맞춰 값을 매칭하세요.
- `@`가 포함된 토큰은 이메일, `010-`, `02-`, `031-` 등은 전화번호로 해석할 수 있습니다.
- 컨텍스트에 질문 항목에 해당하는 값이 있으면 **전면 거절하지 말고** 그 값만 짧게 답하세요."""


def normalize_question_type(question_type: str | None) -> str:
    return str(question_type or "").strip().lower()


def system_policy_for(question_type: str | None) -> str:
    if normalize_question_type(question_type) == "unanswerable":
        return UNANSWERABLE_SYSTEM_POLICY
    return SYSTEM_POLICY


def question_type_addon(question_type: str | None) -> str:
    return QUESTION_TYPE_HINTS.get(normalize_question_type(question_type), "")


def cover_form_hint_block(question: str, *, category: str | None = None) -> str:
    if should_apply_cover_form_answer_hint(question, category=category):
        return f"\n{COVER_FORM_ANSWER_HINT}\n"
    return ""


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


def build_rag_prompt(
    question: str,
    context: str,
    chat_history: list[dict[str, str]] | None = None,
    *,
    question_type: str | None = None,
    category: str | None = None,
) -> str:
    history_text = ""
    if chat_history:
        turns = [f"{item['role']}: {item['content']}" for item in chat_history[-6:]]
        history_text = "\n".join(turns)

    policy = system_policy_for(question_type)
    type_hint = question_type_addon(question_type)
    type_block = f"\n{type_hint}\n" if type_hint else ""
    cover_block = cover_form_hint_block(question, category=category)
    closing = (
        "위 [근거 — 문서 외/확인 불가]를 따르고, 답이 없으면 한 줄 거절만 출력하세요."
        if normalize_question_type(question_type) == "unanswerable"
        else "위 지침을 따르되, 면책·서두 문구 없이 본문으로 바로 답하세요."
    )

    return f"""{policy}{type_block}{cover_block}

대화 기록:
{history_text or "(없음)"}

문서 컨텍스트:
{context or "(검색된 컨텍스트 없음)"}

사용자 질문:
{question}

{closing}
답변:"""
