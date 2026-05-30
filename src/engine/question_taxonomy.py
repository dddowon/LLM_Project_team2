"""Question taxonomy helpers (category, cover-form detection)."""
from __future__ import annotations

import re

COVER_FORM_CATEGORY = "부록·양식"

COVER_FORM_TOPIC_PHRASES: tuple[str, ...] = (
    "이메일",
    "e-mail",
    "e mail",
    "연락처",
    "전화번호",
    "전화 번호",
    "휴대폰",
    "휴대전화",
    "성명",
    "담당자",
    "사업책임자",
    "작성 연월",
    "작성연월",
    "작성 연도",
    "작성연도",
    "연도 및 월",
    "연월",
)

COVER_FORM_CONTEXT_PATTERN = re.compile(
    r"표지|제안요청서\s*\(?\s*표지|목차|서약서|양식|표\s*상단",
    re.I,
)

COVER_FORM_DATE_PATTERN = re.compile(
    r"작성\s*(?:연월|연도)|(?:연도|년도)\s*(?:및|/)?\s*월|연월",
    re.I,
)


def is_cover_form_metadata_question(question: str) -> bool:
    """표지·양식 메타(연락처·성명·이메일·작성연월 등) 질문 여부."""
    q = str(question or "").strip()
    if not q:
        return False
    q_cf = q.casefold()

    if "@" in q or any(p.casefold() in q_cf for p in COVER_FORM_TOPIC_PHRASES):
        return True
    if COVER_FORM_DATE_PATTERN.search(q):
        return True
    if COVER_FORM_CONTEXT_PATTERN.search(q) and re.search(
        r"연락|전화|이메일|성명|담당|연월|연도|작성",
        q,
        re.I,
    ):
        return True
    return False


def should_apply_cover_form_answer_hint(
    question: str,
    *,
    category: str | None = None,
) -> bool:
    if str(category or "").strip() == COVER_FORM_CATEGORY:
        return True
    return is_cover_form_metadata_question(question)
