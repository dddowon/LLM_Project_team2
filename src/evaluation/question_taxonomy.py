"""Backward-compatible re-export (prefer src.engine.question_taxonomy)."""
from src.engine.question_taxonomy import (
    COVER_FORM_CATEGORY,
    COVER_FORM_CONTEXT_PATTERN,
    COVER_FORM_DATE_PATTERN,
    COVER_FORM_TOPIC_PHRASES,
    is_cover_form_metadata_question,
    should_apply_cover_form_answer_hint,
)

__all__ = [
    "COVER_FORM_CATEGORY",
    "COVER_FORM_CONTEXT_PATTERN",
    "COVER_FORM_DATE_PATTERN",
    "COVER_FORM_TOPIC_PHRASES",
    "is_cover_form_metadata_question",
    "should_apply_cover_form_answer_hint",
]
