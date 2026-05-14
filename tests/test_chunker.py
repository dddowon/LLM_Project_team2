from src.preprocessing.chunker import split_text


def test_split_text_respects_overlap_and_min_size() -> None:
    text = "가나다라마바사아자차카타파하. " * 200
    chunks = split_text(text, chunk_size=120, chunk_overlap=20, min_chunk_chars=30)

    assert len(chunks) > 1
    assert all(len(chunk) >= 30 for chunk in chunks)


def test_split_text_rejects_invalid_overlap() -> None:
    try:
        split_text("hello", chunk_size=100, chunk_overlap=100)
    except ValueError as exc:
        assert "chunk_overlap" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
