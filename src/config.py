from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class PathConfig(BaseModel):
    raw_data_dir: Path
    metadata_csv: Path
    processed_dir: Path
    index_dir: Path
    evaluation_set: Path
    evaluation_output: Path


class OpenAIConfig(BaseModel):
    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "gpt-5-mini"


class VectorStoreConfig(BaseModel):
    provider: str = "chroma"
    index_type: str = "hnsw"
    distance_metric: str = "cosine"


class ChunkingConfig(BaseModel):
    chunk_size: int = 1200
    chunk_overlap: int = 200
    min_chunk_chars: int = 80


class RetrievalConfig(BaseModel):
    top_k: int = 5
    score_threshold: float = 0.0


class GenerationConfig(BaseModel):
    max_context_chars: int = 12000
    temperature: float = 0.2


class AppConfig(BaseModel):
    paths: PathConfig
    openai: OpenAIConfig
    vector_store: VectorStoreConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig




def load_config(path: str | Path = "configs/default.yaml") -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return AppConfig.model_validate(data)
