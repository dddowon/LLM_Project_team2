# 운영 파이프라인 (HWP + OCR 전체)

RFP 100건(`data/raw/RFP_file_100`) 기준으로 HWP/PDF 본문 + 이미지 OCR 결과를 RAG·평가에 넣는 표준 순서입니다.

## 환경/원본 준비

```bash
cd ~/LLM_Project_team2
source ~/llm_team2/bin/activate
export PYTHONPYCACHEPREFIX=.cache/pycache

pip install -e ".[dev,hwp]"   # 최초 1회
cp .env.example .env

mkdir -p outputs logs checkpoints data/raw data/v2 \
  data/v2/ocr_images data/v2/ocr_outputs data/v2/ocr_rag \
  .cache/pycache
```

원본 zip(`unzip` 없을 때):

```bash
python - <<'PY'
import zipfile
from pathlib import Path
z = Path("data/raw/RFP_file_100.zip")
d = Path("data/raw/RFP_file_100")
d.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(z) as f:
    f.extractall(d)
print("done:", d)
PY
```

## 표준 실행 순서

### 1) 텍스트 파이프라인

```bash
python -m src.cli run-pipeline \
  --input-dir "data/raw/RFP_file_100" \
  --output-dir "data/v2" \
  --recursive \
  --force-real
```

### 2) OCR 추론 + handoff chunk export

```bash
conda activate ocr_vl15
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}   # WSL + NVIDIA GPU일 때만

OCR_USE_GT=0 ./scripts/run_ocr_stage.sh
```

### 3) OCR chunk 임베딩 + OCR 전용 인덱스

```bash
source ~/llm_team2/bin/activate
OCR_BASE="data/v2/ocr_rag/paddleocr_vl/v4_table_filtered_260531"

python -m src.cli embed-jsonl \
  --input "${OCR_BASE}/ocr_input_chunks.jsonl" \
  --output "${OCR_BASE}/ocr_input_embedded.jsonl" \
  --force-real
python -m src.cli build-chroma \
  --input "${OCR_BASE}/ocr_input_embedded.jsonl" \
  --index-dir "${OCR_BASE}/chroma_index"
```

### 4) 통합 Chroma(query/eval 공용)

```bash
python -m src.cli --config configs/default.yaml merge-embedded \
  --input-dir data/v2 \
  --pattern "*embedded*.jsonl" \
  --index-dir checkpoints/chroma_openai
```

`eval`/`query`는 `checkpoints/chroma_openai`를 사용합니다.

### 5) 질의 smoke test

```bash
python -m src.cli --config configs/default.yaml query \
  "BIFF 온라인서비스 재개발 사업의 주요 과업 범위는 뭐야?"
```

## HWP 파싱/청킹/임베딩(단독)

원클릭:

```bash
python -m src.cli run-pipeline \
  --input-dir "data/raw/폴더 이름" \
  --output-dir "data/v2" \
  --index-dir "checkpoints/chroma_openai" \
  --force-real
```

단계별(`parse-hwp` -> `chunk-jsonl` -> `embed-jsonl` -> `build-chroma`) 실행도 가능하며, 디버깅/중간 산출물 확인 시 권장합니다.

## 파싱/청킹/샘플링만 수행할 때

임베딩·Chroma 없이:

```bash
python -m src.cli parse-hwp --input "data/raw/문서.hwp" --output "data/v2/..._prechunk.jsonl"
python -m src.cli chunk-jsonl --input "data/v2/..._prechunk.jsonl" --output "data/v2/..._chunks.jsonl"
python -m src.cli sampling --input-dir "data/v2" --pattern "*_chunks.jsonl" --output "data/v2/eval_sample_chunks.jsonl"
```

## 운영 팁

- 긴 단계는 `nohup ... > logs/단계명.log 2>&1 &` 권장
- 디스크 여유 10GB+ 권장
- OCR 상세 규칙은 `docs/ocr.md`, 평가 상세는 `docs/evaluation.md` 참고

