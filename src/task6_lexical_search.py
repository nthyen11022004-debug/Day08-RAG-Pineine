"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25
    # (tuỳ chọn, để tokenize tiếng Việt tốt hơn)
    pip install underthesea

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import json
import re
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

# Thử import underthesea để tokenize tiếng Việt tốt hơn; nếu không có thì fallback.
try:
    from underthesea import word_tokenize as _vi_word_tokenize
    _HAS_UNDERTHESEA = True
except ImportError:
    _HAS_UNDERTHESEA = False


DATA_DIR = Path("data/standardized")

# TODO: Load corpus từ data/standardized/ hoặc từ vector store
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}

# Cache cho BM25 index để không phải build lại mỗi lần gọi lexical_search
_BM25_INDEX: Optional[BM25Okapi] = None
_TOKENIZED_CORPUS: Optional[list[list[str]]] = None


def _tokenize(text: str) -> list[str]:
    """
    Tokenize văn bản. Ưu tiên underthesea cho tiếng Việt (xử lý từ ghép
    kiểu "học_phí"), fallback về split() đơn giản nếu không cài underthesea.
    """
    text = text.lower().strip()
    # Bỏ bớt ký tự đặc biệt gây nhiễu, giữ lại chữ/số/khoảng trắng và gạch dưới
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    if _HAS_UNDERTHESEA:
        # underthesea trả về các từ ghép nối bằng "_", ví dụ "học_phí"
        tokens = _vi_word_tokenize(text)
    else:
        tokens = text.split()

    return [t for t in tokens if t.strip()]


def load_corpus(data_dir: Path = DATA_DIR) -> list[dict]:
    """
    Load corpus từ các file JSON/JSONL trong data/standardized/.

    Hỗ trợ 2 định dạng:
      1. File .json chứa list các object {'content': ..., 'metadata': {...}}
      2. File .jsonl với mỗi dòng là 1 object {'content': ..., 'metadata': {...}}

    Nếu document thiếu 'metadata', sẽ tự gán {'source': <tên file>}.

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    corpus: list[dict] = []

    if not data_dir.exists():
        print(f"[WARN] Thư mục {data_dir} không tồn tại. CORPUS sẽ rỗng.")
        return corpus

    for file_path in sorted(data_dir.glob("**/*")):
        if file_path.suffix == ".json":
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                print(f"[WARN] Bỏ qua {file_path}: lỗi parse JSON ({e})")
                continue

            items = data if isinstance(data, list) else [data]
            for item in items:
                content = item.get("content") or item.get("text")
                if not content:
                    continue
                corpus.append({
                    "content": content,
                    "metadata": item.get("metadata", {"source": file_path.name}),
                })

        elif file_path.suffix == ".jsonl":
            with file_path.open(encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"[WARN] Bỏ qua {file_path}:{line_no} ({e})")
                        continue
                    content = item.get("content") or item.get("text")
                    if not content:
                        continue
                    corpus.append({
                        "content": content,
                        "metadata": item.get("metadata", {"source": file_path.name}),
                    })

    return corpus


def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}

    Returns:
        BM25Okapi index đã build.
    """
    global _TOKENIZED_CORPUS

    if not corpus:
        raise ValueError("Corpus rỗng — không thể build BM25 index.")

    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    _TOKENIZED_CORPUS = tokenized_corpus

    # k1=1.5 (term saturation), b=0.75 (length normalization) — theo đề bài
    bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)
    return bm25


def _ensure_index() -> BM25Okapi:
    """
    Đảm bảo CORPUS đã được load và BM25 index đã được build (cache lại,
    tránh rebuild mỗi lần gọi lexical_search).
    """
    global CORPUS, _BM25_INDEX

    if not CORPUS:
        CORPUS = load_corpus()

    if not CORPUS:
        raise RuntimeError(
            "CORPUS rỗng. Hãy đảm bảo data/standardized/ có dữ liệu, "
            "hoặc gán CORPUS thủ công trước khi search."
        )

    if _BM25_INDEX is None:
        _BM25_INDEX = build_bm25_index(CORPUS)

    return _BM25_INDEX


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    if not query or not query.strip():
        return []

    bm25 = _ensure_index()

    tokenized_query = _tokenize(query)
    if not tokenized_query:
        return []

    scores = bm25.get_scores(tokenized_query)

    # Lấy top_k index theo score giảm dần, không cần numpy đầy đủ argsort
    # nếu top_k << len(corpus), nhưng argsort đơn giản là đủ nhanh cho demo.
    ranked_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )[:top_k]

    results = []
    for idx in ranked_indices:
        if scores[idx] > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(scores[idx]),
                "metadata": CORPUS[idx]["metadata"],
            })

    return results


def reset_index() -> None:
    """Xoá cache index (dùng khi CORPUS thay đổi, ví dụ sau khi ingest thêm data)."""
    global _BM25_INDEX, _TOKENIZED_CORPUS
    _BM25_INDEX = None
    _TOKENIZED_CORPUS = None


if __name__ == "__main__":
    # Test
    results = lexical_search("tuition fee payment methods", top_k=5)
    if not results:
        print("Không có kết quả (kiểm tra data/standardized/ có dữ liệu chưa).")
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")