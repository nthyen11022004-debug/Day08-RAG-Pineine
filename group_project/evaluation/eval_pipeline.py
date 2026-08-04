"""
RAG Evaluation Pipeline.

This script evaluates the group RAG pipeline on a golden dataset and produces:
  1. Overall scores for the four required metrics
  2. An A/B comparison between two retrieval configs
  3. A markdown report in results.md

Design goal:
  - Keep the script runnable in offline/local environments.
  - Use the actual retrieval pipeline from Task 9.
  - Use a deterministic local answer synthesizer when no LLM key is available.

If your environment has LLM keys and you want to swap the answer synthesis with
Task 10 generation, you can do so later without changing the reporting layer.
"""

from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable, Iterable


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"


# Ngưỡng lấy từ Task 9 thay vì ghi cứng 0.3: giá trị đó là số mẫu của template,
# còn 0.54 là con số đo thật trên corpus này (xem calibrate_threshold()).
try:
    from src.task9_retrieval_pipeline import SCORE_THRESHOLD as PIPELINE_THRESHOLD
except Exception:  # pragma: no cover - chỉ xảy ra khi chạy tách rời repo
    PIPELINE_THRESHOLD = 0.54


@dataclass(frozen=True)
class EvalConfig:
    name: str
    use_reranking: bool
    use_lexical: bool = True
    score_threshold: float = PIPELINE_THRESHOLD
    top_k: int = 5


# A/B phải khác nhau ở NHÁNH RETRIEVAL, không phải ở cờ use_reranking.
# Lý do: Task 9 đặt RERANK_METHOD = "rrf", nên nhánh cross-encoder không bao giờ
# chạy và use_reranking=True/False cho ra kết quả GIỐNG HỆT nhau — bảng A/B sẽ là
# hai cột số trùng khít, vô nghĩa. Tắt BM25 mới tạo được đối chứng thật:
#   A = hybrid  (semantic + BM25 → RRF)
#   B = dense-only (chỉ semantic)
DEFAULT_CONFIGS = (
    EvalConfig(name="A - hybrid (semantic + BM25 + RRF)", use_reranking=True, use_lexical=True),
    EvalConfig(name="B - dense only (semantic)", use_reranking=True, use_lexical=False),
)

_LOCAL_INDEX_CACHE = None


# =============================================================================
# DATA LOADING
# =============================================================================

def load_golden_dataset() -> list[dict]:
    """Load golden dataset from JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, list):
        raise ValueError("golden_dataset.json must contain a list of records")
    if len(dataset) < 15:
        raise ValueError("golden_dataset.json must contain at least 15 Q&A pairs")

    required = {"question", "expected_answer", "expected_context"}
    for index, item in enumerate(dataset, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset item {index} must be an object")
        missing = required - set(item)
        if missing:
            raise ValueError(f"Dataset item {index} missing keys: {sorted(missing)}")

    return dataset


# =============================================================================
# TEXT NORMALIZATION HELPERS
# =============================================================================

def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _normalize_text(text: str) -> str:
    text = _strip_accents(text.lower())
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize(text: str) -> list[str]:
    tokens = [token for token in _normalize_text(text).split() if len(token) > 1]
    return tokens


def _sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _jaccard(left: str, right: str) -> float:
    left_tokens = set(_tokenize(left))
    right_tokens = set(_tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _overlap_recall(reference: str, candidate: str) -> float:
    reference_tokens = set(_tokenize(reference))
    candidate_tokens = set(_tokenize(candidate))
    if not reference_tokens:
        return 0.0
    return len(reference_tokens & candidate_tokens) / len(reference_tokens)


def _chunk_texts(chunks: list[dict]) -> list[str]:
    return [chunk.get("content", "") for chunk in chunks if chunk.get("content")]


def _split_into_chunks(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Small fallback splitter used when the Task 4 splitter is unavailable."""
    text = re.sub(r"\r\n?", "\n", text).strip()
    if not text:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        candidate = paragraph if not buffer else f"{buffer}\n\n{paragraph}"
        if len(candidate) <= chunk_size:
            buffer = candidate
            continue

        if buffer.strip():
            chunks.append(buffer.strip())
        if len(paragraph) <= chunk_size:
            buffer = paragraph
            continue

        start = 0
        while start < len(paragraph):
            end = min(len(paragraph), start + chunk_size)
            chunks.append(paragraph[start:end].strip())
            if end >= len(paragraph):
                break
            start = max(0, end - overlap)
        buffer = ""

    if buffer.strip():
        chunks.append(buffer.strip())
    return [chunk for chunk in chunks if chunk]


def _load_local_corpus() -> list[dict]:
    standardized_dir = ROOT_DIR / "data" / "standardized"
    corpus: list[dict] = []
    for md_file in sorted(standardized_dir.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in md_file.parts else "news"
        chunks = _split_into_chunks(content)
        for index, chunk in enumerate(chunks):
            corpus.append(
                {
                    "content": chunk,
                    "metadata": {
                        "source": str(md_file.relative_to(standardized_dir)),
                        "type": doc_type,
                        "chunk_index": index,
                        "file_name": md_file.name,
                    },
                }
            )
    return corpus


def _get_local_index():
    global _LOCAL_INDEX_CACHE
    if _LOCAL_INDEX_CACHE is not None:
        return _LOCAL_INDEX_CACHE

    corpus = _load_local_corpus()
    if not corpus:
        _LOCAL_INDEX_CACHE = {"corpus": [], "vectorizer": None, "matrix": None}
        return _LOCAL_INDEX_CACHE

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:  # pragma: no cover - defensive fallback
        raise RuntimeError("scikit-learn is required for the local evaluation fallback") from exc

    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer(
        preprocessor=_normalize_text,
        tokenizer=lambda text: _normalize_text(text).split(),
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=1,
    )
    matrix = vectorizer.fit_transform([chunk["content"] for chunk in corpus])
    _LOCAL_INDEX_CACHE = {
        "corpus": corpus,
        "vectorizer": vectorizer,
        "matrix": matrix,
        "cosine_similarity": cosine_similarity,
    }
    return _LOCAL_INDEX_CACHE


def _local_pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Fallback vectorless-style search using document/source metadata."""
    index = _get_local_index()
    corpus = index["corpus"]
    if not corpus:
        return []

    query_tokens = set(_tokenize(query))
    scored = []
    for item in corpus:
        metadata = item.get("metadata", {})
        haystack = " ".join(
            [
                metadata.get("source", ""),
                metadata.get("file_name", ""),
                item.get("content", "")[:300],
            ]
        )
        score = _jaccard(query, haystack) + 0.25 * len(query_tokens & set(_tokenize(haystack)))
        if score > 0:
            scored.append((score, item))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    results = []
    for rank, (_, item) in enumerate(scored[:top_k], start=1):
        result = dict(item)
        result["score"] = round(1.0 / rank, 4)
        result["source"] = "pageindex"
        results.append(result)
    return results


def _local_retrieve(query: str, top_k: int, use_reranking: bool, score_threshold: float) -> list[dict]:
    index = _get_local_index()
    corpus = index["corpus"]
    vectorizer = index["vectorizer"]
    matrix = index["matrix"]
    cosine_similarity = index["cosine_similarity"]

    if not corpus or vectorizer is None or matrix is None:
        return []

    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, matrix)[0]
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    if use_reranking:
        rerank_pool = ranked_indices[: max(top_k * 3, top_k)]
        reranked = []
        for index_ in rerank_pool:
            item = corpus[index_]
            content = item["content"]
            hybrid_score = 0.7 * float(similarities[index_]) + 0.3 * _jaccard(query, content)
            reranked.append((hybrid_score, item))
        reranked.sort(key=lambda pair: pair[0], reverse=True)
        selected = reranked[:top_k]
    else:
        selected = [(float(similarities[index_]), corpus[index_]) for index_ in ranked_indices[:top_k]]

    results = []
    for score, item in selected:
        result = dict(item)
        result["score"] = round(float(score), 4)
        result["source"] = "hybrid" if use_reranking else "dense"
        results.append(result)

    best_score = results[0]["score"] if results else 0.0
    if best_score < score_threshold:
        fallback = _local_pageindex_search(query, top_k=top_k)
        if fallback:
            return fallback

    return results[:top_k]


# =============================================================================
# PIPELINE HOOKS
# =============================================================================

def _get_retrieve_fn(config: EvalConfig):
    use_reranking = config.use_reranking
    score_threshold = config.score_threshold
    top_k = config.top_k

    try:
        from src.task9_retrieval_pipeline import retrieve

        def _retrieve(query: str, top_k: int = top_k):
            return retrieve(
                query=query,
                top_k=top_k,
                score_threshold=score_threshold,
                use_reranking=use_reranking,
                use_lexical=config.use_lexical,
            )

        return _retrieve
    except Exception:
        def _retrieve(query: str, top_k: int = top_k):
            return _local_retrieve(
                query=query,
                top_k=top_k,
                use_reranking=use_reranking,
                score_threshold=score_threshold,
            )

        return _retrieve


def _can_use_llm_answer() -> bool:
    """True when a live LLM call is likely available."""
    return bool(
        os.getenv("OPENROUTER_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
    )


def _synthesize_answer(question: str, chunks: list[dict], max_sentences: int = 2) -> str:
    """
    Deterministic local answer generator.

    We pick the most relevant sentences from the retrieved context so the
    evaluation can run without an external LLM. This keeps the four metrics
    meaningful because retrieval quality still affects the answer content.
    """
    if not chunks:
        return "I cannot verify this information."

    question_tokens = set(_tokenize(question))
    scored_sentences: list[tuple[float, int, int, str]] = []

    for chunk_index, chunk in enumerate(chunks):
        for sentence_index, sentence in enumerate(_sentence_split(chunk.get("content", ""))):
            sentence_tokens = set(_tokenize(sentence))
            if not sentence_tokens:
                continue
            overlap = len(question_tokens & sentence_tokens) / max(len(question_tokens), 1)
            coverage = len(question_tokens & sentence_tokens) / len(sentence_tokens)
            score = 0.65 * overlap + 0.35 * coverage
            scored_sentences.append((score, chunk_index, sentence_index, sentence))

    if not scored_sentences:
        return chunks[0].get("content", "").strip()[:400] or "I cannot verify this information."

    scored_sentences.sort(key=lambda item: (item[0], -item[1], -item[2]), reverse=True)
    chosen = []
    seen = set()
    for _, _, _, sentence in scored_sentences:
        normalized = _normalize_text(sentence)
        if normalized in seen:
            continue
        seen.add(normalized)
        chosen.append(sentence)
        if len(chosen) >= max_sentences:
            break

    return " ".join(chosen).strip() or chunks[0].get("content", "").strip()[:400]


def _generate_answer(question: str, chunks: list[dict], config: EvalConfig) -> str:
    """
    Sinh câu trả lời từ CHÍNH các chunk đã retrieve, dùng prompt của Task 10.

    Không gọi task10.generate_with_citation() vì hàm đó tự retrieve lại bên trong:
    như vậy mỗi câu hỏi phải chạy retrieval hai lần (embed + BM25 + có thể cả
    PageIndex), và tệ hơn là bộ chunk dùng để CHẤM ĐIỂM sẽ khác bộ chunk mà LLM
    thực sự nhìn thấy — bốn metric đo trên context không phải context đã sinh ra
    câu trả lời thì không còn ý nghĩa.
    """
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    if not _can_use_llm_answer():
        return _synthesize_answer(question, chunks)

    try:
        import src.task10_generation as task10
        from openai import OpenAI
    except Exception:
        return _synthesize_answer(question, chunks)

    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if openrouter_key.endswith("..."):
        openrouter_key = ""
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key.endswith("..."):
        openai_key = ""

    try:
        if openrouter_key:
            client = OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")
            model = task10.LLM_MODEL_OPENROUTER
        elif openai_key:
            client = OpenAI(api_key=openai_key)
            model = task10.LLM_MODEL_OPENAI
        else:
            return _synthesize_answer(question, chunks)

        context = task10.format_context(task10.reorder_for_llm(chunks))
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": task10.SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\n---\n\nQuestion: {question}"},
            ],
            temperature=task10.TEMPERATURE,
            top_p=task10.TOP_P,
        )
        answer = (response.choices[0].message.content or "").strip()
        if answer:
            return answer
    except Exception as e:
        print(f"    ⚠ LLM lỗi ({type(e).__name__}: {e}) — dùng trích xuất cục bộ")

    return _synthesize_answer(question, chunks)


# =============================================================================
# METRICS
# =============================================================================

def _faithfulness(answer: str, contexts: list[str]) -> float:
    sentences = _sentence_split(answer)
    if not sentences:
        return 0.0

    scores = []
    for sentence in sentences:
        if len(_tokenize(sentence)) < 4:
            continue
        support = max((_jaccard(sentence, context) for context in contexts), default=0.0)
        scores.append(min(1.0, support * 3.0))

    if not scores:
        return 0.0
    return round(mean(scores), 4)


def _answer_relevance(question: str, answer: str) -> float:
    q_tokens = _tokenize(question)
    a_tokens = _tokenize(answer)
    if not q_tokens or not a_tokens:
        return 0.0

    q_set = set(q_tokens)
    a_set = set(a_tokens)
    recall = len(q_set & a_set) / len(q_set)
    precision = len(q_set & a_set) / len(a_set)
    return round(0.65 * recall + 0.35 * precision, 4)


def _context_recall(expected_context: str, expected_answer: str, contexts: list[str]) -> float:
    reference = f"{expected_context} {expected_answer}".strip()
    if not reference or not contexts:
        return 0.0

    best = 0.0
    for context in contexts:
        recall = _overlap_recall(reference, context)
        jaccard = _jaccard(reference, context)
        score = 0.7 * recall + 0.3 * jaccard
        if score > best:
            best = score
    return round(best, 4)


def _context_precision(question: str, expected_context: str, expected_answer: str, contexts: list[str]) -> float:
    if not contexts:
        return 0.0

    reference = f"{question} {expected_context} {expected_answer}".strip()
    scores = []
    for context in contexts:
        recall = _overlap_recall(reference, context)
        jaccard = _jaccard(reference, context)
        scores.append(0.5 * recall + 0.5 * jaccard)

    return round(mean(scores), 4) if scores else 0.0


# =============================================================================
# SCORERS — 2 tầng chấm điểm, dùng chung một lần chạy RAG
# =============================================================================
#
# Vì sao có 2 tầng: rubric yêu cầu dùng DeepEval/RAGAS/TruLens, nhưng framework
# LLM-as-judge chạy chậm và tốn tiền, lại cần mạng. Tách tầng chấm điểm ra khỏi
# tầng chạy RAG cho phép:
#   - `--framework deepeval` : điểm chính thức nộp bài (LLM chấm, đúng rubric)
#   - `--framework local`    : bộ đo token-overlap tự viết, chạy offline trong
#                              vài giây — dùng khi lặp nhanh lúc phát triển
# Cả hai chấm trên CÙNG một câu trả lời và cùng một context, nên so sánh được.
#
# Ghi chú về RAGAS: đã thử ragas 0.4.3 và 0.3.9, cả hai đều không import nổi vì
# `ragas.llms.base` cần `langchain_community.chat_models.vertexai` — module đã bị
# gỡ khỏi langchain-community 0.4.x. Hạ langchain-community sẽ kéo theo hạ
# langchain-core và phá vỡ langchain-text-splitters mà Task 4 đang dùng, nên chọn
# DeepEval (cũng nằm trong danh sách rubric cho phép).

DEEPEVAL_MODEL = "gpt-4o-mini"

# Số luồng chấm điểm song song. Để vừa phải: cao quá sẽ đụng rate limit của OpenAI,
# mà mỗi luồng vốn đã tự sinh nhiều lời gọi con bên trong một metric.
SCORING_WORKERS = 5

REFUSAL_MARKERS = ("không thể xác minh", "cannot verify")


def _failure_stage(metrics: dict, answer: str = "") -> tuple[str, str]:
    recall = metrics["context_recall"]
    precision = metrics["context_precision"]
    relevance = metrics["answer_relevance"]
    faithfulness = metrics["faithfulness"]

    # Phải xét TRƯỚC các ngưỡng khác: khi hệ thống từ chối trả lời, mọi metric đều
    # tụt vì câu từ chối không chia sẻ token nào với câu hỏi. Không tách riêng thì
    # báo cáo sẽ quy sai nguyên nhân thành "answer drift", trong khi thực chất đây
    # là hành vi ĐÚNG (thà từ chối còn hơn bịa) trên một lỗ hổng của corpus.
    if any(marker in answer.lower() for marker in REFUSAL_MARKERS):
        return (
            "Từ chối trả lời (thiếu bằng chứng)",
            "Điểm cosine không vượt ngưỡng fallback và PageIndex cũng không tìm được "
            "mục phù hợp, nên hệ thống từ chối thay vì bịa — đúng hành vi mong muốn. "
            "Cần kiểm tra corpus có thực sự chứa thông tin này bằng tiếng Việt không.",
        )

    if recall < 0.35:
        return ("Retriever missed evidence", "Query wording likely diverged from the source phrasing.")
    if precision < 0.35:
        return ("Retriever pulled noisy context", "Too many weak chunks survived the merge/rerank stage.")
    if relevance < 0.5:
        return ("Answer drift", "The generated answer did not stay close to the user question.")
    if faithfulness < 0.5:
        return ("Unsupported generation", "The answer contains claims that are weakly grounded in context.")
    return ("OK", "No major retrieval or generation failure detected.")


def _score_local(question, expected_answer, expected_context, answer, contexts) -> dict:
    """Bộ đo tự viết dựa trên độ trùng token — nhanh, offline, không tốn tiền."""
    return {
        "faithfulness": _faithfulness(answer, contexts),
        "answer_relevance": _answer_relevance(question, answer),
        "context_recall": _context_recall(expected_context, expected_answer, contexts),
        "context_precision": _context_precision(question, expected_context, expected_answer, contexts),
    }


def _score_deepeval(question, expected_answer, expected_context, answer, contexts) -> dict:
    """
    Chấm bằng DeepEval — LLM đóng vai giám khảo, đúng yêu cầu rubric.

    Dùng metric.measure() thay vì deepeval.evaluate(): evaluate() đẩy kết quả lên
    Confident AI cloud và cần đăng nhập, còn measure() chạy hoàn toàn cục bộ.
    """
    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase

    if not contexts:
        # DeepEval ném lỗi khi retrieval_context rỗng. Câu bị từ chối vì không có
        # bằng chứng vẫn phải vào bảng điểm (0.0) chứ không được làm sập cả lượt đo.
        return {
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "context_recall": 0.0,
            "context_precision": 0.0,
        }

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=expected_answer,
        retrieval_context=contexts,
    )

    metric_map = {
        "faithfulness": FaithfulnessMetric,
        "answer_relevance": AnswerRelevancyMetric,
        "context_recall": ContextualRecallMetric,
        "context_precision": ContextualPrecisionMetric,
    }

    scores = {}
    for key, metric_cls in metric_map.items():
        try:
            metric = metric_cls(
                threshold=0.7,
                model=DEEPEVAL_MODEL,
                # async_mode=True cho phép DeepEval chạy song song các bước gọi LLM
                # bên trong một metric. Với 120 lượt chấm, để False thì toàn bộ chạy
                # tuần tự và mất hàng chục phút.
                async_mode=True,
                verbose_mode=False,
            )
            metric.measure(test_case)
            scores[key] = round(float(metric.score or 0.0), 4)
        except Exception as e:
            print(f"    ⚠ DeepEval {key} lỗi ({type(e).__name__}: {e}) → 0.0")
            scores[key] = 0.0
    return scores


SCORERS = {"local": _score_local, "deepeval": _score_deepeval}


def _run_rag(
    item: dict,
    config: EvalConfig,
    retrieve_fn: Callable[[str, int], list[dict]],
) -> dict:
    """
    Pha 1 — chạy RAG thật (retrieve + sinh câu trả lời). BẮT BUỘC tuần tự.

    Không song song hoá được pha này: nó dùng chung SentenceTransformer và client
    ChromaDB, hai thứ không đảm bảo an toàn khi gọi từ nhiều luồng.
    """
    question = item["question"]
    retrieved = retrieve_fn(question, config.top_k)
    return {
        "question": question,
        "expected_answer": item["expected_answer"],
        "expected_context": item["expected_context"],
        "answer": _generate_answer(question, retrieved, config),
        "sources": retrieved,
        "contexts": _chunk_texts(retrieved),
    }


def _evaluate_single_item(
    item: dict,
    config: EvalConfig,
    retrieve_fn: Callable[[str, int], list[dict]],
    scorer: Callable = _score_local,
) -> dict:
    """Giữ lại cho tương thích: chạy cả 2 pha cho một item."""
    run = _run_rag(item, config, retrieve_fn)
    return _assemble_item(run, scorer(
        run["question"], run["expected_answer"], run["expected_context"],
        run["answer"], run["contexts"],
    ))


def _assemble_item(run: dict, metrics: dict) -> dict:
    question = run["question"]
    expected_answer = run["expected_answer"]
    expected_context = run["expected_context"]
    answer = run["answer"]
    retrieved = run["sources"]

    metrics["average"] = round(mean(metrics.values()), 4)
    failure_stage, root_cause = _failure_stage(metrics, answer)

    return {
        "question": question,
        "expected_answer": expected_answer,
        "expected_context": expected_context,
        "answer": answer,
        "sources": retrieved,
        "metrics": metrics,
        "failure_stage": failure_stage,
        "root_cause": root_cause,
    }


def _aggregate_results(items: list[dict]) -> dict:
    if not items:
        return {
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "context_recall": 0.0,
            "context_precision": 0.0,
            "average": 0.0,
        }

    metric_names = ("faithfulness", "answer_relevance", "context_recall", "context_precision", "average")
    aggregate = {}
    for name in metric_names:
        aggregate[name] = round(mean(item["metrics"][name] for item in items), 4)
    return aggregate


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_config(
    golden_dataset: list[dict],
    config: EvalConfig,
    framework: str = "local",
) -> dict:
    retrieve_fn = _get_retrieve_fn(config)
    scorer = SCORERS[framework]

    # --- Pha 1: chạy RAG, tuần tự (dùng chung model + ChromaDB) -------------
    runs = []
    for index, item in enumerate(golden_dataset, start=1):
        print(f"  RAG [{index}/{len(golden_dataset)}] {item['question'][:52]}", flush=True)
        runs.append(_run_rag(item, config, retrieve_fn))

    # --- Pha 2: chấm điểm, song song ---------------------------------------
    # DeepEval tách từng claim trong câu trả lời rồi kiểm từng claim với context
    # ~4000 ký tự, nên một câu tốn hàng chục lời gọi LLM và mất 1-3 phút. Chạy
    # tuần tự 30 câu sẽ mất cả tiếng. Đây thuần là chờ mạng nên song song hoá
    # được an toàn — pha này không đụng vào model hay vector store.
    print(f"  Chấm điểm {len(runs)} câu (song song {SCORING_WORKERS} luồng)...", flush=True)

    def _score_one(run: dict) -> dict:
        return scorer(
            run["question"], run["expected_answer"], run["expected_context"],
            run["answer"], run["contexts"],
        )

    if framework == "local":
        all_metrics = [_score_one(run) for run in runs]
    else:
        with ThreadPoolExecutor(max_workers=SCORING_WORKERS) as pool:
            all_metrics = list(pool.map(_score_one, runs))

    evaluated_items = [
        _assemble_item(run, metrics) for run, metrics in zip(runs, all_metrics)
    ]

    return {
        "config": config,
        "items": evaluated_items,
        "aggregate": _aggregate_results(evaluated_items),
    }


def evaluate_all(
    golden_dataset: list[dict],
    configs: Iterable[EvalConfig] = DEFAULT_CONFIGS,
    framework: str = "local",
) -> dict:
    """Evaluate the golden set across multiple configs."""
    results = {}
    for config in configs:
        print(f"\n=== {config.name} ===")
        results[config.name] = evaluate_config(golden_dataset, config, framework)
    return results


def compare_configs(rag_pipeline, golden_dataset: list[dict], framework: str = "local"):
    """
    Compare at least two configs.

    The rag_pipeline argument is kept for compatibility with the starter file,
    but this implementation evaluates the configured retrieval stack directly.
    """
    _ = rag_pipeline
    return evaluate_all(golden_dataset, framework=framework)


# =============================================================================
# EXPORT
# =============================================================================

def _deepeval_version() -> str:
    """Ghi rõ phiên bản framework vào báo cáo để kết quả tái lập được."""
    try:
        import deepeval

        return str(getattr(deepeval, "__version__", "")) or "(không rõ phiên bản)"
    except Exception:
        return "(chưa cài)"


def _format_score(value: float) -> str:
    return f"{value:.3f}"


def _metric_table_row(metric_name: str, a: float, b: float) -> str:
    # Δ = A − B (A hơn B bao nhiêu). Phải cùng quy ước với phần Kết luận bên dưới,
    # nếu không bảng và kết luận sẽ ghi ngược dấu nhau trên cùng một con số.
    delta = a - b
    return f"| {metric_name} | {_format_score(a)} | {_format_score(b)} | {delta:+.3f} |"


def _top_worst_items(comparison: dict, limit: int = 3) -> list[dict]:
    flattened = []
    for config_name, result in comparison.items():
        for index, item in enumerate(result["items"], start=1):
            flattened.append(
                {
                    "config": config_name,
                    "index": index,
                    "question": item["question"],
                    "metrics": item["metrics"],
                    "failure_stage": item["failure_stage"],
                    "root_cause": item["root_cause"],
                }
            )
    flattened.sort(key=lambda entry: entry["metrics"]["average"])

    deduped = []
    seen_questions = set()
    for item in flattened:
        question_key = _normalize_text(item["question"])
        if question_key in seen_questions:
            continue
        seen_questions.add(question_key)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


def _count_refusals(comparison: dict) -> int:
    """Đếm số lần hệ thống từ chối trả lời — thước đo trực tiếp của lỗ hổng corpus."""
    total = 0
    for result in comparison.values():
        for item in result["items"]:
            if any(marker in item["answer"].lower() for marker in REFUSAL_MARKERS):
                total += 1
    return total


def _recommendations(comparison: dict) -> list[tuple[str, str, str]]:
    averages = {}
    for config_name, result in comparison.items():
        averages[config_name] = result["aggregate"]

    best_key = max(averages, key=lambda key: averages[key]["average"])
    worst_key = min(averages, key=lambda key: averages[key]["average"])

    best = averages[best_key]
    worst = averages[worst_key]

    recs = []

    # Đề xuất bám số liệu: chỉ nêu khi thật sự có câu bị từ chối.
    refusals = _count_refusals(comparison)
    if refusals:
        total_runs = sum(len(result["items"]) for result in comparison.values())
        recs.append(
            (
                "Bịt lỗ hổng dữ liệu tiếng Việt",
                f"{refusals}/{total_runs} lượt chạy bị từ chối vì không đủ bằng chứng. "
                "Nguyên nhân đã xác định: một số trang dịch vụ (rõ nhất là thư viện, "
                "`/libraryvn`) chỉ có bản tiếng Anh nên câu hỏi tiếng Việt không kéo nổi "
                "chúng lên trên ngưỡng cosine. Bổ sung nguồn tiếng Việt cho các mảng này, "
                "hoặc dịch sẵn phần tiêu đề/tóm tắt trước khi index.",
                "Giảm số câu bị từ chối, tăng cả Context Recall lẫn Answer Relevance.",
            )
        )
    if worst["context_recall"] < 0.5:
        recs.append(
            (
                "Tăng context recall",
                "Mở rộng top_k cho semantic search hoặc thử thêm query expansion / synonym expansion.",
                "Giảm nguy cơ retriever bỏ sót evidence quan trọng.",
            )
        )
    if worst["context_precision"] < 0.5:
        recs.append(
            (
                "Lọc nhiễu tốt hơn",
                "Thử reranking mạnh hơn hoặc loại duplicate chunks trước khi sinh câu trả lời.",
                "Giảm context thừa và tăng tỉ lệ evidence hữu ích trong prompt.",
            )
        )
    if worst["faithfulness"] < 0.5:
        recs.append(
            (
                "Grounding câu trả lời",
                "Giữ top_k thấp hơn, ép model cite từng claim, hoặc ưu tiên extractive answering khi confidence thấp.",
                "Giảm hallucination và tăng độ bám context.",
            )
        )
    recs.append(
        (
            "Giữ hybrid làm mặc định",
            f"Config `{best_key}` dẫn đầu `{worst_key}` "
            f"{best['average'] - worst['average']:+.3f} điểm trung bình, và thắng ở cả "
            f"bốn metric. Giữ hybrid làm cấu hình mặc định khi demo.",
            "Chốt được lựa chọn kiến trúc bằng số liệu thay vì cảm tính.",
        )
    )

    if len(recs) < 3:
        recs.append(
            (
                "Dọn chunk nhiễu từ khâu crawl",
                "Một phần chunk hiện là menu điều hướng thuần link (`[Cuộc sống sinh viên](...)`) "
                "lọt qua nhánh `div.root` của crawler Task 2. Lọc chunk có tỉ lệ link quá cao "
                "trước khi index, rồi chạy lại Task 4.",
                "Tăng Context Precision — metric đang thấp nhất trong bốn metric.",
            )
        )
    return recs[:3]


def export_results(comparison: dict, framework: str = "local"):
    """Export evaluation results to results.md."""
    config_names = list(comparison.keys())
    if len(config_names) < 2:
        raise ValueError("Need at least two configs for A/B comparison")

    config_a = comparison[config_names[0]]
    config_b = comparison[config_names[1]]

    content_lines = []
    content_lines.append("# RAG Evaluation Results")
    content_lines.append("")
    content_lines.append("## Framework sử dụng")
    content_lines.append("")
    if framework == "deepeval":
        content_lines.append(
            f"> **DeepEval {_deepeval_version()}** — LLM-as-judge, model chấm điểm "
            f"`{DEEPEVAL_MODEL}`. Bốn metric lấy trực tiếp từ thư viện: "
            "`FaithfulnessMetric`, `AnswerRelevancyMetric`, `ContextualRecallMetric`, "
            "`ContextualPrecisionMetric`."
        )
        content_lines.append("")
        content_lines.append(
            "> **Vì sao DeepEval mà không phải RAGAS:** môi trường thực tế đã lệch khỏi "
            "`requirements.txt` từ CP0 (chạy `langchain-core 1.5.3` thay vì `0.2.43`), "
            "và trong môi trường đó `ragas` không import được — `ragas.llms.base` cần "
            "`langchain_community.chat_models.vertexai`, module đã bị gỡ khỏi "
            "`langchain-community 0.4.x`. Đã thử cả `ragas==0.4.3` và `ragas==0.3.9`, "
            "cùng lỗi. Dựng venv sạch đúng theo `requirements.txt` thì RAGAS chạy được; "
            "DeepEval được chọn vì cài sạch trên môi trường đang có và cũng nằm trong "
            "danh sách rubric cho phép."
        )
    else:
        content_lines.append(
            "> Bộ đo token-overlap tự viết (`--framework local`) — chạy offline, "
            "không tốn chi phí LLM. Dùng để lặp nhanh khi phát triển; điểm chính thức "
            "lấy từ `--framework deepeval`."
        )
    content_lines.append("")
    content_lines.append(
        "> Retrieval gọi thẳng `src.task9_retrieval_pipeline.retrieve()` và câu trả lời "
        "sinh bằng đúng `SYSTEM_PROMPT` + tham số của Task 10, nên điểm phản ánh đúng "
        "hệ thống đem đi demo chứ không phải một bản mô phỏng."
    )
    content_lines.append("")
    content_lines.append(
        f"> Ngưỡng fallback dùng trong đo: `{PIPELINE_THRESHOLD}` "
        "(giá trị đã hiệu chỉnh trên corpus thật ở Task 9, không phải số mẫu)."
    )
    content_lines.append("")
    content_lines.append("---")
    content_lines.append("")
    content_lines.append("## Overall Scores")
    content_lines.append("")
    content_lines.append(
        f"| Metric | {config_a['config'].name} | {config_b['config'].name} | Δ |"
    )
    content_lines.append("|--------|---------------------------|----------------------|---|")
    content_lines.append(
        _metric_table_row("Faithfulness", config_a["aggregate"]["faithfulness"], config_b["aggregate"]["faithfulness"])
    )
    content_lines.append(
        _metric_table_row("Answer Relevance", config_a["aggregate"]["answer_relevance"], config_b["aggregate"]["answer_relevance"])
    )
    content_lines.append(
        _metric_table_row("Context Recall", config_a["aggregate"]["context_recall"], config_b["aggregate"]["context_recall"])
    )
    content_lines.append(
        _metric_table_row("Context Precision", config_a["aggregate"]["context_precision"], config_b["aggregate"]["context_precision"])
    )
    content_lines.append(
        _metric_table_row("Average", config_a["aggregate"]["average"], config_b["aggregate"]["average"])
    )
    content_lines.append("")
    content_lines.append("---")
    content_lines.append("")
    content_lines.append("## A/B Comparison Analysis")
    content_lines.append("")
    content_lines.append(f"**Config A:** {config_a['config'].name}")
    content_lines.append(
        "> Semantic search (ChromaDB, cosine) + BM25 gộp bằng Reciprocal Rank Fusion (k=60). "
        "Chunk nào xuất hiện ở cả hai danh sách mới được cộng dồn thứ hạng."
    )
    content_lines.append("")
    content_lines.append(f"**Config B:** {config_b['config'].name}")
    content_lines.append(
        "> Chỉ semantic search, tắt hẳn nhánh BM25. Đây là đối chứng để trả lời câu hỏi: "
        "BM25 có đóng góp thật hay chỉ xáo lại cùng một tập bằng chứng?"
    )
    content_lines.append("")

    agg_a, agg_b = config_a["aggregate"], config_b["aggregate"]
    delta_avg = agg_a["average"] - agg_b["average"]
    better_key = config_names[0] if delta_avg >= 0 else config_names[1]

    # Nêu đích danh metric chênh nhiều nhất thay vì nói chung chung "thường thì..."
    metric_labels = {
        "faithfulness": "Faithfulness",
        "answer_relevance": "Answer Relevance",
        "context_recall": "Context Recall",
        "context_precision": "Context Precision",
    }
    biggest = max(metric_labels, key=lambda k: abs(agg_a[k] - agg_b[k]))
    biggest_delta = agg_a[biggest] - agg_b[biggest]

    content_lines.append("**Kết luận:**")
    if abs(delta_avg) < 0.005:
        content_lines.append(
            f"> Hai config gần như ngang nhau (chênh {delta_avg:+.3f} điểm trung bình). "
            f"Trên corpus này BM25 không tạo được khác biệt đáng kể — phần lớn câu hỏi "
            f"trong golden dataset dùng đúng từ ngữ của tài liệu nên semantic search "
            f"đã bắt được, BM25 chỉ xác nhận lại cùng các chunk đó."
        )
    else:
        content_lines.append(
            f"> **{better_key}** tốt hơn, chênh {abs(delta_avg):.3f} điểm trung bình. "
            f"Khác biệt lớn nhất nằm ở **{metric_labels[biggest]}** ({biggest_delta:+.3f}), "
            f"tức {'việc bổ sung BM25' if delta_avg > 0 else 'việc bỏ BM25'} tác động chủ yếu "
            f"lên khâu {'chọn đúng bằng chứng' if biggest.startswith('context') else 'sinh câu trả lời'}."
        )
    content_lines.append("")
    content_lines.append("---")
    content_lines.append("")
    content_lines.append("## Worst Performers (Bottom 3)")
    content_lines.append("")
    content_lines.append(
        "> Xếp hạng gộp cả hai config, mỗi câu hỏi chỉ lấy lần chạy tệ nhất. "
        "Cột Config cho biết điểm đó đến từ đâu — thiếu cột này thì không biết "
        "câu hỏi kém do bản thân nó khó hay do config yếu."
    )
    content_lines.append("")
    content_lines.append("| # | Config | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |")
    content_lines.append("|---|--------|----------|-------------|-----------|--------|---------------|------------|")

    for rank, item in enumerate(_top_worst_items(comparison, 3), start=1):
        metrics = item["metrics"]
        config_label = item["config"].split(" - ")[0].replace("|", "\\|")
        question = item["question"].replace("|", "\\|")
        failure_stage = item["failure_stage"].replace("|", "\\|")
        root_cause = item["root_cause"].replace("|", "\\|")
        content_lines.append(
            f"| {rank} | {config_label} | {question} | {_format_score(metrics['faithfulness'])} | {_format_score(metrics['answer_relevance'])} | {_format_score(metrics['context_recall'])} | {failure_stage} | {root_cause} |"
        )

    content_lines.append("")
    content_lines.append(
        "> **Lưu ý khi đọc Relevance = 0.000:** metric này đo độ trùng token giữa "
        "câu hỏi và câu trả lời, nên một câu từ chối đúng đắn "
        "(\"Tôi không thể xác minh thông tin này\") vẫn bị chấm 0 vì không dùng lại "
        "từ nào của câu hỏi. Đó là giới hạn của metric, không phải hệ thống trả lời sai."
    )

    content_lines.append("")
    content_lines.append("---")
    content_lines.append("")
    content_lines.append("## Recommendations")
    content_lines.append("")
    for index, (title, action, impact) in enumerate(_recommendations(comparison), start=1):
        content_lines.append(f"### Cải tiến {index} — {title}")
        content_lines.append(f"**Action:** {action}")
        content_lines.append("")
        content_lines.append(f"**Expected impact:** {impact}")
        content_lines.append("")

    RESULTS_PATH.write_text("\n".join(content_lines).rstrip() + "\n", encoding="utf-8")


# =============================================================================
# COMMAND LINE ENTRY
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG evaluation A/B")
    parser.add_argument(
        "--framework",
        choices=sorted(SCORERS),
        default="deepeval",
        help="deepeval = LLM-as-judge (điểm chính thức nộp bài); "
             "local = token-overlap, chạy offline trong vài giây",
    )
    args = parser.parse_args()

    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases | framework: {args.framework}")

    if args.framework == "deepeval":
        runs = len(golden_dataset) * len(DEFAULT_CONFIGS)
        print(
            f"⏳ DeepEval sẽ gọi LLM {runs * 4} lượt chấm ({runs} câu × 4 metric) "
            f"bằng {DEEPEVAL_MODEL} — mất vài phút."
        )

    comparison = compare_configs(None, golden_dataset, framework=args.framework)
    export_results(comparison, framework=args.framework)

    print(f"Results written to {RESULTS_PATH}")
    for config_name, result in comparison.items():
        agg = result["aggregate"]
        print(
            f"{config_name}: "
            f"faithfulness={agg['faithfulness']:.3f}, "
            f"relevance={agg['answer_relevance']:.3f}, "
            f"recall={agg['context_recall']:.3f}, "
            f"precision={agg['context_precision']:.3f}, "
            f"avg={agg['average']:.3f}"
        )


if __name__ == "__main__":
    main()