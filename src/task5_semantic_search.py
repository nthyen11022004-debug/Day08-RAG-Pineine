"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4

--------------------------------------------------------------------------
KẾT QUẢ ĐO HyDE — vì sao USE_HYDE = False
--------------------------------------------------------------------------
Đo trên corpus RMIT (408 chunk, paraphrase-multilingual-MiniLM-L12-v2) với
5 câu hỏi liên quan và 5 câu lạc đề. "Khoảng cách" = điểm cosine thấp nhất của
nhóm liên quan trừ điểm cao nhất của nhóm lạc đề; dương nghĩa là tách sạch.

    Cách embed query                        Khoảng cách
    ----------------------------------------------------
    Query thô (đang dùng)                     +0.0381  ✓
    Khuôn HyDE tiếng Anh (bản gốc)            -0.0218
    Khuôn HyDE tiếng Việt                     -0.0931
    Trung bình query + khuôn                  -0.0044
    HyDE thật, LLM sinh đoạn văn              -0.0713

Hai nguyên nhân khác nhau:
  - Khuôn CỐ ĐỊNH giống hệt nhau ở mọi câu hỏi, kéo mọi vector query về một
    điểm chung → mất khả năng phân biệt.
  - HyDE THẬT thì tệ theo cách khác: hỏi "công thức nấu phở" LLM vẫn viết ra
    một đoạn tiếng Việt trôi chảy, mà văn tiếng Việt trôi chảy lại khớp với
    tài liệu tiếng Việt tốt hơn câu hỏi cụt → nâng điểm cho cả câu lạc đề.

Hệ quả nếu bật HyDE: câu vô nghĩa "xyzabc123nonsense" đạt 0.5291 trong khi câu
hỏi thật thấp nhất chỉ 0.5560 → không tồn tại ngưỡng nào để fallback của Task 9
kích hoạt đúng. Tắt HyDE thì câu vô nghĩa rơi xuống 0.2280.

Đo lại bất cứ lúc nào: python -m src.task9_retrieval_pipeline --calibrate
"""

# Bật lại để so sánh A/B khi đánh giá (RAGAS ở bài nhóm). Giữ nguyên code HyDE
# thay vì xoá, vì "đã implement, đo được, chứng minh có hại" là kết quả mạnh hơn
# việc dùng bừa một kỹ thuật vì nó nghe hay.
USE_HYDE = False


def _generate_hypothetical_doc(query: str) -> str:
    """Create a HyDE-style passage that represents an ideal answer to *query*."""
    normalized_query = " ".join(query.split())
    if not normalized_query:
        return ""

    return (
        f"This university services document answers the question: {normalized_query}. "
        "It provides official policies, eligibility requirements, procedures, "
        "fees, deadlines, and contact details relevant to the request."
    )


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if top_k <= 0 or not query.strip():
        return []

    try:
        from .task4_chunking_indexing import get_collection, get_embedding_model
    except ImportError:
        # Supports direct execution with ``python src/task5_semantic_search.py``.
        from task4_chunking_indexing import get_collection, get_embedding_model

    # Xem đo đạc ở docstring đầu file: HyDE làm điểm của câu lạc đề dâng lên
    # ngang câu hỏi thật, khiến ngưỡng fallback của Task 9 không dùng được.
    search_text = _generate_hypothetical_doc(query) if USE_HYDE else query

    model = get_embedding_model()
    query_embedding = model.encode(search_text, show_progress_bar=False).tolist()

    collection = get_collection()
    indexed_count = collection.count()
    if indexed_count == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, indexed_count),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for document, metadata, distance in zip(
        results.get("documents", [[]])[0],
        results.get("metadatas", [[]])[0],
        results.get("distances", [[]])[0],
    ):
        # Task 4 configures ChromaDB with cosine distance.
        score = max(0.0, 1.0 - float(distance))
        output.append(
            {
                "content": document,
                "score": round(score, 4),
                "metadata": metadata or {},
            }
        )

    output.sort(key=lambda item: item["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    # Test
    results = semantic_search("what is the tuition fee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
