"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


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

    hypothetical_doc = _generate_hypothetical_doc(query)
    model = get_embedding_model()
    query_embedding = model.encode(hypothetical_doc, show_progress_bar=False).tolist()

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
