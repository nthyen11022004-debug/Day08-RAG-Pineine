"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results

⚠️ BẪY THƯỜNG GẶP — đọc kỹ trước khi code:
    Nếu bạn dùng điểm RRF đã fuse (Task 7) để so với score_threshold, bạn sẽ gặp bug
    thật: RRF max score luôn ≈ 1/(k+1) ≈ 0.0164 (k=60) BẤT KỂ nội dung có liên quan
    hay không. Nếu đặt threshold thấp (như 0.005) để "hợp" với thang điểm RRF, thực
    chất KHÔNG câu hỏi nào đủ thấp để trigger fallback nữa — kể cả query hoàn toàn vô
    nghĩa vẫn trả về kết quả "hybrid" (rác) thay vì fallback đúng như thiết kế.

    Cách sửa đúng: giữ điểm cosine similarity GỐC của semantic_search (trước khi qua
    RRF) làm căn cứ quyết định fallback, tách biệt khỏi điểm RRF dùng để sắp xếp kết
    quả cuối cùng. Calibrate threshold bằng cách tự đo: chạy vài câu hỏi chắc chắn
    liên quan và vài câu chắc chắn lạc đề/rác qua semantic_search, xem khoảng cách
    điểm số giữa hai nhóm rồi chọn ngưỡng nằm giữa.

    → Đã làm đúng như vậy: hàm calibrate_threshold() ở cuối file đo thật trên corpus,
      chạy `python -m src.task9_retrieval_pipeline --calibrate` để đo lại.
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

# ĐO THẬT bằng calibrate_threshold() trên corpus RMIT (408 chunk) + embedding
# paraphrase-multilingual-MiniLM-L12-v2, KHÔNG phải copy số mẫu của template:
#     nhóm liên quan — điểm thấp nhất : 0.5580
#     nhóm lạc đề    — điểm cao nhất  : 0.5199
#     → chọn điểm giữa: 0.54
# Biên khá hẹp (0.038): "asdkjh qwe zxc" đạt tới 0.5199, sát ngưỡng. Đổi corpus
# hoặc đổi embedding model là phải đo lại — chạy:
#     python -m src.task9_retrieval_pipeline --calibrate
SCORE_THRESHOLD = 0.54   # Nếu best score (cosine gốc) < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"  # "cross_encoder" | "mmr" | "rrf"

# Lấy rộng hơn top_k ở tầng retrieval rồi mới cắt: RRF chỉ phát huy tác dụng khi
# hai ranker có đủ ứng viên chồng lấn nhau để cộng dồn thứ hạng.
CANDIDATE_MULTIPLIER = 2


def _safe_search(search_fn, query: str, top_k: int, label: str) -> list[dict]:
    """
    Gọi một retriever, lỗi thì trả rỗng thay vì làm sập cả pipeline.

    Lý do: hai nhánh dense/sparse độc lập nhau. Nếu chroma_db chưa được build thì
    semantic_search hỏng, nhưng BM25 vẫn dùng được — và PageIndex vẫn còn đó làm
    fallback. Để nguyên exception bay lên thì chatbot chết hẳn dù vẫn còn 2 nhánh
    hoạt động tốt.
    """
    try:
        return search_fn(query, top_k=top_k) or []
    except Exception as e:
        print(f"  ⚠ {label} lỗi ({type(e).__name__}: {e}) — bỏ qua nhánh này")
        return []


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
    use_lexical: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → dense_results (giữ điểm cosine gốc)
          ├→ Lexical Search  → sparse_results
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If dense_results[0]["score"] < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm cosine gốc tối thiểu (KHÔNG phải điểm RRF)
        use_reranking: Có áp dụng cross-encoder rerank hay không. Lưu ý: khi
            RERANK_METHOD = "rrf" thì cờ này KHÔNG đổi kết quả, vì RRF đã chạy
            ở bước merge rồi — muốn A/B thật sự khác nhau thì dùng use_lexical.
        use_lexical: Bật/tắt nhánh BM25. Tắt = dense-only, dùng cho A/B testing
            ở bài nhóm (hybrid vs dense-only).

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    if not isinstance(query, str) or not query.strip():
        return []
    if top_k <= 0:
        return []

    candidate_k = top_k * CANDIDATE_MULTIPLIER

    # --- Bước 1: hai nhánh retrieval độc lập -------------------------------
    dense_results = _safe_search(semantic_search, query, candidate_k, "semantic_search")
    sparse_results = (
        _safe_search(lexical_search, query, candidate_k, "lexical_search")
        if use_lexical
        else []
    )

    # --- Bước 2: chốt căn cứ fallback TRƯỚC khi RRF ghi đè score ------------
    # Phải đọc điểm cosine ở đây, vì rerank_rrf() copy dict rồi thay 'score' bằng
    # điểm RRF (~0.016). Đọc sau khi fuse là rơi đúng vào cái bẫy ở docstring.
    best_dense_score = dense_results[0]["score"] if dense_results else 0.0

    # --- Bước 3: merge bằng RRF --------------------------------------------
    merged = rerank_rrf([dense_results, sparse_results], top_k=candidate_k)
    for item in merged:
        item["source"] = "hybrid"
        # Giữ lại điểm cosine gốc để UI/eval giải thích được vì sao chunk được chọn;
        # 'score' sau bước này là điểm RRF, chỉ có ý nghĩa thứ hạng.
        item["dense_score"] = best_dense_score

    # --- Bước 4: rerank -----------------------------------------------------
    if use_reranking and merged and RERANK_METHOD == "cross_encoder":
        # Chỉ cross-encoder mới thực sự chấm lại độ liên quan.
        # KHÔNG gọi rerank(..., method="rrf") ở đây: hàm đó chạy rerank_rrf trên
        # MỘT list duy nhất, tức chỉ gán lại score theo đúng thứ tự sẵn có —
        # không đổi gì ngoài việc ghi đè điểm, thuần tốn công.
        try:
            final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
            for item in final_results:
                item.setdefault("source", "hybrid")
        except Exception as e:
            print(f"  ⚠ rerank lỗi ({type(e).__name__}: {e}) — giữ thứ tự RRF")
            final_results = merged[:top_k]
    else:
        final_results = merged[:top_k]

    # --- Bước 5: fallback khi hybrid không đủ tin cậy -----------------------
    if best_dense_score < score_threshold:
        print(
            f"  ⚠ Semantic best score ({best_dense_score:.3f}) "
            f"< threshold ({score_threshold}) → fallback PageIndex"
        )
        fallback = _safe_search(pageindex_search, query, top_k, "pageindex_search")
        if fallback:
            return fallback[:top_k]

        # Cả hybrid lẫn vectorless đều không tìm được gì đáng tin → trả RỖNG.
        # Trả kết quả hybrid dưới ngưỡng ở đây là có hại: chúng là chunk lạc đề
        # (thực tế đo được: menu điều hướng), Task 10 sẽ nhận context rác rồi bịa
        # ra câu trả lời. Trả rỗng thì Task 10 đi đúng nhánh "không thể xác minh
        # thông tin này" — chính là hành vi rubric yêu cầu.
        print("  → Fallback cũng không có kết quả → trả rỗng")
        return []

    return final_results[:top_k]


# =============================================================================
# CALIBRATION — đo ngưỡng thay vì đoán
# =============================================================================

RELEVANT_QUERIES = [
    "Học phí chương trình cử nhân năm 2026 là bao nhiêu?",
    "Điều kiện để được nhận học bổng là gì?",
    "Hồ sơ nhập học cần những giấy tờ nào?",
    "Thư viện RMIT có dịch vụ gì cho sinh viên?",
    "Sinh viên được hỗ trợ những gì về việc làm?",
]

IRRELEVANT_QUERIES = [
    "xyzabc123nonsense",
    "Công thức nấu phở bò gia truyền",
    "Giá bitcoin hôm nay bao nhiêu",
    "Cách thay lốp xe ô tô",
    "asdkjh qwe zxc",
]


def calibrate_threshold() -> float:
    """
    Đo điểm cosine của nhóm câu hỏi liên quan vs lạc đề, đề xuất ngưỡng ở giữa.

    Đây là cách duy nhất chọn SCORE_THRESHOLD có căn cứ: thang cosine phụ thuộc
    embedding model và corpus, con số của nhóm khác không dùng lại được.
    """
    def best_scores(queries: list[str]) -> list[float]:
        scores = []
        for q in queries:
            results = semantic_search(q, top_k=1)
            score = results[0]["score"] if results else 0.0
            scores.append(score)
            print(f"    {score:.4f}  {q[:55]}")
        return scores

    print("Nhóm LIÊN QUAN:")
    relevant = best_scores(RELEVANT_QUERIES)
    print("\nNhóm LẠC ĐỀ:")
    irrelevant = best_scores(IRRELEVANT_QUERIES)

    low_relevant = min(relevant) if relevant else 0.0
    high_irrelevant = max(irrelevant) if irrelevant else 0.0

    print(f"\n  Liên quan  — thấp nhất: {low_relevant:.4f}")
    print(f"  Lạc đề     — cao nhất:  {high_irrelevant:.4f}")

    if low_relevant <= high_irrelevant:
        print(
            "\n  ⚠ Hai nhóm CHỒNG LẤN — không có ngưỡng nào tách sạch được.\n"
            "    Ngưỡng nào cũng sẽ vừa bỏ sót vừa báo nhầm; cân nhắc đổi embedding\n"
            "    model hoặc bỏ HyDE ở Task 5 rồi đo lại."
        )

    suggested = round((low_relevant + high_irrelevant) / 2, 2)
    print(f"\n  → Đề xuất SCORE_THRESHOLD = {suggested}")
    return suggested


if __name__ == "__main__":
    import sys

    if "--calibrate" in sys.argv:
        calibrate_threshold()
        sys.exit(0)

    test_queries = [
        "Học phí tại RMIT Việt Nam là bao nhiêu?",
        "Làm sao để đặt phòng học nhóm ở thư viện?",
        "Sinh viên quốc tế có những học bổng nào?",
        "xyzabc123nonsense",  # Query không có kết quả → test fallback
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.4f}] [{r['source']}] {r['content'][:80]}...")
