"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"

Gợi ý LLM: OpenRouter có nhiều model gắn hậu tố ":free" không tính phí — xem
https://openrouter.ai/models?max_price=0 — phù hợp nếu chưa có credit trả phí.
Base URL: "https://openrouter.ai/api/v1", dùng chung interface với OpenAI SDK.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3

# TODO: Chọn LLM model (OpenRouter model ID)
LLM_MODEL = "openai/gpt-4o-mini"  # hoặc model ":free" nếu chưa có credit


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi về dịch vụ và chính sách đại học
(học phí, học bổng, ký túc xá, thư viện, đăng ký học phần).

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin từ context được cung cấp — KHÔNG bịa đặt
2. Mỗi khẳng định phải có trích dẫn ngay sau, ví dụ: [Tuition Fees, 2026]
3. Nếu context không đủ thông tin → trả lời: "Tôi không thể xác minh thông tin này từ nguồn hiện có"
4. Trả lời bằng tiếng Việt, có cấu trúc rõ ràng theo đoạn văn
5. Không suy luận hay mở rộng ngoài những gì được nêu trong context"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    # TODO: Implement reordering
    #
    # if len(chunks) <= 2:
    #     return chunks
    #
    # front = chunks[::2]   # index 0, 2, 4 -> đặt ở đầu
    # back = chunks[1::2]   # index 1, 3    -> đặt ở cuối (reversed)
    # return front + back[::-1]
    if len(chunks) <= 2:
        return list(chunks)

    # Keep the best-ranked chunk first and place the second-best one last,
    # where an LLM generally pays more attention. This does not mutate input.
    return list(chunks[::2]) + list(chunks[1::2])[::-1]


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    # TODO: Implement context formatting
    #
    # context_parts = []
    # for i, chunk in enumerate(chunks, 1):
    #     source = chunk.get("metadata", {}).get("source", f"Source {i}")
    #     doc_type = chunk.get("metadata", {}).get("type", "unknown")
    #     context_parts.append(
    #         f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
    #         f"{chunk['content']}\n"
    #     )
    # return "\n---\n".join(context_parts)
    context_parts = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", f"Source {index}")
        doc_type = metadata.get("type", "unknown")
        content = chunk.get("content", "").strip()
        if content:
            context_parts.append(
                f"[Document {index} | Source: {source} | Type: {doc_type}]\n{content}"
            )
    return "\n\n---\n\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # TODO: Implement generation pipeline
    #
    # # Step 1: Retrieve
    # chunks = retrieve(query, top_k=top_k)
    #
    # # Step 2: Reorder
    # reordered = reorder_for_llm(chunks)
    #
    # # Step 3: Format context
    # context = format_context(reordered)
    #
    # # Step 4: Build prompt
    # user_message = f"""Context:\n{context}\n\n---\n\nQuestion: {query}"""
    #
    # # Step 5: Call LLM (OpenRouter — OpenAI-compatible API)
    # from openai import OpenAI
    # api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    # client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    #
    # response = client.chat.completions.create(
    #     model=LLM_MODEL,
    #     messages=[
    #         {"role": "system", "content": SYSTEM_PROMPT},
    #         {"role": "user", "content": user_message}
    #     ],
    #     temperature=TEMPERATURE,
    #     top_p=TOP_P,
    # )
    #
    # answer = response.choices[0].message.content
    #
    # # Step 6: Return
    # return {
    #     "answer": answer,
    #     "sources": chunks,
    #     "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none"
    # }
    if not query or not query.strip():
        raise ValueError("query must not be empty")

    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    context = format_context(reorder_for_llm(chunks))
    if not context:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": chunks,
            "retrieval_source": chunks[0].get("source", "hybrid"),
        }

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openrouter_key and not openai_key:
        raise RuntimeError(
            "Missing API key. Set OPENROUTER_API_KEY or OPENAI_API_KEY in .env."
        )

    from openai import OpenAI

    client = (
        OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")
        if openrouter_key
        else OpenAI(api_key=openai_key)
    )
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query.strip()}"
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    answer = response.choices[0].message.content or (
        "Tôi không thể xác minh thông tin này từ nguồn hiện có."
    )
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid"),
    }


if __name__ == "__main__":
    test_queries = [
        "Học phí tại RMIT Vietnam là bao nhiêu?",
        "Làm sao để đặt phòng học nhóm ở thư viện?",
        "Sinh viên quốc tế có những học bổng nào?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
