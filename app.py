"""
RAG Chatbot — University Services.

Streamlit app kết nối RAG Retrieval (Task 9) và Generation (Task 10).

Chạy:
    streamlit run app.py

Nguyên tắc thiết kế: mọi thứ hiển thị đều lấy từ dữ liệu thật mà src/ trả về,
không hard-code. Cụ thể là 3 thứ mà bản starter bỏ sót:
    - `retrieval_source` ('hybrid' | 'pageindex') — bằng chứng logic fallback của
      Task 9 có chạy hay không, chính là thứ cần chỉ ra khi demo.
    - `metadata['path']` / `['section']` — Task 8 trả về đường dẫn mục, cho phép
      truy vết câu trả lời về đúng đề mục trong tài liệu gốc.
    - Thang điểm khác nhau giữa các nguồn (cosine, BM25, RRF, rank) — hiển thị
      thô sẽ gây hiểu nhầm, nên phải ghi rõ đơn vị.
"""

import html
import os
import re
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Thêm project root vào sys.path để import các task từ src/
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

STANDARDIZED_DIR = PROJECT_ROOT / "data" / "standardized"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="University Services RAG Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Bảng màu chọn theo tiêu chí đọc được trên CẢ nền sáng lẫn tối — Streamlit cho
# người dùng đổi theme bất cứ lúc nào, và `prefers-color-scheme` của trình duyệt
# không phản ánh lựa chọn đó, nên không thể dựa vào media query.
st.markdown(
    """
    <style>
      .badge {
        display: inline-block; padding: 0.12rem 0.5rem; margin-right: 0.35rem;
        border-radius: 999px; font-size: 0.72rem; font-weight: 600;
        letter-spacing: 0.01em; white-space: nowrap;
      }
      .badge-hybrid    { background: rgba(47,129,247,0.18); color: #2f81f7; }
      .badge-pageindex { background: rgba(210,153,34,0.20); color: #d29922; }
      .badge-none      { background: rgba(139,148,158,0.20); color: #8b949e; }
      .badge-legal     { background: rgba(163,113,247,0.18); color: #a371f7; }
      .badge-news      { background: rgba(63,185,80,0.18);  color: #3fb950; }
      .badge-score     { background: rgba(139,148,158,0.16); color: #8b949e;
                         font-family: ui-monospace, monospace; }
      .src-path {
        font-size: 0.78rem; opacity: 0.75; margin: 0.15rem 0 0.4rem 0;
        font-family: ui-monospace, monospace; word-break: break-word;
      }
      .src-snippet {
        font-size: 0.86rem; line-height: 1.55; opacity: 0.92;
        max-height: 8.5rem; overflow: hidden;
      }
      .src-snippet mark {
        background: rgba(210,153,34,0.35); color: inherit;
        padding: 0 0.12rem; border-radius: 2px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# HELPERS
# =============================================================================

# NotImplementedError chỉ nói tên hàm ("Implement semantic_search"), không nói task
# nào — map lại để người demo biết ngay phải mở file nào thay vì đi đọc traceback.
FUNCTION_TO_TASK = {
    "semantic_search": ("Task 5", "src/task5_semantic_search.py"),
    "lexical_search": ("Task 6", "src/task6_lexical_search.py"),
    "build_bm25_index": ("Task 6", "src/task6_lexical_search.py"),
    "rerank": ("Task 7", "src/task7_reranking.py"),
    "rerank_rrf": ("Task 7", "src/task7_reranking.py"),
    "rerank_mmr": ("Task 7", "src/task7_reranking.py"),
    "pageindex_search": ("Task 8", "src/task8_pageindex_vectorless.py"),
    "retrieve": ("Task 9", "src/task9_retrieval_pipeline.py"),
    "generate_with_citation": ("Task 10", "src/task10_generation.py"),
}


def explain_not_implemented(error: NotImplementedError) -> str:
    """Đổi 'Implement semantic_search' thành chỉ dẫn cụ thể."""
    message = str(error)
    for func, (task, path) in FUNCTION_TO_TASK.items():
        if func in message:
            return (
                f"⚠️ **{task} chưa được implement.**\n\n"
                f"Hàm `{func}()` trong `{path}` vẫn đang `raise NotImplementedError`. "
                f"Hoàn thành hàm đó rồi hỏi lại."
            )
    return f"⚠️ **Pipeline chưa hoàn chỉnh:** {message}"


def llm_provider() -> tuple[str, str]:
    """Provider nào đang được dùng — đọc cùng logic với task10 để không lệch."""
    openrouter = os.getenv("OPENROUTER_API_KEY", "").strip()
    if openrouter and not openrouter.endswith("..."):
        return "OpenRouter", "openai/gpt-4o-mini"
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key and not openai_key.endswith("..."):
        return "OpenAI", "gpt-4o-mini"
    return "Chưa cấu hình", "—"


@st.cache_data(ttl=30, show_spinner=False)
def corpus_stats() -> dict:
    """Đếm tài liệu đã chuẩn hoá và kiểm tra vector store."""
    legal = list((STANDARDIZED_DIR / "legal").glob("*.md")) if STANDARDIZED_DIR.exists() else []
    news = list((STANDARDIZED_DIR / "news").glob("*.md")) if STANDARDIZED_DIR.exists() else []

    indexed = None
    if CHROMA_DIR.exists():
        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            indexed = sum(c.count() for c in client.list_collections())
        except Exception:
            # Vector store hỏng/khoá không được làm sập UI — chỉ mất phần thống kê.
            indexed = None

    return {
        "legal": len(legal),
        "news": len(news),
        "chars": sum(f.stat().st_size for f in legal + news),
        "indexed": indexed,
    }


def highlight(text: str, query: str) -> str:
    """
    Bôi vàng từ khoá của câu hỏi trong trích đoạn nguồn.

    Phải escape HTML TRƯỚC khi chèn thẻ <mark>: nội dung crawl từ web có chứa
    dấu ngoặc nhọn, không escape thì vừa vỡ layout vừa là lỗ chèn HTML.
    """
    escaped = html.escape(text)
    terms = {t for t in re.findall(r"\w+", query, flags=re.UNICODE) if len(t) >= 3}
    if not terms:
        return escaped

    pattern = "|".join(sorted((re.escape(t) for t in terms), key=len, reverse=True))
    return re.sub(f"({pattern})", r"<mark>\1</mark>", escaped, flags=re.IGNORECASE)


def score_label(source_kind: str) -> str:
    """
    Ghi rõ điểm đang ở thang nào.

    Không ghi thì người xem sẽ so điểm RRF (~0.016) với cosine (~0.7) và tưởng
    kết quả tệ — đây đúng là cái bẫy mà đề bài cảnh báo ở Task 9.
    """
    return "rank" if source_kind == "pageindex" else "RRF"


def render_sources(sources: list[dict], query: str, source_kind: str) -> None:
    """Vẽ danh sách nguồn. Một hàm dùng chung cho cả lịch sử chat lẫn câu mới."""
    if not sources:
        return

    unit = score_label(source_kind)
    with st.expander(f"📚 Nguồn tham khảo · {len(sources)} đoạn", expanded=False):
        for i, src in enumerate(sources, 1):
            meta = src.get("metadata", {}) or {}
            name = meta.get("source", "Unknown")
            doc_type = meta.get("type", "unknown")
            score = src.get("score", 0.0)
            # Task 8 trả 'path' (đường dẫn mục), Task 4 trả 'chunk_index'.
            path = meta.get("path") or meta.get("section")
            chunk_index = meta.get("chunk_index")

            with st.container(border=True):
                type_class = "legal" if doc_type == "legal" else "news"
                st.markdown(
                    f"<span class='badge badge-{type_class}'>{html.escape(str(doc_type))}</span>"
                    f"<span class='badge badge-score'>{unit} {score:.4f}</span>"
                    f"**[{i}] {html.escape(str(name))}**",
                    unsafe_allow_html=True,
                )

                if path:
                    st.markdown(
                        f"<div class='src-path'>📍 {html.escape(str(path))}</div>",
                        unsafe_allow_html=True,
                    )
                elif chunk_index is not None:
                    st.markdown(
                        f"<div class='src-path'>📍 chunk #{chunk_index}</div>",
                        unsafe_allow_html=True,
                    )

                content = src.get("content", "")
                st.markdown(
                    f"<div class='src-snippet'>{highlight(content[:400], query)}…</div>",
                    unsafe_allow_html=True,
                )

                if len(content) > 400:
                    with st.popover("Xem toàn bộ đoạn"):
                        st.text(content)


def render_retrieval_badge(source_kind: str, elapsed: float | None) -> None:
    """
    Hiện đường dẫn retrieval đã đi qua.

    Đây là điểm demo quan trọng nhất: 'pageindex' nghĩa là hybrid search đã trượt
    ngưỡng và fallback của Task 9 kích hoạt đúng thiết kế.
    """
    labels = {
        "hybrid": ("hybrid", "Semantic + BM25 → RRF"),
        "pageindex": ("pageindex", "Fallback: vectorless (hybrid dưới ngưỡng)"),
        "none": ("none", "Không tìm được tài liệu liên quan"),
    }
    key = source_kind if source_kind in labels else "none"
    badge, description = labels[key]

    timing = f" · {elapsed:.1f}s" if elapsed is not None else ""
    st.markdown(
        f"<span class='badge badge-{badge}'>{badge}</span>"
        f"<span style='font-size:0.78rem;opacity:0.7'>{description}{timing}</span>",
        unsafe_allow_html=True,
    )


# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("🎓 University Services RAG")
    st.caption(
        "Trợ lý hỏi đáp về dịch vụ và chính sách đại học "
        "(học phí, học bổng, ký túc xá, thư viện)"
    )

    st.divider()

    st.subheader("💡 Câu hỏi gợi ý")
    # Đổi sang tiếng Việt cho khớp corpus: dữ liệu crawl từ bản /vi của rmit.edu.vn,
    # hỏi tiếng Anh thì BM25 (Task 6) không khớp được token nào.
    suggestions = [
        "Học phí chương trình cử nhân năm 2026 là bao nhiêu?",
        "Điều kiện để được nhận học bổng là gì?",
        "Hồ sơ nhập học cần những giấy tờ nào?",
        "Thư viện RMIT có những dịch vụ gì cho sinh viên?",
        "Sinh viên được hỗ trợ những gì về đời sống và việc làm?",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True, key=f"sug_{s[:24]}"):
            st.session_state["pending_query"] = s

    st.divider()

    st.subheader("⚙️ Thiết lập")
    top_k = st.slider(
        "Số đoạn đưa vào context (top_k)", 3, 10, 5,
        help="Nhiều hơn không phải lúc nào cũng tốt: context dài dễ gây "
             "'lost in the middle' — LLM nhớ đầu và cuối, quên giữa.",
    )

    st.divider()

    st.subheader("📊 Trạng thái hệ thống")
    stats = corpus_stats()
    provider, model = llm_provider()

    left, right = st.columns(2)
    left.metric("Tài liệu", stats["legal"] + stats["news"])
    right.metric("Chunk đã index", stats["indexed"] if stats["indexed"] is not None else "—")

    st.caption(
        f"📁 {stats['legal']} văn bản chính sách · {stats['news']} bài viết "
        f"· {stats['chars']:,} ký tự"
    )

    if stats["indexed"] is None:
        st.warning("Chưa có vector store — chạy `python -m src.task4_chunking_indexing`", icon="⚠️")

    if provider == "Chưa cấu hình":
        st.error("Chưa có API key trong `.env`", icon="🔑")
    else:
        st.caption(f"🤖 LLM: **{provider}** · `{model}`")

    st.divider()

    st.caption("**Kiến trúc**")
    st.code(
        "Query\n"
        "  ├─ Semantic (Task 5) ─┐\n"
        "  ├─ BM25     (Task 6) ─┴─ RRF (Task 7)\n"
        "  └─ nếu dưới ngưỡng → PageIndex (Task 8)\n"
        "        └─ Generation có citation (Task 10)",
        language=None,
    )

    if st.session_state.messages:
        if st.button("🗑️ Xoá lịch sử chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# =============================================================================
# MAIN CHAT AREA
# =============================================================================

st.title("🎓 University Services RAG Chatbot")
st.caption(
    "Hỏi đáp về dịch vụ và chính sách đại học — câu trả lời kèm trích dẫn nguồn"
)

if not st.session_state.messages:
    st.info(
        "Chọn một câu hỏi gợi ý ở thanh bên, hoặc nhập câu hỏi của bạn bên dưới. "
        "Mỗi câu trả lời đều kèm các đoạn tài liệu gốc đã dùng để tổng hợp.",
        icon="👋",
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_retrieval_badge(msg.get("retrieval_source", "none"), msg.get("elapsed"))
            render_sources(
                msg.get("sources", []),
                msg.get("query", ""),
                msg.get("retrieval_source", "hybrid"),
            )

# =============================================================================
# QUERY HANDLING
# =============================================================================

user_input = st.chat_input("Nhập câu hỏi về chính sách/dịch vụ đại học...")
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        answer = ""
        sources: list[dict] = []
        retrieval_source = "none"
        elapsed = None

        with st.status("Đang chạy RAG pipeline...", expanded=False) as status:
            started = time.perf_counter()
            try:
                st.write("🔍 Truy xuất tài liệu (semantic + BM25 → RRF)")
                from src.task10_generation import generate_with_citation

                response = generate_with_citation(query, top_k=top_k)

                answer = response.get("answer", "Chưa thể trả lời.")
                sources = response.get("sources", []) or []
                retrieval_source = response.get("retrieval_source", "hybrid")

                elapsed = time.perf_counter() - started
                st.write(f"✍️ Tổng hợp câu trả lời có trích dẫn ({len(sources)} đoạn)")
                status.update(
                    label=f"Hoàn tất trong {elapsed:.1f}s", state="complete"
                )

            except NotImplementedError as e:
                answer = explain_not_implemented(e)
                status.update(label="Pipeline chưa hoàn chỉnh", state="error")

            except Exception as e:
                answer = (
                    f"❌ **Lỗi khi chạy RAG Pipeline**\n\n"
                    f"`{type(e).__name__}: {e}`"
                )
                status.update(label="Pipeline lỗi", state="error")

        st.markdown(answer)

        if sources:
            render_retrieval_badge(retrieval_source, elapsed)
            render_sources(sources, query, retrieval_source)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "retrieval_source": retrieval_source,
        "query": query,
        "elapsed": elapsed,
    })
