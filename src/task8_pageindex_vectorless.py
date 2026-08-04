"""
Task 8 — PageIndex Vectorless RAG (bản chạy local, KHÔNG cần PAGEINDEX_API_KEY).

Vì sao không dùng API của pageindex.ai:
    Package `pageindex` trên PyPI chỉ có `client.py` — thuần client gọi dịch vụ trả
    phí. Nhưng PageIndex là dự án mã nguồn mở và bản thân thuật toán không hề cần
    dịch vụ đó: nó chỉ cần một LLM. Nên module này tự implement thuật toán, dùng
    `OPENAI_API_KEY` nhóm đã có, không tốn thêm chi phí và không phụ thuộc bên thứ ba.

Vectorless RAG hoạt động thế nào (khác hoàn toàn Task 5):
    - Task 5 (dense): embed toàn bộ chunk thành vector, so cosine → "vibe retrieval",
      không giải thích được vì sao chunk đó được chọn.
    - PageIndex (vectorless): dựng cây mục lục (Table of Contents) từ cấu trúc thật
      của tài liệu, rồi đưa CHỈ phần mục lục cho LLM và bảo nó suy luận xem câu trả
      lời nằm ở mục nào — giống hệt cách con người tra một cuốn cẩm nang dày.
      Không có embedding, không có vector store, không có chunk nhân tạo.
    - Đổi lại: mỗi kết quả truy vết được về đúng mục/đề mục, nên giải thích được.

Vì sao nó hợp vai fallback ở Task 9:
    Hybrid search thất bại chủ yếu khi câu hỏi dùng từ ngữ khác hẳn tài liệu — đúng
    lúc đó thì suy luận trên đề mục lại ăn đứt so khớp bề mặt.

Nguồn thuật toán: https://github.com/VectifyAI/PageIndex
"""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONFIGURATION
# =============================================================================

# Mục dài hơn ngưỡng này bị cắt nhỏ. KHÔNG phải chunking như Task 4: ở đây ranh
# giới mục vẫn được tôn trọng, chỉ mục nào quá dài mới tách, nên vẫn giữ được
# ngữ cảnh của đề mục cha.
MAX_SECTION_CHARS = 4000

# Số mục tối đa nhét vào mục lục gửi cho LLM. Chỉ có TIÊU ĐỀ được gửi, không có
# nội dung — đó chính là lý do vectorless rẻ: ~300 tiêu đề chỉ tốn vài nghìn token.
MAX_TOC_ENTRIES = 400

LLM_MODEL_OPENAI = "gpt-4o-mini"
LLM_MODEL_OPENROUTER = "openai/gpt-4o-mini"

# Heading kiểu markdown (file news từ Task 2).
MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

# Heading đánh số kiểu "4. Tuition Fees" / "2.1 Creation of Financial Liability".
# File legal là PDF convert ra nên không có heading markdown, nhưng vẫn giữ nguyên
# hệ thống mục đánh số của văn bản gốc — đây chính là cấu trúc PageIndex khai thác.
NUMBERED_HEADING = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){0,2})\.?\s+(\S.{2,90})$")

# Regex trên cũng khớp cả mục danh sách đánh số trong thân bài ("1. Học bổng hỗ trợ
# toàn bộ hoặc một phần học phí..."), biến mỗi câu thành một "mục" rác. Đo trên
# corpus thật: heading thật trong rmit-hoc-phi.md có độ dài trung vị 34, dài nhất
# ~57 ký tự; còn mục danh sách bị bắt nhầm đều từ 69 ký tự trở lên. Cắt ở 60.
MAX_HEADING_CHARS = 60

# Block metadata do Task 3 chèn đầu mỗi file. Hữu ích khi đọc, nhưng nếu để lọt vào
# cây thì nó thành một "mục" chỉ chứa URL và ngày crawl — LLM sẽ chọn nhầm nó.
META_LINE = re.compile(r"^\*\*(Source|Crawled|Type):\*\*|^---\s*$")

# Footer đóng dấu ở mọi trang PDF ("17 June 2026 ... Page 7 of 59"). Không lọc thì
# vừa bị nhận nhầm thành heading, vừa lặp 59 lần trong nội dung.
PAGE_FOOTER = re.compile(r"Page\s+\d+\s+of\s+\d+", re.IGNORECASE)

# Mục lục trong PDF dùng dấu chấm dẫn ("1. Foreword ......... 3") — là mục lục,
# không phải nội dung, nên không tính thành mục.
DOT_LEADER = re.compile(r"\.{4,}")


# =============================================================================
# BUILD DOCUMENT TREE — thay cho việc "upload" lên dịch vụ
# =============================================================================

_TREE_CACHE: list[dict] | None = None


def _is_heading(line: str) -> tuple[int, str] | None:
    """
    Trả về (level, title) nếu dòng là heading, ngược lại None.

    level 1-6 cho markdown; với heading đánh số thì level = độ sâu của số
    ("4" → 1, "4.1" → 2, "4.1.2" → 3).
    """
    if PAGE_FOOTER.search(line) or DOT_LEADER.search(line):
        return None

    md = MD_HEADING.match(line)
    if md:
        return len(md.group(1)), md.group(2).strip()

    if len(line.strip()) > MAX_HEADING_CHARS:
        return None  # quá dài để là heading → là câu trong thân bài

    num = NUMBERED_HEADING.match(line)
    if num:
        number, title = num.group(1), num.group(2).strip()
        return number.count(".") + 1, f"{number}. {title}"

    return None


def _split_long_section(node: dict) -> list[dict]:
    """Cắt mục quá dài thành nhiều phần, giữ nguyên tiêu đề mục cha."""
    content = node["content"]
    if len(content) <= MAX_SECTION_CHARS:
        return [node]

    parts = []
    paragraphs = content.split("\n\n")
    buffer = ""
    for para in paragraphs:
        if buffer and len(buffer) + len(para) > MAX_SECTION_CHARS:
            parts.append(buffer.strip())
            buffer = ""
        buffer += para + "\n\n"
    if buffer.strip():
        parts.append(buffer.strip())

    out = []
    for i, part in enumerate(parts, 1):
        child = dict(node)
        child["title"] = f"{node['title']} (phần {i}/{len(parts)})"
        child["content"] = part
        out.append(child)
    return out


def build_document_tree() -> list[dict]:
    """
    Dựng cây mục lục từ toàn bộ markdown trong data/standardized/.

    Đây là bước thay thế cho `upload_documents()` của bản hosted: PageIndex hosted
    upload PDF lên server để họ dựng cây; ở đây ta dựng ngay tại local.

    Returns:
        List of {
            'id': str,        # mã ngắn để LLM tham chiếu, vd "S042"
            'title': str,     # tiêu đề mục
            'path': str,      # đường dẫn phân cấp: "file > mục cha > mục con"
            'content': str,
            'metadata': dict,
        }
    """
    nodes: list[dict] = []

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        doc_type = "legal" if "legal" in md_file.parts else "news"
        text = md_file.read_text(encoding="utf-8")

        # Ngăn xếp tiêu đề cha để dựng breadcrumb — thứ khiến kết quả truy vết được.
        stack: list[tuple[int, str]] = []
        current_title = md_file.stem
        current_lines: list[str] = []

        def flush():
            body = "\n".join(current_lines).strip()
            if not body:
                return
            crumb = " > ".join([md_file.stem] + [t for _, t in stack])
            nodes.append({
                "title": current_title,
                "path": crumb,
                "content": body,
                "metadata": {
                    "source": md_file.name,
                    "type": doc_type,
                    "section": current_title,
                },
            })

        for line in text.split("\n"):
            if PAGE_FOOTER.search(line) or META_LINE.match(line):
                continue

            heading = _is_heading(line)
            if heading:
                flush()
                level, title = heading
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                current_title = title
                current_lines = []
            else:
                current_lines.append(line)

        flush()

    # Cắt mục quá dài rồi mới gán id, để id luôn liên tục.
    expanded: list[dict] = []
    for node in nodes:
        expanded.extend(_split_long_section(node))

    for i, node in enumerate(expanded):
        node["id"] = f"S{i:03d}"

    return expanded


def get_tree() -> list[dict]:
    """Cây được cache lại — dựng lại mỗi lần query thì Task 9 sẽ chậm thảm hại."""
    global _TREE_CACHE
    if _TREE_CACHE is None:
        _TREE_CACHE = build_document_tree()
    return _TREE_CACHE


def format_toc(tree: list[dict]) -> str:
    """
    Tạo mục lục dạng text để đưa cho LLM.

    CHỈ gửi tiêu đề + đường dẫn, KHÔNG gửi nội dung. Đây là điểm mấu chốt khiến
    vectorless rẻ và khả thi: 300 mục chỉ tốn vài nghìn token, trong khi gửi toàn
    bộ nội dung sẽ là hàng trăm nghìn token.
    """
    lines = []
    for node in tree[:MAX_TOC_ENTRIES]:
        preview = node["content"][:80].replace("\n", " ")
        lines.append(f"[{node['id']}] {node['path']} :: {preview}")
    return "\n".join(lines)


# =============================================================================
# LLM REASONING — thay cho cosine similarity
# =============================================================================

def _llm_client():
    """
    Trả về (client, model) hoặc (None, None) nếu chưa có key nào.

    Đọc OPENROUTER trước rồi mới tới OPENAI, nhưng phải strip() và kiểm tra rỗng:
    .env.example để sẵn placeholder, nếu chỉ dùng `or` thì chuỗi placeholder vẫn
    truthy và sẽ được chọn trước key thật.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return None, None

    or_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if or_key and not or_key.endswith("..."):
        return (
            OpenAI(api_key=or_key, base_url="https://openrouter.ai/api/v1"),
            LLM_MODEL_OPENROUTER,
        )

    oa_key = os.getenv("OPENAI_API_KEY", "").strip()
    if oa_key and not oa_key.endswith("..."):
        return OpenAI(api_key=oa_key), LLM_MODEL_OPENAI

    return None, None


SELECT_PROMPT = """Bạn là công cụ tra cứu tài liệu. Dưới đây là MỤC LỤC của kho tài \
liệu về dịch vụ và chính sách đại học RMIT. Mỗi dòng có dạng:
[mã] đường dẫn phân cấp :: trích đoạn đầu mục

Nhiệm vụ: chọn tối đa {top_k} mục có khả năng CHỨA CÂU TRẢ LỜI cho câu hỏi.
Suy luận theo nghĩa của đề mục, không chỉ so khớp từ khoá bề mặt.
Nếu không mục nào liên quan, trả về danh sách rỗng.

Chỉ trả về JSON đúng định dạng: {{"ids": ["S001", "S042"]}}

MỤC LỤC:
{toc}

CÂU HỎI: {query}"""


def select_sections_with_llm(query: str, tree: list[dict], top_k: int) -> list[str]:
    """Hỏi LLM xem câu trả lời nằm ở mục nào. Trả về danh sách id."""
    client, model = _llm_client()
    if client is None:
        return []

    prompt = SELECT_PROMPT.format(top_k=top_k, toc=format_toc(tree), query=query)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,  # tra cứu cần ổn định, không cần sáng tạo
        response_format={"type": "json_object"},
    )

    payload = json.loads(response.choices[0].message.content)
    ids = payload.get("ids", [])
    return [i for i in ids if isinstance(i, str)][:top_k]


def select_sections_by_keyword(query: str, tree: list[dict], top_k: int) -> list[str]:
    """
    Dự phòng khi không có LLM key: chấm điểm theo số từ khoá khớp tiêu đề.

    Không có nhánh này thì test sẽ đỏ trên máy chưa cấu hình key, và Task 9 sẽ
    vỡ đúng lúc cần fallback nhất.
    """
    terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]
    if not terms:
        return []

    scored = []
    for node in tree:
        haystack = (node["path"] + " " + node["content"][:300]).lower()
        score = sum(haystack.count(t) for t in terms)
        if score > 0:
            scored.append((score, node["id"]))

    scored.sort(reverse=True)
    return [node_id for _, node_id in scored[:top_k]]


# =============================================================================
# PUBLIC API
# =============================================================================

def upload_documents():
    """
    Dựng (và in) cây mục lục từ data/standardized/.

    Bản hosted của PageIndex sẽ upload PDF lên server ở bước này; bản local dựng
    cây ngay tại chỗ nên không có gì phải upload — giữ nguyên tên hàm để khớp
    interface đề bài.
    """
    tree = build_document_tree()
    print(f"✓ Đã dựng cây mục lục: {len(tree)} mục từ {STANDARDIZED_DIR}")

    by_source: dict[str, int] = {}
    for node in tree:
        by_source[node["metadata"]["source"]] = by_source.get(node["metadata"]["source"], 0) + 1
    for source, count in sorted(by_source.items(), key=lambda x: -x[1])[:10]:
        print(f"    {count:4d} mục  {source}")

    return tree


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    tree = get_tree()
    if not tree:
        return []

    try:
        ids = select_sections_with_llm(query, tree, top_k)
    except Exception as e:
        # Hết quota / mạng lỗi không được làm sập fallback của Task 9.
        print(f"  ⚠ LLM lỗi ({type(e).__name__}), chuyển sang so khớp từ khoá: {e}")
        ids = []

    if not ids:
        ids = select_sections_by_keyword(query, tree, top_k)

    by_id = {node["id"]: node for node in tree}

    results = []
    for rank, node_id in enumerate(ids, 1):
        node = by_id.get(node_id)
        if node is None:
            continue  # LLM đôi khi bịa mã không tồn tại
        results.append({
            "content": node["content"],
            # PageIndex không trả điểm tương đồng — điểm ở đây thuần theo THỨ HẠNG.
            # Đừng đem so với SCORE_THRESHOLD cosine của Task 9; đây chỉ để sắp xếp.
            "score": round(1.0 / rank, 4),
            "metadata": {**node["metadata"], "path": node["path"]},
            "source": "pageindex",
        })

    return results[:top_k]


if __name__ == "__main__":
    client, model = _llm_client()
    print(f"LLM: {model or 'KHÔNG CÓ KEY → dùng so khớp từ khoá'}\n")

    print("Dựng cây mục lục...")
    upload_documents()

    test_queries = [
        "Học phí chương trình cử nhân năm 2026 là bao nhiêu?",
        "Điều kiện để được nhận học bổng là gì?",
        "Hồ sơ nhập học cần những giấy tờ nào?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}\nQ: {q}\n{'-'*70}")
        for r in pageindex_search(q, top_k=3):
            print(f"[{r['score']:.2f}] {r['metadata']['path'][:70]}")
            print(f"        {r['content'][:120].strip()}...")
