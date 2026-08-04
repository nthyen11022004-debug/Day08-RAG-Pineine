"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install "markitdown[pdf]"
    # Lưu ý: cần extra [pdf] để convert được file PDF. Chỉ "pip install markitdown"
    # (không có extra) sẽ báo MissingDependencyException khi convert PDF, dù JSON/DOCX
    # vẫn convert bình thường.

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục

Ghi chú về nguồn dữ liệu:
    - legal/ là PDF song ngữ Việt–Anh của RMIT, MarkItDown giữ đúng dấu tiếng Việt.
    - news/ là JSON do Task 2 crawl, đã có sẵn markdown nên không cần MarkItDown —
      chỉ cần bọc thêm header metadata để Task 10 trích dẫn được nguồn.
"""

import json
import re
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Ngưỡng khớp với test: file .md phải > 200 ký tự mới coi là convert thành công.
MIN_CONTENT_CHARS = 200

# Rác lặp lại trong PDF của RMIT: nhãn phân loại tài liệu được đóng dấu ở mọi trang.
# Không dọn thì chuỗi này lọt vào gần như mọi chunk ở Task 4, làm nhiễu BM25 (Task 6)
# và ăn token vô ích trong context của Task 10.
PDF_NOISE_PATTERN = re.compile(r"^RMIT Classification:.*$", re.MULTILINE)


def clean_text(text: str) -> str:
    """Dọn rác lặp lại và khoảng trắng thừa sau khi convert."""
    text = PDF_NOISE_PATTERN.sub("", text)
    text = text.replace("\x0c", "\n")   # form feed ngăn trang trong PDF
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()
    converted = 0

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            try:
                result = md.convert(str(filepath))
            except Exception as e:
                # Một PDF hỏng không được làm chết cả mẻ convert.
                print(f"  ✗ Bỏ qua ({type(e).__name__}: {e})")
                continue

            content = clean_text(result.text_content)
            if len(content) < MIN_CONTENT_CHARS:
                # PDF scan (ảnh, không có text layer) sẽ ra gần như rỗng — báo rõ
                # thay vì lưu file rác rồi để Task 4 index khoảng trắng.
                print(f"  ✗ Bỏ qua: chỉ {len(content)} ký tự, có thể là PDF scan")
                continue

            header = (
                f"# {filepath.stem}\n\n"
                f"**Source:** {filepath.name}\n"
                f"**Type:** legal\n\n---\n\n"
            )

            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(header + content, encoding="utf-8")
            converted += 1
            print(f"  ✓ Saved: {output_path.name} ({len(content)} ký tự)")

    print(f"→ {converted} văn bản chính sách")
    return converted


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    converted = 0

    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            data = json.loads(filepath.read_text(encoding="utf-8"))

            content = clean_text(data.get("content_markdown", ""))
            if len(content) < MIN_CONTENT_CHARS:
                print(f"  ✗ Bỏ qua: chỉ {len(content)} ký tự")
                continue

            # Header giữ lại url + ngày crawl để Task 10 trích dẫn được nguồn thật,
            # vì metadata của ChromaDB ở Task 4 chỉ mang tên file.
            header = (
                f"# {data.get('title', 'Unknown')}\n\n"
                f"**Source:** {data.get('url', 'N/A')}\n"
                f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"
            )

            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(header + content, encoding="utf-8")
            converted += 1
            print(f"  ✓ Saved: {output_path.name} ({len(content)} ký tự)")

    print(f"→ {converted} bài viết")
    return converted


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    n_legal = convert_legal_docs()

    print("\n--- News Articles ---")
    n_news = convert_news_articles()

    print("\n✓ Done! Output tại:", OUTPUT_DIR)
    print(f"  Tổng: {n_legal + n_news} file markdown")

    if n_legal == 0:
        print("⚠ Chưa có văn bản chính sách nào — kiểm tra data/landing/legal/")
    if n_news == 0:
        print("⚠ Chưa có bài viết nào — chạy Task 2 trước")


if __name__ == "__main__":
    convert_all()
