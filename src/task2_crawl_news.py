"""
Task 2 — Crawl bài viết/thông báo về dịch vụ đại học.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài viết từ trang công khai của một trường đại học.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    playwright install chromium   # bắt buộc — pip install crawl4ai KHÔNG tự tải browser binary,
                                   # thiếu bước này sẽ báo lỗi
                                   # "BrowserType.launch: Executable doesn't exist"

Nguồn: RMIT Vietnam (rmit.edu.vn) — trang công khai, không cần đăng nhập.
    - Trang tin tức https://www.rmit.edu.vn/news được server-render (HTML có sẵn link
      bài viết), nên script tự khám phá URL bài mới thay vì hard-code danh sách dễ chết.
    - Kèm thêm các trang dịch vụ sinh viên (thư viện, hỗ trợ, học bổng, ký túc xá,
      đăng ký học phần) vì chủ đề bài lab là "University Services" — đây mới là nội dung
      trả lời được các câu hỏi kiểu "đặt phòng học nhóm ở thư viện thế nào?".

Hai chế độ crawl:
    - Crawl4AI (mặc định nếu đã cài + đã `playwright install chromium`): render JS đầy đủ.
    - Fallback requests + BeautifulSoup: dùng khi chưa cài được Crawl4AI. Các trang RMIT
      ở trên là HTML tĩnh nên fallback cho kết quả tương đương.
"""

import asyncio
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

BASE_URL = "https://www.rmit.edu.vn"

# Chatbot của nhóm hỏi đáp tiếng Việt nên corpus phải là tiếng Việt.
# Quan trọng nhất là vì Task 6 dùng BM25 — thuần lexical, khớp theo token. Query
# "học phí" KHÔNG bao giờ khớp document "tuition fee", điểm luôn bằng 0, coi như
# mất hẳn một nửa hybrid search. Embedding đa ngữ (bge-m3) ở Task 5 có thể bắc cầu
# Việt→Anh, nhưng BM25 thì không có cách nào.
# Đổi thành "en" nếu muốn crawl lại bản tiếng Anh.
LANGUAGE = "vi"

# Một số WAF trả 403 cho request không có User-Agent trông giống trình duyệt.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
}

# Delay giữa 2 request để không dội trang trường (crawl có trách nhiệm).
REQUEST_DELAY_SECONDS = 1.0

# Bao nhiêu bài từ trang /news — cộng với SERVICE_PAGES là vượt xa mức tối thiểu 5 bài.
MAX_NEWS_ARTICLES = 8

# Ngưỡng tối thiểu cho thân bài. Dùng ở 2 chỗ:
#   - chọn chiến lược bóc nội dung (xem _clean_html_to_article)
#   - loại bài rỗng trước khi lưu; test yêu cầu file > 500 bytes
MIN_BODY_CHARS = 1500
MIN_SAVE_CHARS = 500

# Tên class/id của các khối điều hướng trên rmit.edu.vn (Adobe AEM dựng menu bằng
# <div>/<ul>, không dùng thẻ <nav>). Khớp phần nào cũng loại cả khối.
NAV_CLASS_PATTERN = re.compile(
    r"top-nav|topnav|primarynav|primary-nav|primary-navbar|nav-wraper|mobinav"
    r"|breadcrumb|pageheader|header-gridcontent|megamenu|cookie|skipcontent"
)


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# Cấu hình theo ngôn ngữ. Trang dịch vụ là trục nội dung chính của chủ đề
# "University Services" — tin tức trên /news phần lớn là bài nghiên cứu, không trả
# lời được câu hỏi kiểu "đặt phòng học nhóm ở thư viện thế nào?".
# Các URL dưới đây đều đã kiểm tra trả về HTTP 200.
SITE_CONFIG = {
    "vi": {
        "news_index": f"{BASE_URL}/vi/tin-tuc/tat-ca-tin-tuc",
        "article_pattern": re.compile(
            r"/vi/tin-tuc/tat-ca-tin-tuc/\d{4}/[a-z]{3}/[a-z0-9-]+$"
        ),
        "service_pages": [
            f"{BASE_URL}/vi/hoc-tap-tai-rmit/hoc-phi",
            f"{BASE_URL}/vi/hoc-tap-tai-rmit/hoc-bong",
            f"{BASE_URL}/vi/hoc-tap-tai-rmit",
            f"{BASE_URL}/vi/doi-song-sinh-vien/ho-tro-sinh-vien",
            f"{BASE_URL}/vi/doi-song-sinh-vien",
            f"{BASE_URL}/vi/doi-song-sinh-vien/co-hoi-trai-nghiem-va-viec-lam",
            f"{BASE_URL}/vi/su-kien",
            # Thư viện KHÔNG có bản tiếng Việt (/vi/thu-vien trả 404) nên giữ trang
            # tiếng Anh — nếu bỏ thì không có dữ liệu nào trả lời được câu hỏi về
            # thư viện, vốn là 1 trong 3 test query của Task 10.
            f"{BASE_URL}/libraryvn",
        ],
    },
    "en": {
        "news_index": f"{BASE_URL}/news",
        "article_pattern": re.compile(r"/news/all-news/\d{4}/[a-z]{3}/[a-z0-9-]+$"),
        "service_pages": [
            f"{BASE_URL}/libraryvn",
            f"{BASE_URL}/students/support",
            f"{BASE_URL}/students/campus-life",
            f"{BASE_URL}/students/my-studies/enrolment",
            f"{BASE_URL}/students/my-studies/fees-and-payments",
            f"{BASE_URL}/study-at-rmit/scholarships",
        ],
    },
}

NEWS_INDEX_URL = SITE_CONFIG[LANGUAGE]["news_index"]
ARTICLE_URL_PATTERN = SITE_CONFIG[LANGUAGE]["article_pattern"]
SERVICE_PAGES = SITE_CONFIG[LANGUAGE]["service_pages"]

# Để trống -> script tự khám phá URL bài viết từ NEWS_INDEX_URL.
# Điền tay vào đây nếu muốn khoá cứng danh sách bài (ví dụ để kết quả tái lập được).
ARTICLE_URLS = []


# =============================================================================
# URL DISCOVERY
# =============================================================================

def discover_article_urls(limit: int = MAX_NEWS_ARTICLES) -> list[str]:
    """
    Lấy danh sách URL bài viết từ trang tin tức (NEWS_INDEX_URL).

    Trang này server-render nên chỉ cần requests, không cần trình duyệt.
    """
    import requests
    from bs4 import BeautifulSoup

    response = requests.get(NEWS_INDEX_URL, headers=HEADERS, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.content.decode("utf-8", errors="replace"), "html.parser")

    urls: list[str] = []
    seen: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        url = urljoin(BASE_URL, anchor["href"]).split("#")[0].rstrip("/")

        # Bài viết thật có dạng .../<năm>/<tháng>/<slug>;
        # loại các link phân trang, chuyên mục, trang chủ tin tức.
        if not ARTICLE_URL_PATTERN.search(url):
            continue
        if url in seen:
            continue

        seen.add(url)
        urls.append(url)

        if len(urls) >= limit:
            break

    return urls


# =============================================================================
# CRAWL — 2 backend
# =============================================================================

def _to_markdown(element) -> str:
    """Convert một khối HTML sang markdown, gộp các dòng trống thừa."""
    from markdownify import markdownify

    markdown = markdownify(str(element), heading_style="ATX", strip=["img"])
    return re.sub(r"\n{3,}", "\n\n", markdown).strip()


def _clean_html_to_article(html: str, url: str) -> dict:
    """
    Bóc tiêu đề + nội dung chính từ HTML thô, trả về markdown.

    Vứt nav/footer/script để chunk ở Task 4 không bị nhiễu bởi menu lặp lại
    trên mọi trang — nhiễu này làm BM25 (Task 6) đánh giá sai độ liên quan.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "form", "noscript", "iframe", "svg"]):
        tag.decompose()

    # AEM dựng menu bằng <div>/<ul> chứ không dùng thẻ <nav>, nên vòng decompose
    # ở trên không bắt được. Phải quét thêm theo tên class/id, nếu không mọi trang
    # sẽ mở đầu bằng "SKIP TO CONTENT" + toàn bộ menu — đoạn text giống hệt nhau
    # trên mọi file, làm BM25 ở Task 6 chấm sai và tốn chunk vô ích ở Task 4.
    for tag in soup.find_all(["div", "section", "ul"]):
        # decompose() một khối cha làm rỗng luôn các thẻ con còn nằm trong danh
        # sách này — đụng vào chúng sẽ nổ AttributeError. Bỏ qua thẻ đã bị huỷ.
        if tag.decomposed:
            continue
        signature = " ".join(tag.get("class") or []) + " " + (tag.get("id") or "")
        if NAV_CLASS_PATTERN.search(signature.lower()):
            tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(strip=True)

    published = ""
    meta_date = soup.find("meta", attrs={"property": "article:published_time"})
    if meta_date and meta_date.get("content"):
        published = meta_date["content"]

    # rmit.edu.vn chạy Adobe AEM: KHÔNG có <main>, không có #content, và 2 thẻ
    # <article> trên trang lại là card "tin liên quan" ở sidebar. Bám theo chuỗi
    # main/article/body thông thường sẽ lấy nhầm sidebar thay vì nội dung bài.
    #
    # Tầng 1: nội dung bài nằm rải trong nhiều div.text-component-inner — gộp lại
    #         cho ra thân bài sạch nhất (không dính teaser tin liên quan).
    # Tầng 2: các trang dịch vụ (scholarships, enrolment, support) dùng component
    #         khác nên tầng 1 ra quá ít; lùi về div.root — container gốc của AEM.
    markdown = ""
    blocks = soup.select("div.text-component-inner")
    if blocks:
        markdown = "\n\n".join(_to_markdown(b) for b in blocks)

    if len(markdown) < MIN_BODY_CHARS:
        # Một trang AEM có nhiều div.root song song (header / nội dung / footer),
        # nên select_one() sẽ vớ phải khối header rỗng. Lấy khối nhiều chữ nhất.
        roots = soup.select("div.root")
        best = max(roots, key=lambda r: len(r.get_text(" ", strip=True)), default=None)
        markdown = _to_markdown(best or soup.body or soup)

    return {
        "url": url,
        "title": title or "Unknown",
        "date_published": published,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": markdown,
    }


def crawl_article_static(url: str) -> dict:
    """Crawl bằng requests + BeautifulSoup (fallback khi chưa có Crawl4AI)."""
    import requests

    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    # Decode thẳng từ bytes: rmit.edu.vn khai báo charset=utf-8 và byte đúng utf-8.
    # Không dùng response.apparent_encoding — chardet đoán mò, có thể đoán ra
    # cp1252/Windows-1254 rồi làm hỏng dấu tiếng Việt của trang vốn đã đúng.
    html = response.content.decode("utf-8", errors="replace")

    return _clean_html_to_article(html, url)


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài viết và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler(verbose=False) as crawler:
        result = await crawler.arun(url=url)

        if not result.success:
            raise RuntimeError(f"Crawl4AI thất bại: {result.error_message}")

        # crawl4ai >=0.4 trả về MarkdownGenerationResult thay vì str.
        markdown = getattr(result.markdown, "raw_markdown", result.markdown)

        metadata = result.metadata or {}
        return {
            "url": url,
            "title": metadata.get("title", "Unknown"),
            "date_published": metadata.get("article:published_time", ""),
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": str(markdown).strip(),
        }


def crawl4ai_available() -> bool:
    """
    Crawl4AI dùng được chưa?

    `import crawl4ai` chạy được KHÔNG có nghĩa là crawl được: pip install crawl4ai
    không tải browser binary, phải chạy thêm `playwright install chromium`. Thiếu
    bước đó thì mọi lần arun() đều nổ "Executable doesn't exist" — nên phải kiểm
    tra cả browser, nếu không script sẽ chọn backend hỏng và trả về 0 bài.
    """
    try:
        import crawl4ai  # noqa: F401
    except ImportError:
        return False

    # Kiểm tra thư mục cache của playwright thay vì gọi sync_playwright() — gọi
    # driver thật chỉ để hỏi 1 đường dẫn sẽ in ra một đống log teardown nhiễu.
    override = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    candidates = [Path(override)] if override else [
        Path.home() / "AppData" / "Local" / "ms-playwright",   # Windows
        Path.home() / "Library" / "Caches" / "ms-playwright",  # macOS
        Path.home() / ".cache" / "ms-playwright",              # Linux
    ]
    return any(any(d.glob("chromium*")) for d in candidates if d.is_dir())


# =============================================================================
# ORCHESTRATION
# =============================================================================

def slugify(url: str) -> str:
    """Lấy phần cuối URL làm tên file cho dễ truy vết nguồn khi trích dẫn."""
    slug = urlparse(url).path.rstrip("/").split("/")[-1] or "index"
    slug = re.sub(r"[^a-z0-9-]+", "-", slug.lower()).strip("-")
    return slug[:60]


async def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLE_URLS + SERVICE_PAGES."""
    setup_directory()

    news_urls = ARTICLE_URLS
    if not news_urls:
        print(f"Đang lấy danh sách bài viết từ {NEWS_INDEX_URL} ...")
        news_urls = discover_article_urls()
        print(f"  ✓ Tìm được {len(news_urls)} bài viết\n")

    urls = news_urls + SERVICE_PAGES
    use_crawl4ai = crawl4ai_available()
    print(f"Backend: {'Crawl4AI' if use_crawl4ai else 'requests + BeautifulSoup (fallback)'}\n")

    saved = 0
    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] Crawling: {url}")
        try:
            if use_crawl4ai:
                article = await crawl_article(url)
            else:
                article = crawl_article_static(url)
        except Exception as e:
            # Một URL hỏng (404/timeout) không được làm chết cả mẻ crawl.
            print(f"  ✗ Bỏ qua ({type(e).__name__}: {e})")
            continue

        if len(article["content_markdown"]) < MIN_SAVE_CHARS:
            print(f"  ✗ Bỏ qua: nội dung quá ngắn ({len(article['content_markdown'])} ký tự)")
            continue

        filename = f"article_{i:02d}_{slugify(url)}.json"
        filepath = DATA_DIR / filename
        # encoding="utf-8" bắt buộc: mặc định của Windows là cp1252, sẽ nổ
        # UnicodeEncodeError với tiếng Việt — và test đọc lại file bằng utf-8.
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        saved += 1
        print(f"  ✓ Saved: {filepath.name} ({len(article['content_markdown'])} ký tự)")

        time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\n{'='*60}")
    print(f"Hoàn tất: {saved}/{len(urls)} bài đã lưu vào {DATA_DIR}")
    if saved < 5:
        print("⚠ Chưa đủ 5 bài — kiểm tra kết nối mạng hoặc bổ sung URL vào ARTICLE_URLS")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
