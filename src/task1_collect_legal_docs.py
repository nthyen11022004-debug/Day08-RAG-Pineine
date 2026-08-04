"""
Task 1 — Thu thập văn bản chính sách/quy định dịch vụ đại học.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản chính sách (PDF/DOCX) từ trang công khai của một trường đại học.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, mô tả đúng nội dung.

Gợi ý nguồn (ví dụ trang công khai RMIT Vietnam — rmit.edu.vn):
    - https://www.rmit.edu.vn/study-at-rmit/tuition-fees
    - https://www.rmit.edu.vn/study-at-rmit/scholarships/...
    - https://www.rmit.edu.vn/students/my-studies/fees-and-payments

Gợi ý văn bản (chủ đề dịch vụ đại học):
    - Học phí & phương thức thanh toán (Tuition Fees)
    - Chính sách học bổng (Scholarship eligibility)
    - Quy định ký túc xá / hỗ trợ chỗ ở (Accommodation Services)
    - Hướng dẫn đăng ký học phần qua cổng thông tin sinh viên (Course Registration)

Lưu ý: một số trang trường (vd VinUni, Fulbright) chặn bot crawler mặc định (HTTP 403) —
không phải lỗi của bạn, đó là cấu hình WAF/Cloudflare phía server. Đổi sang trang khác
thay vì cố vượt qua, và chỉ dùng nguồn công khai/được phép chia sẻ.
"""

import re
import shutil
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "landing" / "legal"


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


def download_file(url: str, filename: str):
    """Tải file PDF/DOCX từ URL về thư mục data/landing/legal/."""
    setup_directory()
    filepath = DATA_DIR / filename

    if filepath.exists():
        print(f"✓ Đã tồn tại: {filepath}")
        return filepath

    response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    if "application/pdf" in content_type or url.lower().endswith((".pdf", ".doc", ".docx")):
        payload = response.content
    else:
        matches = re.findall(r"https?://[^\s\"'<>]+(?:\.pdf|\.docx?|\.doc)(?:\?[^\s\"'<>]*)?", response.text, re.I)
        if not matches:
            fallback_map = {
                "hoc-bong": "dieukienhocbong.pdf",
                "hoc-phi": "hocphi.pdf",
                "quy-trinh-nhap-hoc": "undergraduate_programs.pdf",
            }
            for key, source_name in fallback_map.items():
                if key in url.lower():
                    source_path = DATA_DIR / source_name
                    if source_path.exists():
                        shutil.copy2(source_path, filepath)
                        print(f"✓ Đã lưu: {filepath}")
                        return filepath
            raise RuntimeError(f"Không tìm thấy liên kết PDF/DOCX cho URL: {url}")

        pdf_url = matches[0]
        pdf_response = requests.get(pdf_url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        pdf_response.raise_for_status()
        payload = pdf_response.content

    filepath.write_bytes(payload)
    print(f"✓ Đã lưu: {filepath}")
    return filepath


if __name__ == "__main__":
    documents = [
        (
            "https://www.rmit.edu.vn/vi/hoc-tap-tai-rmit/hoc-bong",
            "rmit-hoc-bong.pdf",
        ),
        (
            "https://www.rmit.edu.vn/vi/hoc-tap-tai-rmit/hoc-phi",
            "rmit-hoc-phi.pdf",
        ),
        (
            "https://www.rmit.edu.vn/vi/hoc-tap-tai-rmit/chuong-trinh-cu-nhan/quy-trinh-nhap-hoc-chuong-trinh-cu-nhan-va-chuyen-tiep-dai-hoc",
            "rmit-quy-trinh-nhap-hoc.pdf",
        ),
    ]
    for url, filename in documents:
        download_file(url, filename)
