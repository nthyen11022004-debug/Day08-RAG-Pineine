# RAG Evaluation Results

## Framework sử dụng

> Bộ đo token-overlap tự viết (`--framework local`) — chạy offline, không tốn chi phí LLM. Dùng để lặp nhanh khi phát triển; điểm chính thức lấy từ `--framework deepeval`.

> Retrieval gọi thẳng `src.task9_retrieval_pipeline.retrieve()` và câu trả lời sinh bằng đúng `SYSTEM_PROMPT` + tham số của Task 10, nên điểm phản ánh đúng hệ thống đem đi demo chứ không phải một bản mô phỏng.

> Ngưỡng fallback dùng trong đo: `0.54` (giá trị đã hiệu chỉnh trên corpus thật ở Task 9, không phải số mẫu).

---

## Overall Scores

| Metric | A - hybrid (semantic + BM25 + RRF) | B - dense only (semantic) | Δ |
|--------|---------------------------|----------------------|---|
| Faithfulness | 0.630 | 0.593 | +0.037 |
| Answer Relevance | 0.623 | 0.568 | +0.054 |
| Context Recall | 0.706 | 0.607 | +0.099 |
| Context Precision | 0.346 | 0.312 | +0.034 |
| Average | 0.576 | 0.520 | +0.056 |

---

## A/B Comparison Analysis

**Config A:** A - hybrid (semantic + BM25 + RRF)
> Semantic search (ChromaDB, cosine) + BM25 gộp bằng Reciprocal Rank Fusion (k=60). Chunk nào xuất hiện ở cả hai danh sách mới được cộng dồn thứ hạng.

**Config B:** B - dense only (semantic)
> Chỉ semantic search, tắt hẳn nhánh BM25. Đây là đối chứng để trả lời câu hỏi: BM25 có đóng góp thật hay chỉ xáo lại cùng một tập bằng chứng?

**Kết luận:**
> **A - hybrid (semantic + BM25 + RRF)** tốt hơn, chênh 0.056 điểm trung bình. Khác biệt lớn nhất nằm ở **Context Recall** (+0.099), tức việc bổ sung BM25 tác động chủ yếu lên khâu chọn đúng bằng chứng.

---

## Worst Performers (Bottom 3)

> Xếp hạng gộp cả hai config, mỗi câu hỏi chỉ lấy lần chạy tệ nhất. Cột Config cho biết điểm đó đến từ đâu — thiếu cột này thì không biết câu hỏi kém do bản thân nó khó hay do config yếu.

| # | Config | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|--------|----------|-------------|-----------|--------|---------------|------------|
| 1 | B | Hồ sơ nhập học cần chuẩn bị gì về bằng tốt nghiệp và học bạ? | 0.141 | 0.000 | 0.198 | Từ chối trả lời (thiếu bằng chứng) | Điểm cosine không vượt ngưỡng fallback và PageIndex cũng không tìm được mục phù hợp, nên hệ thống từ chối thay vì bịa — đúng hành vi mong muốn. Cần kiểm tra corpus có thực sự chứa thông tin này bằng tiếng Việt không. |
| 2 | A | Thư viện RMIT mở cửa vào giờ nào trong học kỳ? | 0.136 | 0.000 | 0.273 | Từ chối trả lời (thiếu bằng chứng) | Điểm cosine không vượt ngưỡng fallback và PageIndex cũng không tìm được mục phù hợp, nên hệ thống từ chối thay vì bịa — đúng hành vi mong muốn. Cần kiểm tra corpus có thực sự chứa thông tin này bằng tiếng Việt không. |
| 3 | B | Phương thức thanh toán trực tuyến nào được khuyến nghị tại RMIT? | 0.436 | 0.731 | 0.232 | Retriever missed evidence | Query wording likely diverged from the source phrasing. |

> **Lưu ý khi đọc Relevance = 0.000:** metric này đo độ trùng token giữa câu hỏi và câu trả lời, nên một câu từ chối đúng đắn ("Tôi không thể xác minh thông tin này") vẫn bị chấm 0 vì không dùng lại từ nào của câu hỏi. Đó là giới hạn của metric, không phải hệ thống trả lời sai.

---

## Recommendations

### Cải tiến 1 — Bịt lỗ hổng dữ liệu tiếng Việt
**Action:** 3/30 lượt chạy bị từ chối vì không đủ bằng chứng. Nguyên nhân đã xác định: một số trang dịch vụ (rõ nhất là thư viện, `/libraryvn`) chỉ có bản tiếng Anh nên câu hỏi tiếng Việt không kéo nổi chúng lên trên ngưỡng cosine. Bổ sung nguồn tiếng Việt cho các mảng này, hoặc dịch sẵn phần tiêu đề/tóm tắt trước khi index.

**Expected impact:** Giảm số câu bị từ chối, tăng cả Context Recall lẫn Answer Relevance.

### Cải tiến 2 — Lọc nhiễu tốt hơn
**Action:** Thử reranking mạnh hơn hoặc loại duplicate chunks trước khi sinh câu trả lời.

**Expected impact:** Giảm context thừa và tăng tỉ lệ evidence hữu ích trong prompt.

### Cải tiến 3 — Giữ hybrid làm mặc định
**Action:** Config `A - hybrid (semantic + BM25 + RRF)` dẫn đầu `B - dense only (semantic)` +0.056 điểm trung bình, và thắng ở cả bốn metric. Giữ hybrid làm cấu hình mặc định khi demo.

**Expected impact:** Chốt được lựa chọn kiến trúc bằng số liệu thay vì cảm tính.
