# Bài Tập Nhóm — University Services RAG Chatbot

## Mục Tiêu

Sau khi hoàn thành bài cá nhân, nhóm ngồi lại để xây dựng **1 trong 2 sản phẩm**:

---

## Yêu cầu 1: Sản phẩm nhóm RAG Chatbot

Chatbot trả lời câu hỏi về dịch vụ và chính sách của RMIT Việt Nam, dữ liệu crawl từ
trang công khai bản tiếng Việt (`rmit.edu.vn/vi`).

| Yêu cầu | Trạng thái | Ghi chú |
|---|---|---|
| Giao diện chat | ✅ | Streamlit — `app.py` |
| Trả lời có citation | ✅ | Định dạng `[Document N]`, sinh bởi Task 10 |
| Hiển thị source documents | ✅ | Kèm điểm số, đường dẫn mục, **bôi vàng từ khoá** trong trích đoạn |
| Follow-up questions (conversation memory) | ❌ | **Chưa làm** — thuộc phần Bonus, `generate_with_citation()` hiện chưa nhận lịch sử hội thoại |

**Điểm nhấn của UI** — 3 thứ bản starter không có:

1. **Badge `hybrid` / `pageindex`** hiển thị đường retrieval đã đi qua. Đây là bằng chứng
   trực quan cho thấy logic fallback của Task 9 chạy thật, chứ không phải chatbot cứ có
   gì trả nấy.
2. **Ghi rõ thang điểm.** Điểm RRF (~0.016) và điểm rank của PageIndex (~1.0) khác thang
   nhau; hiển thị số trần sẽ khiến người xem tưởng kết quả tệ.
3. **Báo lỗi chỉ đúng task.** `NotImplementedError("Implement retrieve")` hiện thành
   *"Task 9 chưa được implement — mở src/task9_retrieval_pipeline.py"* thay vì traceback.

**Stack thực tế:**
```
Streamlit (app.py) → Task 9 retrieve() → Task 10 generate_with_citation() → Display
```

---

## Yêu cầu 2: RAG Evaluation Pipeline

Sử dụng **1 trong 3 framework** sau để evaluate pipeline RAG của nhóm:

### Framework lựa chọn

| Framework | Cài đặt | Đặc điểm |
|-----------|---------|-----------|
| [DeepEval](https://github.com/confident-ai/deepeval) | `pip install deepeval` | Nhiều metric built-in, dễ integrate với pytest |
| [RAGAS](https://github.com/explodinggradients/ragas) | `pip install ragas` | Chuẩn industry cho RAG eval, 3 trục chính |
| [TruLens](https://github.com/truera/trulens) | `pip install trulens` | Dashboard UI, feedback functions mạnh |

### Yêu cầu Evaluation

1. **Tạo Golden Dataset** — tối thiểu 15 cặp Q&A (question, expected_answer, expected_context)
2. **Chạy evaluation** trên toàn bộ golden dataset với các metrics sau:
   - **Faithfulness** — câu trả lời có bám đúng context không?
   - **Answer Relevance** — câu trả lời có đúng câu hỏi không?
   - **Context Recall** — retriever có lấy đủ evidence không?
   - **Context Precision** — trong context lấy về, bao nhiêu % thực sự hữu ích?
3. **So sánh A/B** — chạy eval trên ít nhất 2 config khác nhau (ví dụ: có reranking vs không reranking, hoặc hybrid vs dense-only)
4. **Báo cáo** — bảng điểm + phân tích worst performers + đề xuất cải tiến

Xem code mẫu (DeepEval/RAGAS/TruLens) chi tiết trong `README.md` gốc mục "Yêu cầu 2".

### Deliverable Evaluation

- [x] File `group_project/evaluation/golden_dataset.json` — **15 cặp Q&A** tiếng Việt,
      `expected_context` trích nguyên văn từ corpus đã index
- [x] File `group_project/evaluation/eval_pipeline.py` — chạy được, 2 chế độ chấm điểm
- [x] File `group_project/evaluation/results.md` — bảng điểm + phân tích, **sinh tự động**
- [x] So sánh A/B: **hybrid** vs **dense-only**

### Framework đã chọn: DeepEval (không phải RAGAS)

Bốn metric lấy trực tiếp từ thư viện: `FaithfulnessMetric`, `AnswerRelevancyMetric`,
`ContextualRecallMetric`, `ContextualPrecisionMetric`.

**Vì sao không dùng RAGAS như tên checkpoint:** ở CP0 nhóm gặp `ResolutionImpossible`
khi cài `requirements.txt` (bản khi đó để `ragas==0.1.21` cạnh
`langchain-text-splitters>=0.2.0`, hai gói này yêu cầu `langchain-core` loại trừ nhau).
Để không mất thời gian, môi trường được cài vòng qua các pin — kết quả là máy chạy
`langchain-core 1.5.3` thay vì `0.2.43` như file khai báo. Trong môi trường đó RAGAS
không import nổi:

```
ragas/llms/base.py line 12:
    from langchain_community.chat_models.vertexai import ChatVertexAI
ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'
```

`langchain-community 0.4.x` đã gỡ module `vertexai`. Đã thử cả `ragas==0.4.3` và
`ragas==0.3.9` — cùng lỗi, vì nguyên nhân nằm ở `langchain-community` chứ không phải
phiên bản ragas.

> **Nói cho công bằng:** `requirements.txt` hiện đã được ghim lại thành bộ langchain
> 0.2.x, và bản `langchain-community 0.2.19` thì **vẫn còn** module `vertexai`. Nghĩa
> là dựng một venv sạch đúng theo file đó thì RAGAS chạy được. Nhóm chọn DeepEval vì
> nó cài sạch trên môi trường đang có, không phải vì RAGAS hỏng về bản chất. DeepEval
> nằm cùng danh sách framework rubric cho phép.

### Hai chế độ chấm điểm

Tầng **chấm điểm** tách rời tầng **chạy RAG**, nên cả hai chấm trên cùng một câu trả
lời và cùng một context:

| Lệnh | Cách chấm | Dùng khi |
|---|---|---|
| `--framework deepeval` (mặc định) | LLM-as-judge (`gpt-4o-mini`) | Điểm chính thức nộp bài |
| `--framework local` | Token-overlap tự viết, offline | Lặp nhanh khi phát triển, 0 chi phí |

---

## Yêu Cầu Chung

1. **Tích hợp pipeline** từ bài cá nhân của các thành viên
2. **Demo hoạt động được** trong buổi trình bày (chạy local hoặc deploy)
3. **Evaluation pipeline** chạy được và có báo cáo kết quả
4. **Code push lên repository** chung của nhóm
5. **README** mô tả kiến trúc và phân công (điền bên dưới)

---

## Kiến Trúc Hệ Thống

```
                        ┌─────────────────────────┐
   Câu hỏi tiếng Việt ──►│   app.py  (Streamlit)   │
                        └────────────┬────────────┘
                                     ▼
                    ┌────────────────────────────────┐
                    │  Task 9 — retrieve()           │
                    └────────────────────────────────┘
                         │                    │
              ┌──────────┘                    └──────────┐
              ▼                                          ▼
    Task 5  semantic_search                    Task 6  lexical_search
    ChromaDB · cosine                          BM25Okapi · k1=1.5, b=0.75
    MiniLM đa ngữ · 384 dim                    cùng 408 chunk của Task 4
              │                                          │
              └────────────────┬─────────────────────────┘
                               ▼
                  Task 7  rerank_rrf · RRF(d) = Σ 1/(60+rank)
                               │
                  ┌────────────┴─────────────┐
                  │  Ngưỡng: điểm cosine GỐC │
                  │  của Task 5, KHÔNG phải  │
                  │  điểm RRF đã fuse        │
                  └────────────┬─────────────┘
                ≥ 0.54         │         < 0.54
          ┌────────────────────┴───────────────────┐
          ▼                                        ▼
   source = "hybrid"                    Task 8  pageindex_search
                                        cây mục lục + LLM suy luận
                                        (vectorless, không embedding)
                                                   │
                                        ┌──────────┴──────────┐
                                        ▼                     ▼
                              source = "pageindex"       vẫn rỗng → []
                                        │                     │
          └─────────────────────────────┴─────────────────────┘
                               ▼
                  Task 10  generate_with_citation
                  reorder_for_llm [1,3,5,4,2] chống "lost in the middle"
                  → gpt-4o-mini (temp=0.3, top_p=0.9)
                               ▼
              Câu trả lời + citation + badge hybrid/pageindex
```

**Ba quyết định thiết kế:**

1. **Ngưỡng fallback so với điểm cosine GỐC, không phải điểm RRF.** Điểm RRF sau khi
   fuse luôn ≈ `1/(60+1) = 0.0164` bất kể nội dung có liên quan hay không — dùng nó làm
   ngưỡng thì fallback không bao giờ kích hoạt. Ngưỡng `0.54` **đo thật** trên corpus
   bằng `python -m src.task9_retrieval_pipeline --calibrate`.

2. **BM25 và semantic dùng chung một bộ chunk.** RRF gộp hai danh sách bằng khoá
   `source::chunk_index`. Nếu hai nhánh trả về đơn vị khác nhau (document vs chunk) thì
   không khoá nào trùng, RRF chỉ nối list chứ không cộng dồn thứ hạng.

3. **Corpus lấy từ bản `/vi` của rmit.edu.vn.** BM25 là lexical thuần: query "học phí"
   không thể khớp document "tuition fee", điểm luôn bằng 0. Dùng nguồn tiếng Anh là mất
   trắng một nửa hybrid search.

---

## Phân Công Công Việc

| Thành viên | MSSV | Vai trò | Nhiệm vụ | Trạng thái |
|-----------|------|---------|----------|------------|
| Nguyễn Thị Hoàng Yến | 2A202601959 | Role 1 — Team Leader & RAG Architect | Quản trị repo chung, điều phối PR & tiến độ; **Task 6** — BM25 lexical search, tách từ tiếng Việt | ✅ Xong |
| Ngô Thị Hằng | 2A202601365 | Role 2 — Data & Retrieval Specialist | **Task 1** — thu thập 3 PDF chính sách; **Task 4** — chunking & indexing ChromaDB; **Task 5** — semantic search + HyDE | ✅ Xong |
| Nguyễn Huy Hoàng | 2A202601113 | Role 3 — Frontend & Chatbot Dev | **Task 2** — crawl 16 trang tiếng Việt; **Task 3** — convert markdown; **Task 8** — vectorless RAG chạy local; **Task 9** — retrieval pipeline + hiệu chỉnh ngưỡng; `app.py` | ✅ Xong |
| Quách Xuân Trường | 2A202601371 | Role 4 — Evaluation & QA Engineer | **Task 7** — reranking RRF/MMR; **Task 10** — generation có citation; golden dataset 15 câu + `eval_pipeline.py` | ✅ Xong |

---

## Hướng Dẫn Chạy

```bash
# 1. Cài dependencies
pip install -r requirements.txt
pip install deepeval          # framework evaluation (chưa có trong requirements.txt)

# 2. Khai báo key — điền OPENAI_API_KEY hoặc OPENROUTER_API_KEY vào .env
cp .env.example .env
```

> ⚠️ **Để trống dòng key không dùng, đừng giữ chuỗi mẫu.** Code chọn provider theo thứ
> tự `OPENROUTER_API_KEY` → `OPENAI_API_KEY`; placeholder `sk-or-v1-...` vẫn là giá trị
> truthy nên sẽ được chọn trước key thật rồi lỗi 401.

```bash
# 3. Dựng vector store — BẮT BUỘC trước lần chạy đầu.
#    chroma_db/ đã được gitignore nên clone về sẽ không có sẵn.
python -m src.task4_chunking_indexing

# 4. Chạy chatbot
streamlit run app.py
```

Kiểm tra ở sidebar: *Tài liệu* = 19, *Chunk đã index* = 408. Nếu cột chunk hiện `—`
thì bước 3 chưa chạy.

```bash
# Chấm điểm pipeline Task 1-10
pytest tests/ -v

# Evaluation A/B → ghi ra group_project/evaluation/results.md
python -m group_project.evaluation.eval_pipeline                     # DeepEval
python -m group_project.evaluation.eval_pipeline --framework local   # offline, vài giây

# Đo lại ngưỡng fallback (bắt buộc khi đổi corpus hoặc embedding model)
python -m src.task9_retrieval_pipeline --calibrate
```

### Câu hỏi nên dùng khi demo

| Câu hỏi | Đường đi mong đợi |
|---|---|
| *Điều kiện để được nhận học bổng là gì?* | Badge **`hybrid`**, trả lời kèm `[Document N]` |
| *Công thức nấu phở bò gia truyền* | Badge **`pageindex`**, trả lời *"Tôi không thể xác minh thông tin này"* |

Câu thứ hai là điểm đáng chỉ ra nhất: nó chứng minh ngưỡng fallback hoạt động thật,
chatbot **từ chối** thay vì bịa ra câu trả lời từ context lạc đề.

---

## Lưu ý

Hãy giữ lại repo này nếu như bạn học track 3 giai đoạn 2, chúng ta sẽ phát triển tiếp dự án lên knowledge graph để khắc phục các câu hỏi hóc búa khi có các câu hỏi khó.
