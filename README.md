# Viral Content Intelligence — Hệ thống Dự đoán Tiềm năng Triệu View & Tối ưu hóa Chiến lược Phát hành Video YouTube

> Đồ án Module 3 — Xây dựng hệ thống Machine Learning thực chiến (End-to-End ML Pipeline & Explainable AI)

Hệ thống **End-to-End Machine Learning** đóng vai trò như một **"Cố vấn Nội dung AI" (AI Pre-Publish Consultant)** dành cho các nhà sáng tạo nội dung (YouTubers) và đội ngũ Digital Marketing tại Việt Nam. Dựa trên siêu dữ liệu (metadata) trước thời điểm đăng tải (tiêu đề, mô tả, thẻ tags, thời lượng, danh mục, khung giờ phát hành và quy mô kênh), hệ thống dự đoán xác suất video trở thành **Viral Hit**, đồng thời sử dụng **SHAP & LIME** để giải thích mức độ đóng góp của từng yếu tố và gợi ý tối ưu hóa tiêu đề/cấu hình phát hành.

---

## 1. Thành viên và phân công

Nhóm áp dụng mô hình **3 nhánh Modeling độc lập (Parallel End-to-End Modeling Tracks)**: mỗi thành viên phụ trách mô hình sẽ tự thực hiện trọn vẹn từ **EDA $\rightarrow$ Preprocessing $\rightarrow$ Feature Engineering $\rightarrow$ Modeling & Optuna Tuning $\rightarrow$ Error Analysis** sao cho phù hợp nhất với bản chất toán học (Inductive Bias) của thuật toán mình đảm nhận, sau đó chấm điểm Benchmark công bằng trên cùng tập `youtube_test.csv`.

| Họ tên | MSSV | Vai trò | Phụ trách chi tiết |
|---|---|---|---|
| **Nguyễn Hoàng Long** | [MSSV] | **Leader / Data & XAI** | Thu thập & chuẩn hóa dữ liệu (`data/collection/`), Explainability SHAP/LIME (`05_Explainability.ipynb`), điều phối, review PR, báo cáo, slide, README |
| **Thịnh** | [MSSV] | **Backend / Frontend** | Xây dựng REST API với FastAPI (`backend/`), giao diện Web App với Streamlit (`frontend/`), tích hợp mô hình & SHAP, kiểm thử demo |
| **[Thành viên 3]** | [MSSV] | **ML Engineer (Track 1)** | Tự thực hiện EDA, Preprocessing, Feature Engineering (nhấn mạnh Feature Selection & SVD), huấn luyện & tuning **Random Forest** bằng Optuna, đánh giá & Error Analysis (Confusion Matrix) |
| **[Thành viên 4]** | [MSSV] | **ML Engineer (Track 2)** | Tự thực hiện EDA, Preprocessing, Feature Engineering (nhấn mạnh Interaction Ratios & Cyclical Time), huấn luyện & tuning **XGBoost** bằng Optuna, đánh giá & Error Analysis (Confusion Matrix) |
| **[Thành viên 5]** | [MSSV] | **ML Engineer (Track 3)** | Tự thực hiện EDA, Preprocessing, Feature Engineering (nhấn mạnh Native Categorical & TF-IDF N-grams), huấn luyện & tuning **LightGBM** bằng Optuna, đánh giá & Error Analysis (Confusion Matrix) |

---

## 2. Bài toán và dữ liệu

- **Loại bài toán:** Phân loại nhị phân (**Binary Classification**) — Dự đoán một video có bùng nổ thành **Viral Hit** (`is_viral = 1`) hay không (`is_viral = 0`) trước khi bấm nút đăng tải.
- **Nguồn dữ liệu:** **Google YouTube Data API v3** (`channels`, `playlistItems`, `videos` endpoints), tự thu thập bằng script [`data/collection/crawl_youtube.py`](data/collection/crawl_youtube.py) và chuẩn hóa chia tập bằng [`data/collection/prepare_dataset_split.py`](data/collection/prepare_dataset_split.py).
- **Quy mô dữ liệu:** **38.823 videos** hợp lệ từ **223 kênh YouTube Việt Nam** trải đều 7 lĩnh vực (Công nghệ, Gaming, Ẩm thực/Du lịch, Giáo dục/Tài chính, Giải trí/Phim ảnh, Âm nhạc, Đời sống/Thể thao), gồm **41 cột thuộc tính**:
  - Tập huấn luyện (`data/raw/youtube_train.csv` — 80%): **31.058 dòng** (`18.36%` Viral Hit | `81.64%` Normal).
  - Tập kiểm thử (`data/raw/youtube_test.csv` — 20%): **7.765 dòng** (`18.57%` Viral Hit | `81.43%` Normal).
- **Thời điểm thu thập:** Tháng 11/2026 (lọc bỏ các video mới đăng dưới 48 giờ để đảm bảo lượt xem đã ổn định).
- **Biến mục tiêu:** `is_viral` (`0` hoặc `1`), được định nghĩa theo **Bộ tiêu chí Viral 3 lớp (3-Criteria Viral Definition)**:
  1. **Đột phá nội bộ kênh (Relative Breakout):** $\text{viral\_ratio} = \frac{\text{view\_count}}{\text{channel\_median\_views}} \ge 2.0$ *(lượt xem gấp ít nhất 2 lần mức trung vị của chính kênh đó; `channel_median_views` chỉ được tính trên tập Train để triệt tiêu Data Leakage)*.
  2. **Độ phủ tối thiểu (Minimum Reach Floor):** $\text{view\_count} \ge 10.000$ lượt xem *(loại bỏ nhiễu biến động nhỏ ở các kênh siêu nhỏ)*.
  3. **Tương tác tự nhiên (Organic Engagement):** $\text{engagement\_rate} = \frac{\text{like\_count} + \text{comment\_count}}{\text{view\_count}} \ge 0.00180$ *(vượt ngưỡng sàn 5th percentile của tập Train nhằm loại bỏ các video chạy quảng cáo lấy view ảo không có tương tác)*.

> **Quy tắc chống Rò rỉ dữ liệu (Zero Data Leakage):** Khi huấn luyện mô hình, bắt buộc loại bỏ 8 cột hậu nghiệm chỉ xuất hiện sau khi đăng video khỏi ma trận đặc trưng `X`:
> `['view_count', 'like_count', 'comment_count', 'channel_median_views', 'viral_ratio', 'viral_score_log', 'engagement_rate', 'is_viral']`.

---

## 3. Kết quả chính (Benchmark trên tập `youtube_test.csv`)

| Mô hình | ROC-AUC | PR-AUC | F1-Score | Precision | Recall | Ghi chú |
|---|:---:|:---:|:---:|:---:|:---:|---|
| Baseline (`LogisticRegression`) | *[待 điền]* | *[待 điền]* | *[待 điền]* | *[待 điền]* | *[待 điền]* | Mô hình cơ sở tham chiếu |
| **Random Forest** *(Optuna tuned)* | *[待 điền]* | *[待 điền]* | *[待 điền]* | *[待 điền]* | *[待 điền]* | Nhánh ML 1 (Bagging) |
| **XGBoost** *(Optuna tuned)* | *[待 điền]* | *[待 điền]* | *[待 điền]* | *[待 điền]* | *[待 điền]* | Nhánh ML 2 (Depth-wise Boosting) |
| **LightGBM** *(Optuna tuned)* | *[待 điền]* | *[待 điền]* | *[待 điền]* | *[待 điền]* | *[待 điền]* | Nhánh ML 3 (Leaf-wise Boosting) |
| **[Mô hình tốt nhất]** | **[待 điền]** | **[待 điền]** | **[待 điền]** | **[待 điền]** | **[待 điền]** | **Mô hình đóng gói triển khai (`best_model.joblib`)** |

**Phương pháp phân chia & đánh giá:**
- Chia tập Train/Test (80/20) bằng **Phân tầng kép (Dual Stratification)** theo tổ hợp `channel_id` $\times$ `is_viral` kết hợp **Stratified K-Fold Cross-Validation** trên tập Train — đảm bảo cả 223 kênh đều xuất hiện đúng tỷ lệ 80/20 ở cả hai tập và giữ nguyên phân phối nhãn `~18.4% Viral` trên từng kênh.

---

## 4. Cấu trúc thư mục

```text
.
├── data/
│   ├── raw/
│   │   ├── youtube_train.csv             # Tập huấn luyện chuẩn (31.058 dòng x 41 cột)
│   │   └── youtube_test.csv              # Tập kiểm thử cố định (7.765 dòng x 41 cột)
│   ├── processed/                        # Dữ liệu sau tiền xử lý & tạo đặc trưng
│   └── collection/
│       ├── crawl_youtube.py              # Script thu thập dữ liệu qua YouTube Data API v3
│       └── prepare_dataset_split.py      # Script gán nhãn Viral 3 lớp & chia Stratified Train/Test
├── notebooks/
│   ├── 01_EDA.ipynb                      # Khám phá & trực quan hóa dữ liệu
│   ├── 02_Preprocessing.ipynb            # Làm sạch & chuẩn hóa dữ liệu
│   ├── 03_Feature_Engineering.ipynb      # Trích xuất đặc trưng NLP, Temporal, Channel & Metadata
│   ├── 04_Modeling.ipynb                 # Huấn luyện, Optuna tuning, Benchmark & Error Analysis
│   └── 05_Explainability.ipynb           # Giải thích mô hình với SHAP & LIME
├── backend/                              # FastAPI Server
│   ├── main.py                           # Endpoints /health, /predict
│   ├── schemas.py                        # Pydantic validation schemas
│   ├── model/best_model.joblib           # Pipeline mô hình tốt nhất đã huấn luyện
│   └── requirements.txt
├── frontend/                             # Giao diện người dùng Streamlit
│   ├── app.py                            # AI YouTube Viral Content Studio
│   └── requirements.txt
├── report/Final_Report.pdf               # Báo cáo tổng kết đồ án
├── presentation/Demo_Slides.pdf          # Slide thuyết trình bảo vệ
├── .env.example                          # Mẫu cấu hình YOUTUBE_API_KEY
├── pyproject.toml                        # Quản lý dependencies với uv
└── uv.lock
```

---

## 5. Cài đặt & Thu thập dữ liệu

Yêu cầu: **Python >= 3.12** và **[uv](https://docs.astral.sh/uv/)**.

```bash
git clone https://github.com/Long-bit-lang/ProjectModule3.git
cd Module3
uv sync --all-groups
```

*(Tùy chọn)* Nếu muốn tự chạy lại quá trình thu thập dữ liệu từ YouTube Data API v3:

```bash
cp .env.example .env
# Điền YOUTUBE_API_KEY vào file .env
uv run python data/collection/crawl_youtube.py --videos-per-channel 200 --discover
uv run python data/collection/prepare_dataset_split.py
```

---

## 6. Chạy hệ thống

**Bước 1 — Khởi động Backend (FastAPI)** tại thư mục gốc:

```bash
uv run --group backend uvicorn backend.main:app --reload --port 8000
```

Kiểm tra trạng thái server tại http://localhost:8000/health và tài liệu Swagger UI tại http://localhost:8000/docs.

**Bước 2 — Khởi động Frontend (Streamlit)** trên một terminal khác:

```bash
uv run --group frontend streamlit run frontend/app.py
```

Giao diện ứng dụng chạy tại http://localhost:8501.

**Chạy Jupyter Notebooks:**

```bash
uv run jupyter lab
```

---

## 7. Đặc tả API (API Specification)

### `GET /health`

Kiểm tra trạng thái server và mô hình đã nạp vào bộ nhớ.

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `POST /predict`

Dự đoán xác suất video đạt **Viral Hit** và trả về các đặc trưng đóng góp lớn nhất (SHAP contributions).

**Request:**

```json
{
  "title": "Top 5 Bí Mật Đằng Sau Thuật Toán YouTube 2026 Mà Không Ai Nói Cho Bạn Biết!",
  "description": "Khám phá cơ chế đề xuất video của YouTube kèm hướng dẫn chi tiết từng bước...",
  "tags": "youtube|thuat toan youtube|kiem tien youtube|sang tao noi dung|ai",
  "category_id": 28,
  "duration_sec": 745,
  "publish_hour": 19,
  "publish_dayofweek": 5,
  "definition": "hd",
  "caption": true,
  "channel_subscribers": 150000,
  "channel_total_views": 28000000,
  "channel_video_count": 180
}
```

**Response:**

```json
{
  "prediction": 1,
  "label": "Viral Hit",
  "viral_probability": 0.842,
  "top_features": [
    { "feature": "is_prime_time", "contribution": 0.145 },
    { "feature": "title_has_number", "contribution": 0.118 },
    { "feature": "duration_sec", "contribution": 0.092 },
    { "feature": "tag_count", "contribution": -0.041 }
  ]
}
```

---

## 8. Quy ước làm việc nhóm

- Nhánh `main` luôn ở trạng thái chạy ổn định; mọi thay đổi đi qua Pull Request và cần ít nhất 1 thành viên review.
- Mỗi thành viên phát triển trên nhánh riêng: `feature/<ten-tinh-nang>` (ví dụ: `feature/model-random-forest`, `feature/model-xgboost`, `feature/model-lightgbm`, `feature/shap-explainability`, `feature/fastapi-streamlit`); merge xong thì xóa nhánh.
- Commit tuân theo chuẩn Conventional Commits: `feat:`, `fix:`, `docs:`, `chore:` kèm mô tả ngắn gọn.
- Cả 3 nhánh mô hình sử dụng cố định tập `data/raw/youtube_train.csv` và `data/raw/youtube_test.csv` để đảm bảo kết quả Benchmark nhất quán tuyệt đối.

---

## 9. Hạn chế và hướng phát triển

- **Hạn chế hiện tại:**
  - Mô hình mới khai thác siêu dữ liệu văn bản, thời gian và thông số kỹ thuật, chưa trích xuất trực tiếp đặc trưng thị giác từ ảnh bìa (`thumbnail_url`) hay nhịp độ âm thanh/video.
  - Chưa theo dõi chuỗi thời gian tốc độ tăng trưởng lượt xem trong 1–3 giờ đầu tiên sau khi đăng (early velocity signals).
- **Hướng phát triển:**
  - Tích hợp mô hình Computer Vision (CLIP / ResNet) để chấm điểm độ thu hút của ảnh Thumbnail (màu sắc, khuôn mặt biểu cảm, chữ trên ảnh).
  - Sử dụng PhoBERT để trích xuất ngữ nghĩa sâu tiếng Việt từ `title` và `description` kết hợp với LightGBM/XGBoost (Multimodal Stacking).
  - Tích hợp LLM tạo sinh tự động viết lại 5 phương án tiêu đề có điểm SHAP dự đoán Viral cao nhất cho người dùng lựa chọn.

---

## 10. Tài liệu tham khảo

1. Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions (SHAP)*. Advances in Neural Information Processing Systems (NeurIPS).
2. Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. Advances in Neural Information Processing Systems (NeurIPS).
3. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
4. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier (LIME)*. ACM SIGKDD.
5. Google Developers (2026). *YouTube Data API v3 Reference Documentation*. https://developers.google.com/youtube/v3
