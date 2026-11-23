# [TÊN ĐỀ TÀI]

> Đồ án Module 3 — Xây dựng hệ thống Machine Learning thực chiến (End-to-End ML Pipeline)

[Mô tả 2–3 câu: bài toán nghiệp vụ là gì, dự đoán/phân cụm/phát hiện cái gì, dành cho ai.]

## 1. Thành viên và phân công

| Họ tên | MSSV | Vai trò | Phụ trách |
|---|---|---|---|
| Nguyễn Hoàng Long | [MSSV] | Leader | Explainability SHAP/LIME (notebook 05), điều phối, review, báo cáo, slide, README |
| [Tên] | [MSSV] | Data Engineer | Thu thập dữ liệu (`data/collection/`), EDA (notebook 01) |
| [Tên] | [MSSV] | Data Analyst | Preprocessing, Feature Engineering (notebook 02, 03) |
| [Tên] | [MSSV] | ML Engineer | Modeling, tuning, đánh giá, Error Analysis (notebook 04) |
| [Tên] | [MSSV] | Backend / Frontend | FastAPI, giao diện, tích hợp, test demo |

## 2. Bài toán và dữ liệu

- **Loại bài toán:** [Regression / Classification / Clustering / Anomaly Detection]
- **Nguồn dữ liệu:** [Chợ Tốt / TopCV / Open-Meteo / Google Forms ...], tự thu thập bằng script trong `data/collection/`
- **Quy mô:** [N] dòng, [M] đặc trưng
- **Thời điểm thu thập:** [dd/mm/yyyy – dd/mm/yyyy]
- **Biến mục tiêu:** `[ten_cot]`

## 3. Kết quả chính

| Mô hình | [Metric 1] | [Metric 2] | Ghi chú |
|---|---|---|---|
| Baseline ([mean / Linear / ...]) | | | |
| [Mô hình 2] | | | |
| [Mô hình 3] | | | |
| **[Mô hình tốt nhất] (sau tuning)** | | | Mô hình triển khai |

Phương pháp đánh giá: [Holdout / Stratified K-Fold / TimeSeriesSplit] — lý do: [1 câu].

## 4. Cấu trúc thư mục

```
.
├── data/
│   ├── raw/                  # Dữ liệu thô tự thu thập
│   ├── processed/            # Dữ liệu sau tiền xử lý
│   └── collection/           # Script thu thập dữ liệu
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Modeling.ipynb
│   └── 05_Explainability.ipynb
├── backend/                  # FastAPI
│   ├── main.py               # Endpoints /health, /predict
│   ├── schemas.py            # Pydantic schema
│   ├── model/best_model.joblib
│   └── requirements.txt
├── frontend/                 # Streamlit / Gradio
│   ├── app.py
│   └── requirements.txt
├── report/Final_Report.pdf
├── presentation/Demo_Slides.pdf
├── pyproject.toml
└── uv.lock
```

## 5. Cài đặt

Yêu cầu: [Python 3.11](https://www.python.org/) và [uv](https://docs.astral.sh/uv/).

```bash
git clone [URL-REPO]
cd [TEN-THU-MUC]
uv sync --all-groups
```

## 6. Chạy hệ thống

**Bước 1 — Backend (FastAPI)**, chạy ở thư mục gốc:

```bash
uv run --group backend uvicorn backend.main:app --reload --port 8000
```

Kiểm tra: mở http://localhost:8000/health và tài liệu API tại http://localhost:8000/docs

**Bước 2 — Frontend (Streamlit)**, mở terminal khác:

```bash
uv run --group frontend streamlit run frontend/app.py
```

Giao diện chạy tại http://localhost:8501

**Chạy lại notebook:**

```bash
uv run jupyter lab
```

Chạy lần lượt notebook từ `01` đến `05`.

## 7. API

### `GET /health`

Kiểm tra server và mô hình đã nạp.

```json
{ "status": "ok", "model_loaded": true }
```

### `POST /predict`

Request:

```json
{
  "[feature_1]": "[giá trị]",
  "[feature_2]": 0
}
```

Response:

```json
{
  "prediction": 0,
  "confidence": 0.0,
  "top_features": [
    { "feature": "[ten]", "contribution": 0.0 }
  ]
}
```

## 8. Quy ước làm việc nhóm

- Nhánh `main` luôn chạy được; mọi thay đổi đi qua Pull Request và cần 1 người review.
- Mỗi người làm trên nhánh riêng: `feature/<ten-tinh-nang>`; merge xong thì xóa nhánh.
- Commit theo dạng: `feat:`, `fix:`, `docs:`, `chore:` kèm mô tả ngắn.
- Mỗi notebook chỉ một người chỉnh sửa tại một thời điểm để tránh xung đột.

## 9. Hạn chế và hướng phát triển

- [Hạn chế 1]
- [Hướng phát triển 1]

## 10. Tài liệu tham khảo

1. [Tác giả, tiêu đề, năm]
2. [Tác giả, tiêu đề, năm]
3. [Tác giả, tiêu đề, năm]