"""
Script tạo nhãn mục tiêu chuẩn hóa (is_viral & viral_score) và chia sẵn tập Train / Test cố định.
Thiết kế chuẩn học thuật Machine Learning (Zero Data Leakage):
1. Lọc bỏ các video mới đăng < 48 giờ (chưa ổn định lượt xem) và các kênh < 10 videos.
2. Định nghĩa "Viral Hit" (is_viral = 1) theo Bộ tiêu chí 3 lớp (3-Criteria Viral Definition):
   - Điều kiện 1 (Relative Breakout): viral_ratio = view_count / channel_median_views >= 2.0
     (Lượt xem gấp ít nhất 2 lần mức trung vị bình thường của chính kênh đó).
   - Điều kiện 2 (Minimum Reach Floor): view_count >= 10,000 views
     (Chặn bẫy kênh siêu nhỏ vài trăm view).
   - Điều kiện 3 (Organic Engagement Check): engagement_rate = (like_count + comment_count) / view_count
     vượt qua ngưỡng sàn 5th percentile của tập Train (loại bỏ video chạy quảng cáo lấy view ảo không có tương tác).
3. Chia Stratified 80% Train / 20% Test theo từng cặp (channel_id, is_viral) để
   mọi kênh trong 223 kênh đều có đúng 80% video ở Train và 20% video ở Test.
4. Mọi ngưỡng thống kê (channel_median_views, engagement_floor) chỉ được tính trên tập TRAIN,
   sau đó ánh xạ sang tập TEST -> Tuyệt đối không rò rỉ thống kê từ Test sang Train!
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

VIRAL_RATIO_THRESHOLD = 2.0
MIN_VIEWS_FLOOR = 10_000
ENGAGEMENT_QUANTILE_FLOOR = 0.05


def compute_engagement_rate(df: pd.DataFrame) -> pd.Series:
    likes = df["like_count"].fillna(0).clip(lower=0)
    comments = df["comment_count"].fillna(0).clip(lower=0)
    views = df["view_count"].clip(lower=1)
    return (likes + comments) / views


def prepare_and_split(
    raw_csv_path: Path,
    output_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    if not raw_csv_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file {raw_csv_path}. Hãy chạy crawl_youtube.py trước!"
        )

    df = pd.read_csv(raw_csv_path)
    print(f"📥 Đọc dữ liệu thô ban đầu: {len(df):,} dòng, {df['channel_id'].nunique()} kênh.")

    # 1. Xử lý trùng lặp video_id, làm phẳng xuống dòng trong text và lọc giá trị bất hợp lệ
    df = df.drop_duplicates(subset=["video_id"]).copy()
    for text_col in ("title", "description", "channel_title"):
        if text_col in df.columns:
            df[text_col] = (
                df[text_col]
                .fillna("")
                .astype(str)
                .str.replace(r"[\r\n\t]+", " ", regex=True)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
    # Bảo toàn 100% thông tin thống kê của toàn bộ description gốc TRƯỚC KHI cắt ngắn 500 ký tự đầu:
    if "description" in df.columns:
        full_desc = df["description"]
        df["desc_full_char_len"] = full_desc.str.len()
        df["desc_full_word_count"] = full_desc.str.split().str.len()
        df["desc_url_count"] = full_desc.str.count(r"https?://")
        df["desc_hashtag_count"] = full_desc.str.count(r"#\w+")
        df["desc_has_timestamps"] = full_desc.str.contains(
            r"\b\d{1,2}:\d{2}\b", regex=True
        ).astype(int)
        # Giữ 500 ký tự đầu (phần tóm tắt nội dung riêng của video, loại bỏ phần footer copy-paste cuối bài)
        df["description"] = full_desc.str.slice(0, 500)
    df = df[df["view_count"].notna() & (df["view_count"] > 0)].copy()
    df = df[df["duration_sec"] > 0].copy()

    # 2. Lọc bỏ các video vừa đăng chưa đủ 48 giờ (lượt xem chưa bão hòa)
    df["published_dt"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["crawled_dt"] = pd.to_datetime(df["crawled_at"], utc=True, errors="coerce")
    df = df[df["published_dt"].notna()].copy()

    video_age_hours = (df["crawled_dt"] - df["published_dt"]).dt.total_seconds() / 3600.0
    df = df[video_age_hours >= 48.0].copy()

    # 3. Chỉ giữ các kênh có ít nhất 10 videos hợp lệ để trung vị kênh đáng tin cậy
    channel_counts = df.groupby("channel_id")["video_id"].transform("count")
    df = df[channel_counts >= 10].copy()

    # 4. Tạo nhãn sơ bộ để phân tầng (Stratify) đều cả Kênh lẫn Tỷ lệ Viral giữa Train & Test
    prelim_median = df.groupby("channel_id")["view_count"].transform("median")
    prelim_eng = compute_engagement_rate(df)
    prelim_eng_floor = float(prelim_eng.quantile(ENGAGEMENT_QUANTILE_FLOOR))
    prelim_viral = (
        ((df["view_count"] / (prelim_median + 1.0)) >= VIRAL_RATIO_THRESHOLD)
        & (df["view_count"] >= MIN_VIEWS_FLOOR)
        & (prelim_eng >= prelim_eng_floor)
    ).astype(int)

    stratify_key = df["channel_id"].astype(str) + "_" + prelim_viral.astype(str)

    # Nếu một kênh có nhóm (channel_id, 0) hoặc (channel_id, 1) xuất hiện < 2 lần,
    # gộp toàn bộ kênh đó về stratify theo channel_id để đảm bảo mọi nhóm đều >= 2 mẫu
    key_counts = stratify_key.value_counts()
    rare_keys = set(key_counts[key_counts < 2].index)
    rare_channels = {k.rsplit("_", 1)[0] for k in rare_keys}
    stratify_key = np.where(
        df["channel_id"].isin(rare_channels),
        df["channel_id"].astype(str),
        stratify_key,
    )

    df = df.drop(columns=["published_dt", "crawled_dt"]).reset_index(drop=True)

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(df, stratify_key))

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    # 5. CHỐNG DATA LEAKAGE: Chỉ tính `channel_median_views` & `engagement_floor` từ tập TRAIN
    train_channel_medians = train_df.groupby("channel_id")["view_count"].median()
    train_eng_rate = compute_engagement_rate(train_df)
    train_eng_floor = float(train_eng_rate.quantile(ENGAGEMENT_QUANTILE_FLOOR))

    for part_df in (train_df, test_df):
        part_df["channel_median_views"] = part_df["channel_id"].map(train_channel_medians)
        part_df["viral_ratio"] = part_df["view_count"] / (part_df["channel_median_views"] + 1.0)
        part_df["viral_score_log"] = np.log1p(part_df["viral_ratio"])
        eng_rate = compute_engagement_rate(part_df)
        part_df["engagement_rate"] = eng_rate
        part_df["is_viral"] = (
            (part_df["viral_ratio"] >= VIRAL_RATIO_THRESHOLD)
            & (part_df["view_count"] >= MIN_VIEWS_FLOOR)
            & (eng_rate >= train_eng_floor)
        ).astype(int)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "youtube_train.csv"
    test_path = output_dir / "youtube_test.csv"

    # Xóa file youtube_dataset_full.csv dư thừa nếu còn tồn tại
    full_clean_path = output_dir / "youtube_dataset_full.csv"
    if full_clean_path.exists():
        full_clean_path.unlink()

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\n✅ Đã cập nhật bộ dữ liệu chuẩn với Định nghĩa Viral 3 Lớp (Zero Leakage):")
    print(f"   • Điều kiện 1 (Relative Breakout) : viral_ratio >= {VIRAL_RATIO_THRESHOLD}x trung vị kênh")
    print(f"   • Điều kiện 2 (Minimum Reach)     : view_count  >= {MIN_VIEWS_FLOOR:,} views")
    print(f"   • Điều kiện 3 (Organic Engagement): engagement_rate >= {train_eng_floor:.5f} (Top 95% tương tác tự nhiên)")
    print(f"   • Tổng số mẫu hợp lệ : {len(train_df) + len(test_df):,} videos ({train_df['channel_id'].nunique()} kênh)")
    print(
        f"   • Tập Train (80%)     : {len(train_df):,} dòng "
        f"(Tỷ lệ Viral: {train_df['is_viral'].mean() * 100:.2f}%) -> {train_path}"
    )
    print(
        f"   • Tập Test  (20%)     : {len(test_df):,} dòng "
        f"(Tỷ lệ Viral: {test_df['is_viral'].mean() * 100:.2f}%) -> {test_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tạo nhãn mục tiêu và chia tập Train/Test chuẩn cho 3 thành viên ML."
    )
    raw_dir = Path(__file__).resolve().parents[1] / "raw"
    parser.add_argument(
        "--input",
        type=str,
        default=str(raw_dir / "youtube_videos_raw.csv"),
        help="Đường dẫn file dữ liệu thô",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(raw_dir),
        help="Thư mục lưu file train/test",
    )
    args = parser.parse_args()

    prepare_and_split(Path(args.input), Path(args.output_dir))


if __name__ == "__main__":
    main()
