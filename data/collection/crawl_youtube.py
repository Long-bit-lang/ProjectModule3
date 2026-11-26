"""
Script thu thập dữ liệu video YouTube Việt Nam qua YouTube Data API v3.
Thiết kế tối ưu Quota (chỉ tốn ~1.200 / 10.000 quota cho ~15.000 videos):
- Sử dụng danh sách kênh đa dạng (Handles) + Tìm kiếm kênh tự động theo từ khóa
- Lấy danh sách video qua Uploads Playlist (1 quota / 50 videos) thay vì Search (100 quota)
- Lấy chi tiết 50 videos cùng lúc qua endpoint videos.list (1 quota / 50 videos)
- Tự động lưu checkpoint sau mỗi kênh để không bao giờ mất dữ liệu giữa chừng.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"

# Bảng ánh xạ Category ID chính thức của YouTube
CATEGORY_MAP: dict[str, str] = {
    "1": "Film & Animation",
    "2": "Autos & Vehicles",
    "10": "Music",
    "15": "Pets & Animals",
    "17": "Sports",
    "19": "Travel & Events",
    "20": "Gaming",
    "22": "People & Blogs",
    "23": "Comedy",
    "24": "Entertainment",
    "25": "News & Politics",
    "26": "Howto & Style",
    "27": "Education",
    "28": "Science & Technology",
    "29": "Nonprofits & Activism",
}

# Danh sách kênh YouTube Việt Nam đa dạng lĩnh vực & quy mô (lấy tốn chỉ 1 quota/handle)
SEED_CHANNEL_HANDLES: list[str] = [
    # 1. Công nghệ & Review (Tech)
    "@VinhVatVo", "@DuyTham", "@Schannelvn", "@vatvostudio", "@TonyPhungStudio",
    "@ReLab", "@AnhemTV", "@Gearvn", "@CellphoneS", "@ThinkView", "@KhoiNgong",
    # 2. Gaming & Streamer
    "@MixiGaming", "@CrisDevilGamer", "@ThayGiaoBa", "@TrucTiepGame",
    "@SangTraan", "@AsMobile", "@GaoBacTV", "@MeoSimmy", "@oopsbanana",
    # 3. Ẩm thực & Du lịch (Food & Travel)
    "@KhoaiLangThang", "@NinhTitoo", "@KhoaPug", "@HoaBanFood",
    "@BestEverFoodReviewShow", "@ChallengeMe", "@CoBaBinhDuong", "@VuongAnhOle",
    # 4. Giáo dục, Tài chính, Kiến thức & Podcast
    "@Spiderum", "@Vietcetera", "@HieuTV", "@Web5Ngay", "@VTV24",
    "@Kenh14Official", "@CafeBiz", "@LuatSuNguyenVanLoc", "@DuongFG",
    "@KienThucThuVi", "@ToiDiCodeDao", "@ThaiPhamHappyLive",
    # 5. Phim ảnh, Giải trí & Hài kịch
    "@PhePhim", "@FAPTV", "@VieChannelHTV2", "@DongTayPromotionOfficial",
    "@HuynhLapOfficial", "@TranThanhTown", "@W2WCartoon", "@W2WMovie",
    "@GhiềnMìGõ", "@KemXoiTV", "@1977Vlog",
    # 6. Âm nhạc & Nghệ sĩ
    "@SonTungMTP", "@DenVau", "@HoangThuyLinh", "@HaAnhTuan",
    "@MonoOfficial", "@HIEUTHUHAI", "@tlinhofficial", "@BichPhuongOfficial",
    # 7. Đời sống, Vlog, Sức khỏe & Thể thao
    "@GiangOi", "@DinoVu", "@HannahOlala", "@ChloeNguyen",
    "@AnNguyenFitness", "@ShinPhamm", "@BongDaTV", "@FPTBongDa",
]

# Từ khóa để tự động khám phá thêm các kênh vừa và nhỏ tại Việt Nam (nếu bật --discover)
DISCOVERY_KEYWORDS: list[str] = [
    "review công nghệ điện thoại laptop",
    "vlog du lịch ẩm thực đường phố việt nam",
    "hướng dẫn lập trình học python ai",
    "đầu tư tài chính chứng khoán cá nhân",
    "phân tích tóm tắt phim chiếu rạp",
    "podcast kỹ năng sống phát triển bản thân",
    "highlight gaming liên quân valorant",
    "tập gym giảm cân nấu ăn healthy",
]

ISO_DURATION_RE = re.compile(
    r"^PT(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?$"
)

CSV_COLUMNS: list[str] = [
    "video_id",
    "channel_id",
    "channel_title",
    "channel_custom_url",
    "channel_published_at",
    "channel_country",
    "channel_subscribers",
    "channel_total_views",
    "channel_video_count",
    "title",
    "description",
    "tags",
    "tag_count",
    "category_id",
    "category_name",
    "published_at",
    "duration_iso",
    "duration_sec",
    "dimension",
    "definition",
    "caption",
    "licensed_content",
    "made_for_kids",
    "default_language",
    "default_audio_language",
    "topic_categories",
    "thumbnail_url",
    "view_count",
    "like_count",
    "comment_count",
    "crawled_at",
]


def load_api_key(cli_key: str | None = None) -> str:
    """Đọc YOUTUBE_API_KEY từ tham số CLI, biến môi trường hoặc file .env."""
    if cli_key:
        return cli_key.strip()
    if os.environ.get("YOUTUBE_API_KEY"):
        return os.environ["YOUTUBE_API_KEY"].strip()

    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == "YOUTUBE_API_KEY":
                return v.strip().strip('"').strip("'")

    print(
        "❌ Không tìm thấy YOUTUBE_API_KEY!\n"
        "Hãy tạo file .env ở thư mục gốc dự án với nội dung:\n"
        "YOUTUBE_API_KEY=AIzaSy...\n"
        "Hoặc truyền trực tiếp qua cờ: --api-key AIzaSy...",
        file=sys.stderr,
    )
    sys.exit(1)


def parse_iso_duration(duration_str: str) -> int:
    """Chuyển đổi chuỗi thời lượng ISO 8601 (VD: PT12M30S) sang tổng số giây."""
    if not duration_str:
        return 0
    # Xử lý trường hợp có ngày (P1DT...)
    if "D" in duration_str:
        return 86400
    match = ISO_DURATION_RE.match(duration_str)
    if not match:
        return 0
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = int(match.group("seconds") or 0)
    return hours * 3600 + minutes * 60 + seconds


class YouTubeCrawler:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.session = requests.Session()
        self.estimated_quota_used = 0

    def _get(self, endpoint: str, params: dict[str, Any], quota_cost: int = 1) -> dict[str, Any]:
        params["key"] = self.api_key
        url = f"{YOUTUBE_API_BASE}/{endpoint}"
        for attempt in range(3):
            try:
                resp = self.session.get(url, params=params, timeout=20)
                self.estimated_quota_used += quota_cost
                if resp.status_code == 200:
                    return resp.json()
                error_msg = resp.text
                if resp.status_code in (403, 429):
                    print(f"⚠️ Lỗi Quota / Rate Limit ({resp.status_code}): {error_msg}", file=sys.stderr)
                    raise RuntimeError("Quota Exceeded hoặc API Key chưa bật YouTube Data API v3.")
                print(f"⚠️ HTTP {resp.status_code} tại {endpoint} (thử lại {attempt + 1}/3)...")
                time.sleep(1.5 * (attempt + 1))
            except requests.RequestException as exc:
                print(f"⚠️ Lỗi kết nối mạng: {exc} (thử lại {attempt + 1}/3)...")
                time.sleep(2)
        return {}

    def resolve_handle_to_channel(self, handle: str) -> dict[str, Any] | None:
        """Lấy thông tin kênh trực tiếp từ @handle (Chỉ tốn 1 quota!)."""
        clean_handle = handle if handle.startswith("@") else f"@{handle}"
        data = self._get(
            "channels",
            {
                "part": "snippet,contentDetails,statistics",
                "forHandle": clean_handle,
            },
            quota_cost=1,
        )
        items = data.get("items", [])
        if not items:
            return None
        return self._extract_channel_meta(items[0])

    def fetch_channels_by_ids(self, channel_ids: list[str]) -> list[dict[str, Any]]:
        """Lấy thông tin tối đa 50 kênh cùng lúc theo ID (Chỉ tốn 1 quota / 50 kênh)."""
        results: list[dict[str, Any]] = []
        for i in range(0, len(channel_ids), 50):
            batch = channel_ids[i : i + 50]
            data = self._get(
                "channels",
                {
                    "part": "snippet,contentDetails,statistics",
                    "id": ",".join(batch),
                },
                quota_cost=1,
            )
            for item in data.get("items", []):
                meta = self._extract_channel_meta(item)
                if meta:
                    results.append(meta)
        return results

    def discover_channels_by_keywords(self, keywords: list[str], pages_per_kw: int = 1) -> list[str]:
        """Tìm thêm channel_id mới qua từ khóa tiếng Việt (Tốn 100 quota / lượt search)."""
        discovered_ids: set[str] = set()
        for kw in keywords:
            page_token: str | None = None
            for _ in range(pages_per_kw):
                params: dict[str, Any] = {
                    "part": "snippet",
                    "q": kw,
                    "type": "video",
                    "regionCode": "VN",
                    "relevanceLanguage": "vi",
                    "maxResults": 50,
                }
                if page_token:
                    params["pageToken"] = page_token
                data = self._get("search", params, quota_cost=100)
                for item in data.get("items", []):
                    cid = item.get("snippet", {}).get("channelId")
                    if cid:
                        discovered_ids.add(cid)
                page_token = data.get("nextPageToken")
                if not page_token:
                    break
        return list(discovered_ids)

    @staticmethod
    def _extract_channel_meta(item: dict[str, Any]) -> dict[str, Any] | None:
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        uploads_playlist = (
            item.get("contentDetails", {})
            .get("relatedPlaylists", {})
            .get("uploads")
        )
        if not uploads_playlist:
            return None
        return {
            "channel_id": item.get("id", ""),
            "channel_title": snippet.get("title", ""),
            "channel_custom_url": snippet.get("customUrl", ""),
            "channel_published_at": snippet.get("publishedAt", ""),
            "channel_country": snippet.get("country", "VN"),
            "channel_subscribers": int(stats.get("subscriberCount", 0) or 0),
            "channel_total_views": int(stats.get("viewCount", 0) or 0),
            "channel_video_count": int(stats.get("videoCount", 0) or 0),
            "uploads_playlist_id": uploads_playlist,
        }

    def fetch_channel_video_ids(self, playlist_id: str, max_videos: int = 200) -> list[str]:
        """Lấy danh sách video_id từ Uploads Playlist của kênh (1 quota / 50 videos)."""
        video_ids: list[str] = []
        page_token: str | None = None
        while len(video_ids) < max_videos:
            batch_size = min(50, max_videos - len(video_ids))
            params: dict[str, Any] = {
                "part": "contentDetails",
                "playlistId": playlist_id,
                "maxResults": batch_size,
            }
            if page_token:
                params["pageToken"] = page_token
            data = self._get("playlistItems", params, quota_cost=1)
            items = data.get("items", [])
            if not items:
                break
            for item in items:
                vid = item.get("contentDetails", {}).get("videoId")
                if vid:
                    video_ids.append(vid)
            page_token = data.get("nextPageToken")
            if not page_token:
                break
        return video_ids

    def fetch_videos_details(
        self, video_ids: list[str], channel_meta: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Lấy toàn bộ metadata & statistics của tối đa 50 videos/lần (1 quota / 50 videos)."""
        rows: list[dict[str, Any]] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for i in range(0, len(video_ids), 50):
            batch = video_ids[i : i + 50]
            data = self._get(
                "videos",
                {
                    "part": "snippet,contentDetails,statistics,status,topicDetails",
                    "id": ",".join(batch),
                },
                quota_cost=1,
            )
            for item in data.get("items", []):
                snippet = item.get("snippet", {})
                content = item.get("contentDetails", {})
                stats = item.get("statistics", {})
                status = item.get("status", {})
                topics = item.get("topicDetails", {}).get("topicCategories", [])

                # Bỏ qua các video ẩn lượt xem hoặc chưa công khai
                if "viewCount" not in stats:
                    continue

                tags_list = snippet.get("tags", []) or []
                cat_id = str(snippet.get("categoryId", ""))
                duration_iso = content.get("duration", "")
                duration_sec = parse_iso_duration(duration_iso)

                thumbs = snippet.get("thumbnails", {})
                thumb_url = (
                    thumbs.get("maxres", {}).get("url")
                    or thumbs.get("high", {}).get("url")
                    or thumbs.get("medium", {}).get("url")
                    or ""
                )

                clean_topics = [t.split("/")[-1] for t in topics if isinstance(t, str)]

                rows.append(
                    {
                        "video_id": item.get("id", ""),
                        "channel_id": channel_meta["channel_id"],
                        "channel_title": channel_meta["channel_title"],
                        "channel_custom_url": channel_meta["channel_custom_url"],
                        "channel_published_at": channel_meta["channel_published_at"],
                        "channel_country": channel_meta["channel_country"],
                        "channel_subscribers": channel_meta["channel_subscribers"],
                        "channel_total_views": channel_meta["channel_total_views"],
                        "channel_video_count": channel_meta["channel_video_count"],
                        "title": " ".join(str(snippet.get("title", "")).splitlines()).strip(),
                        "description": " ".join(str(snippet.get("description", "")).splitlines()).strip(),
                        "tags": "|".join(tags_list),
                        "tag_count": len(tags_list),
                        "category_id": cat_id,
                        "category_name": CATEGORY_MAP.get(cat_id, "Other"),
                        "published_at": snippet.get("publishedAt", ""),
                        "duration_iso": duration_iso,
                        "duration_sec": duration_sec,
                        "dimension": content.get("dimension", "2d"),
                        "definition": content.get("definition", "hd"),
                        "caption": str(content.get("caption", "false")).lower() == "true",
                        "licensed_content": bool(content.get("licensedContent", False)),
                        "made_for_kids": bool(status.get("madeForKids", False)),
                        "default_language": snippet.get("defaultLanguage", ""),
                        "default_audio_language": snippet.get("defaultAudioLanguage", ""),
                        "topic_categories": "|".join(clean_topics),
                        "thumbnail_url": thumb_url,
                        "view_count": int(stats.get("viewCount", 0) or 0),
                        "like_count": int(stats.get("likeCount", 0) or 0),
                        "comment_count": int(stats.get("commentCount", 0) or 0),
                        "crawled_at": now_iso,
                    }
                )
        return rows


def load_existing_video_ids(csv_path: Path) -> tuple[set[str], set[str]]:
    """Đọc các video_id và channel_id đã cào để hỗ trợ chạy tiếp (Resume/Checkpoint)."""
    seen_videos: set[str] = set()
    seen_channels: set[str] = set()
    if not csv_path.exists():
        return seen_videos, seen_channels

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("video_id"):
                seen_videos.add(row["video_id"])
            if row.get("channel_id"):
                seen_channels.add(row["channel_id"])
    return seen_videos, seen_channels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Thu thập dữ liệu video YouTube Việt Nam tối ưu Quota cho đồ án Module 3."
    )
    parser.add_argument("--api-key", type=str, default=None, help="YouTube Data API v3 Key")
    parser.add_argument(
        "--videos-per-channel",
        type=int,
        default=200,
        help="Số lượng video tối đa lấy trên mỗi kênh (mặc định: 200)",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Tự động tìm kiếm thêm các kênh vừa & nhỏ qua từ khóa (tốn thêm ~800 quota)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "raw" / "youtube_videos_raw.csv"),
        help="Đường dẫn file CSV đầu ra",
    )
    args = parser.parse_args()

    api_key = load_api_key(args.api_key)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_videos, seen_channels = load_existing_video_ids(output_path)
    if seen_videos:
        print(
            f"🔄 Phát hiện dữ liệu cũ tại {output_path.name}: "
            f"{len(seen_videos):,} videos từ {len(seen_channels)} kênh. Sẽ tự động cào nối tiếp!"
        )

    crawler = YouTubeCrawler(api_key)
    channels_to_crawl: dict[str, dict[str, Any]] = {}

    print(f"🔍 Đang phân giải {len(SEED_CHANNEL_HANDLES)} kênh hạt giống (@handle)...")
    for handle in SEED_CHANNEL_HANDLES:
        meta = crawler.resolve_handle_to_channel(handle)
        if meta and meta["channel_id"] not in seen_channels:
            channels_to_crawl[meta["channel_id"]] = meta

    if args.discover:
        print(f"🌐 Đang khám phá thêm kênh mới qua {len(DISCOVERY_KEYWORDS)} chủ đề tìm kiếm...")
        discovered_ids = crawler.discover_channels_by_keywords(DISCOVERY_KEYWORDS, pages_per_kw=1)
        new_ids = [
            cid for cid in discovered_ids
            if cid not in channels_to_crawl and cid not in seen_channels
        ]
        for meta in crawler.fetch_channels_by_ids(new_ids):
            # Lọc các kênh có tối thiểu 15 videos và >= 1000 subscribers
            if meta["channel_video_count"] >= 15 and meta["channel_subscribers"] >= 1000:
                channels_to_crawl[meta["channel_id"]] = meta

    total_channels = len(channels_to_crawl)
    print(
        f"✅ Sẵn sàng cào {total_channels} kênh mới "
        f"(Tối đa {args.videos_per_channel} videos/kênh). "
        f"Quota đã dùng: ~{crawler.estimated_quota_used} units.\n"
    )

    file_exists = output_path.exists() and output_path.stat().st_size > 0
    total_saved = len(seen_videos)

    with output_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()

        for idx, (cid, cmeta) in enumerate(channels_to_crawl.items(), start=1):
            try:
                vids = crawler.fetch_channel_video_ids(
                    cmeta["uploads_playlist_id"], max_videos=args.videos_per_channel
                )
                new_vids = [v for v in vids if v not in seen_videos]
                if not new_vids:
                    continue

                video_rows = crawler.fetch_videos_details(new_vids, cmeta)
                for row in video_rows:
                    writer.writerow(row)
                    seen_videos.add(row["video_id"])
                f.flush()

                total_saved += len(video_rows)
                print(
                    f"[{idx:02d}/{total_channels:02d}] {cmeta['channel_title'][:28]:<28} "
                    f"| +{len(video_rows):>3} videos "
                    f"| Tổng: {total_saved:>6,} videos "
                    f"| Quota: ~{crawler.estimated_quota_used:,} units"
                )
            except RuntimeError as err:
                print(f"\n🛑 Dừng sớm do giới hạn API: {err}")
                break
            except Exception as exc:
                print(f"⚠️ Bỏ qua kênh {cmeta['channel_title']} do lỗi: {exc}")
                continue

    print(
        f"\n🎉 Hoàn tất! Đã lưu tổng cộng {total_saved:,} videos tại:\n"
        f"👉 {output_path}\n"
        f"📊 Tổng Quota ước tính đã sử dụng: ~{crawler.estimated_quota_used:,} / 10,000 units."
    )


if __name__ == "__main__":
    main()
