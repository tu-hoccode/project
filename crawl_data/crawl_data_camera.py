import os
import re
import json
import time
import random
import logging
from typing import Dict, Any, List, Set

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import pandas as pd

# ===================== CONFIG =====================
CATEGORY_ID = 1801           # Camera trên Tiki
MAX_PRODUCTS = 100           # cào 100 sản phẩm
LIMIT_PER_PAGE = 40          # theo API Tiki
DOWNLOAD_IMAGES = True       # bật/tắt tải ảnh

OUTPUT_STEM = "data_camera"

IMAGES_DIR = "images/camera"
NDJSON_PATH = f"{OUTPUT_STEM}.ndjson"
XLSX_PATH  = f"data/{OUTPUT_STEM}.xlsx"      
BATCH_SIZE = 100                              # ghi mỗi 100 sản phẩm

REQUEST_TIMEOUT = 15
RETRY_TOTAL = 5
BACKOFF_FACTOR = 0.7
JITTER_RANGE = (0.3, 0.9)  # delay ngẫu nhiên

HEADERS_BASE = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/129.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://tiki.vn/",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
}
API_URL = "https://tiki.vn/api/personalish/v1/blocks/listings"
# ==================================================

# ---------------- Logging ----------------
LOG_PATH = "crawl_camera.log"
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger("tiki_crawler")

# ---------------- HTTP Session ----------------
def build_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=RETRY_TOTAL,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update(HEADERS_BASE)
    return s

# ---------------- Utils ----------------
def jitter_sleep():
    time.sleep(random.uniform(*JITTER_RANGE))

def flatten_json(obj: Any, parent_key: str = "", result: Dict[str, Any] = None) -> Dict[str, Any]:
    if result is None:
        result = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            k = str(k)
            new_key = f"{parent_key}_{k}" if parent_key else k
            flatten_json(v, new_key, result)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_key = f"{parent_key}_{i}" if parent_key else str(i)
            flatten_json(v, new_key, result)
    else:
        result[parent_key] = obj
    return result

def sanitize_col(col: str, idx: int) -> str:
    if not col:
        return f"Column_{idx}"
    col = col.replace(":", "_").replace("/", "_").replace("\n", " ").strip()
    return re.sub(r"\s+", " ", col)

def save_batch_to_ndjson(batch: List[Dict[str, Any]]):
    if not batch:
        return
    with open(NDJSON_PATH, "a", encoding="utf-8") as f:
        for rec in batch:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    batch.clear()

def finalize_to_excel():
    if not os.path.exists(NDJSON_PATH):
        LOGGER.warning("Chưa có NDJSON để xuất Excel.")
        return

    # đảm bảo có thư mục data/ trước khi ghi Excel
    os.makedirs("data", exist_ok=True)

    records = []
    with open(NDJSON_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except json.JSONDecodeError:
                LOGGER.warning("Bỏ qua một dòng NDJSON hỏng.")
                continue

    if not records:
        LOGGER.warning("NDJSON rỗng sau khi đọc.")
        return

    df = pd.json_normalize(records)
    df.columns = [sanitize_col(c, i) for i, c in enumerate(df.columns)]

    try:
        df.to_excel(XLSX_PATH, index=False)
        LOGGER.info(f"Đã xuất {len(df)} dòng vào {XLSX_PATH}")
    except Exception as e:
        LOGGER.exception(f"Lỗi xuất Excel, sẽ fallback sang CSV: {e}")
        df.to_csv(f"{OUTPUT_STEM}_fallback.csv", index=False, encoding="utf-8")
        LOGGER.info(f"Đã xuất {len(df)} dòng vào {OUTPUT_STEM}_fallback.csv (fallback)")

# ---------------- Core Functions ----------------
def fetch_listing_page(session: requests.Session, category_id: int, page: int, limit: int) -> Dict[str, Any]:
    params = {"limit": limit, "category": category_id, "page": page}
    jitter_sleep()
    r = session.get(API_URL, params=params, timeout=REQUEST_TIMEOUT)
    if r.status_code != 200:
        LOGGER.warning(f"HTTP {r.status_code} tại page={page}: {r.text[:400]}")
        return {}
    try:
        return r.json()
    except Exception as e:
        LOGGER.error(f"Lỗi parse JSON page={page}: {e}")
        return {}

def download_image(session: requests.Session, url: str, filename: str) -> str:
    try:
        jitter_sleep()
        res = session.get(url, timeout=REQUEST_TIMEOUT)
        ctype = res.headers.get("Content-Type", "").lower()
        if res.status_code == 200 and "image" in ctype:
            with open(filename, "wb") as f:
                f.write(res.content)
            return filename
        else:
            LOGGER.info(f"Bỏ qua ảnh (status={res.status_code}, ctype={ctype}): {url}")
            return ""
    except Exception as e:
        LOGGER.warning(f"Lỗi tải ảnh {url}: {e}")
        return ""

def crawl_tiki_api(category_id: int, max_products: int, limit: int, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    session = build_session()
    seen_ids: Set[int] = set()
    batch: List[Dict[str, Any]] = []
    total = 0
    page = 1

    LOGGER.info(f"Start crawl: category_id={category_id}, max_products={max_products}, limit={limit}")

    while total < max_products:
        data = fetch_listing_page(session, category_id, page, limit)
        if not data:
            LOGGER.info(f"Dừng: không lấy được dữ liệu tại page={page}")
            break

        products = data.get("data", [])
        if not products:
            LOGGER.info(f"Dừng: page={page} không có sản phẩm.")
            break

        for p in products:
            if total >= max_products:
                break

            pid = p.get("id")
            name = p.get("name")
            if not pid or not name:
                LOGGER.info(f"Bỏ qua bản ghi thiếu id hoặc name: {p}")
                continue
            if pid in seen_ids:
                continue

            seen_ids.add(pid)

            rec = flatten_json(p)
            rec["crawl_ts"] = pd.Timestamp.utcnow().isoformat()
            rec["source_category"] = category_id
            rec["source_page"] = page

            if DOWNLOAD_IMAGES:
                img_url = p.get("thumbnail_url") or p.get("image_url")
                if img_url:
                    fname = os.path.join(output_dir, f"{pid}.jpg")
                    rec["image_path"] = download_image(session, img_url, fname)
                else:
                    rec["image_path"] = ""
            else:
                rec["image_path"] = ""

            batch.append(rec)
            total += 1

            if len(batch) >= BATCH_SIZE:
                save_batch_to_ndjson(batch)

        page += 1

    # flush phần còn lại & xuất Excel
    save_batch_to_ndjson(batch)
    finalize_to_excel()
    LOGGER.info(f"Hoàn tất: thu được {total} sản phẩm.")

# ---------------- Entrypoint ----------------
if __name__ == "__main__":
    try:
        crawl_tiki_api(
            category_id=CATEGORY_ID,
            max_products=MAX_PRODUCTS,
            limit=LIMIT_PER_PAGE,
            output_dir=IMAGES_DIR,
        )
        print(
            f"Xong. Xem dữ liệu ở '{XLSX_PATH}'. "
            f"(Trung gian: '{NDJSON_PATH}') Ảnh ở '{IMAGES_DIR}/'."
        )
    except Exception as e:
        LOGGER.exception(f"CRASH: {e}")
        print("Có lỗi nghiêm trọng, xem thêm trong crawl_camera.log")
