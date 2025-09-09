# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import unicodedata, re
from pathlib import Path

# ===================== CẤU HÌNH =====================
FILES = [
    "data_cleaned/phone.xlsx",  # Đường dẫn tương đối
    "data_cleaned/laptop.xlsx",
    "data_cleaned/camera.xlsx",
    "data_cleaned/speaker.xlsx",
    "data_cleaned/tv.xlsx"
]
SAVE_OUTPUT = True
OUTPUT_PATH = "integrated_data.xlsx"

# Các cột “đuôi” theo thứ tự yêu cầu
TAIL_ORDER = [
    "image_path",
    "source",  # đứng ngay sau image_path
    "thumbnail_url",
    "impression_info_0_metadata_delivery_zone",
]

# ===================== HELPERS =====================
def strip_accents_lower(s):
    """Dùng riêng cho matching/dedup (không đổi dữ liệu hiển thị)."""
    if pd.isna(s):
        return s
    s = str(s).strip().lower()
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def to_num_general(x):
    """Chuẩn hóa số từ chuỗi đa định dạng: giữ ., , , -; xử lý nghìn & thập phân an toàn."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "-"}:
        return np.nan
    s = re.sub(r"[^\d.,-]", "", s)
    if "," in s and "." in s:
        s = s.replace(",", "")        # giả định , là thousands
    elif "," in s and "." not in s:
        s = s.replace(",", ".")       # giả định , là decimal
    if s.count(".") > 1:
        s = s.replace(".", "")        # vệ sinh nếu còn >1 dấu .
    try:
        return float(s) if s else np.nan
    except:
        return np.nan

def choose_best_column(df, candidates):
    """Chọn cột có nhiều giá trị hợp lệ nhất; nếu hoà, ưu tiên theo thứ tự trong candidates."""
    avail = [c for c in candidates if c in df.columns]
    if not avail:
        return None
    counts = pd.Series({c: df[c].notna().sum() for c in avail})
    best = counts.sort_values(ascending=False)
    top_val = best.iloc[0]
    ties = [c for c in best.index if counts[c] == top_val]
    for c in candidates:
        if c in ties:
            return c
    return best.index[0]

# ===================== UNIFY SCHEMA =====================
def unify_schema(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Đưa dữ liệu nguồn về schema chung tối thiểu.
    - Giữ nguyên chữ có dấu/hoa (chỉ strip khoảng trắng & rỗng).
    - Dùng cột 'brand' hiện có (không fallback).
    - 'source' lấy từ category/category_l1, nếu không có thì dùng tên file.
    """
    # cột chính
    id_col    = choose_best_column(df, ["id", "product_id", "sku"])
    name_col  = choose_best_column(df, ["name", "product_name", "title"])
    brand_col = "brand" if "brand" in df.columns else None
    price_col = choose_best_column(df, ["price", "price_num", "final_price"])
    img_col   = choose_best_column(df, ["image_path", "image", "image_url", "imageUrl"])
    thumb_col = choose_best_column(df, ["thumbnail_url", "thumbnail", "image_url", "imageUrl"])
    rating    = choose_best_column(df, ["rating_average", "rating", "avg_rating"])
    qty       = choose_best_column(df, ["quantity_sold_value", "quantity_sold", "sold"])
    seller    = choose_best_column(df, ["seller_id", "seller", "shop_sku", "seller_product_id"])
    cat_for_source = choose_best_column(df, ["category", "category_l1"])  # -> source
    disc_col  = choose_best_column(df, ["discount_percent", "discount_rate", "discount_pct", "discount"])

    # cột danh mục hiển thị lên đầu
    visible_cat = (
        "visible_impression_info_amplitude_category_l1_name"
        if "visible_impression_info_amplitude_category_l1_name" in df.columns else None
    )

    out = pd.DataFrame()
    if id_col:      out["id"] = df[id_col]
    if name_col:    out["name"] = df[name_col]
    if brand_col:   out["brand"] = df[brand_col]
    if price_col:   out["price"] = pd.to_numeric(df[price_col].map(to_num_general), errors="coerce")
    if img_col:     out["image_path"] = df[img_col]
    if thumb_col:   out["thumbnail_url"] = df[thumb_col]
    if rating:      out["rating_average"] = pd.to_numeric(df[rating].map(to_num_general), errors="coerce")
    if qty:         out["quantity_sold_value"] = pd.to_numeric(df[qty].map(to_num_general), errors="coerce")
    if seller:      out["seller_id"] = df[seller]
    if disc_col:
        out["discount_percent"] = pd.to_numeric(df[disc_col].map(to_num_general), errors="coerce")

    # visible category (nếu có) -> giữ nguyên
    if visible_cat:
        out["visible_impression_info_amplitude_category_l1_name"] = df[visible_cat]

    # source
    out["source"] = df[cat_for_source] if cat_for_source else source_name

    # --- PASSTHROUGH cột đặc thù ---
    PASSTHROUGH_COLS = [
        "impression_info_0_metadata_delivery_zone",
        # thêm tên cột khác nếu cần
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for pc in PASSTHROUGH_COLS:
        if pc in df.columns:
            out[pc] = df[pc]
        elif pc.lower() in lower_map:
            out[pc] = df[lower_map[pc.lower()]]

    # Dọn khoảng trắng & giá trị rỗng cho một số cột text chính
    for c in ["name", "brand", "source", "visible_impression_info_amplitude_category_l1_name"]:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip()
            out.loc[out[c].isin(["", "nan", "none", "null"]), c] = np.nan

    return out

# ===================== DEDUP PRIORITY (BẢN VÁ) =====================
def deduplicate_priority(df: pd.DataFrame) -> pd.DataFrame:
    """
    Khử trùng lặp ưu tiên:
    - Ưu tiên có ảnh, quantity_sold_value cao, rating cao.
    - Khóa cứng: id
    - Khóa mềm: name_norm, brand_norm, price, source_norm, visible_cat_norm
    """
    tmp = df.copy()

    # flags & rank
    tmp["_has_img"] = tmp["image_path"].astype(str).str.strip().ne("") if "image_path" in tmp.columns else False
    if "quantity_sold_value" not in tmp.columns:
        tmp["quantity_sold_value"] = np.nan
    if "rating_average" not in tmp.columns:
        tmp["rating_average"] = np.nan

    # Ưu tiên: có ảnh (x2) + bán (x1) + rating (x0.1)
    tmp["_rank"] = (
        tmp["_has_img"].astype(int) * 2
        + tmp["quantity_sold_value"].fillna(0)
        + tmp["rating_average"].fillna(0) * 0.1
    )

    # Chuẩn hoá cho khóa mềm (matching)
    tmp["name_norm"] = tmp["name"].astype(str).map(strip_accents_lower)  if "name"  in tmp.columns else ""
    tmp["brand_norm"] = tmp["brand"].astype(str).map(strip_accents_lower) if "brand" in tmp.columns else ""
    tmp["source_norm"] = tmp["source"].astype(str).map(strip_accents_lower) if "source" in tmp.columns else ""
    tmp["visible_cat_norm"] = (
        tmp["visible_impression_info_amplitude_category_l1_name"].astype(str).map(strip_accents_lower)
        if "visible_impression_info_amplitude_category_l1_name" in tmp.columns else ""
    )

    # Sắp theo ưu tiên rồi drop dup
    tmp = tmp.sort_values(by=["_rank"], ascending=False)

    # Khóa cứng theo id
    if "id" in tmp.columns and tmp["id"].notna().any():
        tmp = tmp.drop_duplicates(subset=["id"], keep="first")

    # Khóa mềm: thêm source & visible_cat để tránh gộp nhầm giữa danh mục/nguồn khác nhau
    soft_keys = [c for c in ["name_norm", "brand_norm", "price", "source_norm", "visible_cat_norm"] if c in tmp.columns]
    if soft_keys:
        tmp = tmp.drop_duplicates(subset=soft_keys, keep="first")

    # Dọn helper cols
    drop_cols = [c for c in ["_has_img", "_rank", "name_norm", "brand_norm", "source_norm", "visible_cat_norm"] if c in tmp.columns]
    return tmp.drop(columns=drop_cols)

# ===================== CHẠY INTEGRATION =====================
def run_integration(files, out_path, save_output=True):
    frames = []
    for f in files:
        p = Path(f)
        if not p.exists():
            print(f"Bỏ qua: không thấy file {f}")
            continue
        raw = pd.read_excel(p)
        uni = unify_schema(raw, source_name=p.stem)
        frames.append(uni)
        print(f"✔ {p.name} -> {uni.shape}")

    if not frames:
        raise RuntimeError("Không có file hợp lệ để integration.")

    df_integrated = pd.concat(frames, ignore_index=True, sort=False)

    # Loại bản ghi thiếu tối thiểu cho name/brand (nếu cần)
    for col in ["name", "brand"]:
        if col in df_integrated.columns:
            before = len(df_integrated)
            df_integrated = df_integrated[df_integrated[col].astype(str).str.strip().ne("")]
            if len(df_integrated) != before:
                print(f"Loại {before-len(df_integrated)} hàng thiếu '{col}'")

    # Khử trùng lặp theo ưu tiên (bản vá)
    df_integrated = deduplicate_priority(df_integrated)

    # Sắp cột:
    #   - Đầu: visible_impression_info_amplitude_category_l1_name (nếu có)
    front = []
    if "visible_impression_info_amplitude_category_l1_name" in df_integrated.columns:
        front.append("visible_impression_info_amplitude_category_l1_name")
    front += [c for c in [
        "price", "discount_percent", "rating_average", "name", "brand", "id", "seller_id", "quantity_sold_value"
    ] if c in df_integrated.columns]

    #   - Đuôi: image_path -> source -> thumbnail_url -> impression_info_0_metadata_delivery_zone
    end_cols = [c for c in TAIL_ORDER if c in df_integrated.columns]

    #   - Còn lại: giữ nguyên thứ tự phát sinh
    others = [c for c in df_integrated.columns if c not in set(front + end_cols)]
    df_integrated = df_integrated[front + others + end_cols]

    print("Integration xong. Shape:", df_integrated.shape)
    if save_output:
        df_integrated.to_excel(out_path, index=False)
        print("Đã lưu:", out_path)

    return df_integrated

if __name__ == "__main__":
    run_integration(FILES, OUTPUT_PATH, SAVE_OUTPUT)
