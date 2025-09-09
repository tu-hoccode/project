#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

# ========= CẤU HÌNH =========
INPUT_XLSX  = "processing_data_4.xlsx"   # hoặc file sau Knowledge-enriched Data
OUTPUT_XLSX = "processing_data_final.xlsx"

# ========= HÀM PHỤ =========
def to_numeric_safe(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def strip_text_safe(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

def clip_iqr(s: pd.Series, k: float = 1.5):
    """Winsorize/clip theo IQR để giảm outlier, không làm méo dữ liệu nhiều."""
    if s.dropna().empty:
        return s
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - k * iqr, q3 + k * iqr
    low = max(0, low)
    return s.clip(lower=low, upper=high)

# ========= 1) ĐỌC & CHỌN CỘT =========
df = pd.read_excel(INPUT_XLSX)

columns_keep = [
    # Thông tin cơ bản
    "visible_impression_info_amplitude_category_l1_name", "id", "seller_id", "name", "brand", "price", "currency",
    # Enriched
    "price_level", "popularity", "quality_label", "brand_type", "dup_status",
    # Thông tin bán hàng
    "rating_average", "quantity_sold_value",
    # Cluster/Feature
    "cluster_id", "pca1", "pca2"
]
exist_keep = [c for c in columns_keep if c in df.columns]
df_clean = df[exist_keep].copy()

# ========= 2) ĐỔI TÊN CỘT =========
rename_map = {
    "visible_impression_info_amplitude_category_l1_name": "Category",
    "id": "ProductID",
    "seller_id": "SellerID",
    "name": "Title",
    "brand": "Brand",
    "price": "Price",
    "currency": "Currency",
    "price_level": "PriceSegment",
    "popularity": "PopularitySegment",
    "quality_label": "RatingSegment",
    "brand_type": "BrandCategory",
    "dup_status": "DuplicateStatus",
    "rating_average": "RatingAverage",
    "quantity_sold_value": "QuantitySold",
    "cluster_id": "ClusterID",
    "pca1": "FeaturePCA1",
    "pca2": "FeaturePCA2"
}
rename_map = {k: v for k, v in rename_map.items() if k in df_clean.columns}
df_clean = df_clean.rename(columns=rename_map)

# ========= 3) CHUẨN HÓA KIỂU DỮ LIỆU =========
text_cols = ["Category", "Title", "Brand", "Currency", "PriceSegment",
             "PopularitySegment", "RatingSegment", "BrandCategory", "DuplicateStatus"]
df_clean = strip_text_safe(df_clean, text_cols)

num_cols = ["Price", "RatingAverage", "QuantitySold", "FeaturePCA1", "FeaturePCA2"]
df_clean = to_numeric_safe(df_clean, num_cols)

# ========= 4) RÀNG BUỘC HỢP LỆ =========
if "Price" in df_clean.columns:
    df_clean.loc[df_clean["Price"] <= 0, "Price"] = np.nan
if "RatingAverage" in df_clean.columns:
    df_clean.loc[(df_clean["RatingAverage"] <= 0) | (df_clean["RatingAverage"] > 5), "RatingAverage"] = np.nan
if "QuantitySold" in df_clean.columns:
    df_clean.loc[df_clean["QuantitySold"] < 0, "QuantitySold"] = np.nan

# ========= 5) IMPUTE RATINGAVERAGE =========
if "RatingAverage" in df_clean.columns:
    rat = df_clean["RatingAverage"].copy()
    if "Brand" in df_clean.columns:
        rat = rat.fillna(df_clean.groupby("Brand")["RatingAverage"].transform("median"))
    if "Category" in df_clean.columns:
        rat = rat.fillna(df_clean.groupby("Category")["RatingAverage"].transform("median"))
    rat = rat.fillna(rat.median(skipna=True))
    df_clean["RatingAverage"] = rat.round(2)

# ========= 6) ĐỒNG BỘ LẠI RATINGSEGMENT =========
if "RatingAverage" in df_clean.columns:
    bins = [0, 3, 4, 5]
    labels = ["Poor", "Good", "Excellent"]
    rs = pd.cut(df_clean["RatingAverage"], bins=bins, labels=labels, include_lowest=True)
    if hasattr(rs, "cat"):
        rs = rs.cat.add_categories("Unknown")
    df_clean["RatingSegment"] = rs.fillna("Unknown")

# ========= 7) CLIP OUTLIER =========
if "Price" in df_clean.columns:
    df_clean["Price"] = clip_iqr(df_clean["Price"], k=1.5)
if "QuantitySold" in df_clean.columns:
    df_clean["QuantitySold"] = clip_iqr(df_clean["QuantitySold"], k=1.5)

# ========= 8) TẠO LOG FEATURES =========
if "Price" in df_clean.columns:
    df_clean["PriceLog1p"] = np.log1p(df_clean["Price"])
if "QuantitySold" in df_clean.columns:
    df_clean["QuantitySoldLog1p"] = np.log1p(df_clean["QuantitySold"].fillna(0))

# ========= 9) DROP TRÙNG & MISSING =========
essential = [c for c in ["ProductID", "Title", "Brand", "Price"] if c in df_clean.columns]
if essential:
    df_clean = df_clean.dropna(subset=essential)
if "ProductID" in df_clean.columns:
    df_clean = df_clean.drop_duplicates(subset=["ProductID"], keep="first")
df_clean = df_clean.reset_index(drop=True)

# ========= 10) LƯU FILE =========
df_clean.to_excel(OUTPUT_XLSX, index=False)
print(f"[INFO] Đã lưu {OUTPUT_XLSX} — dữ liệu đã clean, impute, clip outlier, đồng bộ nhãn & thêm log features.")
