import re
import pandas as pd
import numpy as np

# ===== LOAD DATA =====
df = pd.read_excel("processing_data_4.xlsx")

# ===== RENAME =====
rename_map = {
    "discount_percent_clean": "discount_percent",
    "quantity_sold_value": "quantity_sold",
    "seller_id_le": "seller_id",
    "brand_le": "brand"
}
df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

# ===== CLEAN BY DOMAIN RULES =====
if "rating_average" in df.columns:
    df = df[(df["rating_average"].between(1, 5)) | (df["rating_average"].isna())]

if "discount_percent" in df.columns:
    df = df[(df["discount_percent"].between(0, 100)) | (df["discount_percent"].isna())]

if "price" in df.columns:
    df = df[df["price"] > 0]

# ===== REMOVE OUTLIERS (IQR) =====
def remove_outliers_iqr(df, col):
    if col not in df.columns:
        return df
    s = pd.to_numeric(df[col], errors="coerce")
    Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    return df[(s >= lower) & (s <= upper)]

for col in ["price", "quantity_sold"]:
    df = remove_outliers_iqr(df, col)

# ===== DROP REDUNDANT COLUMNS =====
drop_patterns = [
    r"^image_path_.*", r"^thumbnail_url$",
    r"^dup_path_.*", r"^dup_id_.*",
    r"_norm$", r"_key$", r"_hint$", r"_clean$", r"_le$",
]
drop_cols = [c for c in df.columns for pat in drop_patterns if re.search(pat, c)]
df = df.drop(columns=list(set(drop_cols)), errors="ignore")

# ===== REORDER COLUMNS =====
col_order = []
col_order += [c for c in ["id", "cluster_id", "dup_component_id"] if c in df.columns]
col_order += [c for c in ["price", "discount_percent", "rating_average", "quantity_sold", "seller_id", "brand"] if c in df.columns]
col_order += [c for c in df.columns if c.startswith("is_duplicate") or c.startswith("pred_") or c.startswith("proba_")]
col_order += [c for c in ["avg_price", "avg_rating", "avg_quantity_sold", "dup_ratio"] if c in df.columns]
col_order += [c for c in ["price_segment", "rating_tag", "popularity", "cluster_meaning"] if c in df.columns]
col_order += [c for c in ["dup_degree"] if c in df.columns]
col_order += [c for c in df.columns if c.lower().startswith("pca")]
remaining_cols = [c for c in df.columns if c not in col_order]
col_order += remaining_cols

df_clean = df[col_order].reset_index(drop=True)

# ===== SAVE FINAL CLEAN DATA =====
df_clean.to_excel("../data_final/data_final.xlsx", index=False)

# ===== DATA DICTIONARY =====
data_dict = pd.DataFrame({
    "column": df_clean.columns,
    "dtype": df_clean.dtypes.astype(str).values,
    "non_null_count": df_clean.notna().sum().values,
    "null_count": df_clean.isna().sum().values,
})

data_dict.to_csv("data_dictionary.csv", index=False)

print("[DONE] Saved data_final.xlsx and data_dictionary.csv")
