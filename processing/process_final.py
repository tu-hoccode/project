import pandas as pd

# ===== LOAD DATA =====
df = pd.read_excel("processing_data_5.xlsx")

# ===== CHỌN CỘT ĐẶC TRƯNG & LABEL =====
# Numeric metadata
num_cols = ["price", "discount_percent", "rating_average", "quantity_sold"]

# Categorical metadata (sẽ cần encode khi train)
cat_cols = ["brand", "seller_id"]

# Graph features
graph_cols = ["dup_degree", "dup_component_id"]

# PCA features (embedding ảnh giảm chiều)
pca_cols = [c for c in df.columns if c.lower().startswith("pca")]

# Enriched tags
enrich_cols = ["price_segment", "rating_tag", "popularity", "cluster_id", "cluster_meaning"]

# Label (nếu có)
label_cols = ["is_duplicate"]

# Cột bổ sung
name_col = ["name"]
category_col = ["visible_impression_info_amplitude_category_l1_name"]

# Gom tất cả
keep_cols = name_col + category_col + num_cols + cat_cols + graph_cols + pca_cols + enrich_cols + label_cols

# Lọc cột có thật trong dataframe
keep_cols = [c for c in keep_cols if c in df.columns]

df_features = df[keep_cols].copy()

# ===== RENAME CỘT =====
df_features = df_features.rename(
    columns={"visible_impression_info_amplitude_category_l1_name": "category"}
)

# ===== ĐƯA 2 CỘT MỚI LÊN ĐẦU =====
col_order = ["name", "category"] + [c for c in df_features.columns if c not in ["name", "category"]]
df_features = df_features[col_order]

# ===== SAVE =====
df_features.to_excel("../data_final/data_final.xlsx", index=False)

print(f"[DONE] Đã lưu data_final.xlsx với {len(df_features.columns)} cột và {len(df_features)} dòng.")
