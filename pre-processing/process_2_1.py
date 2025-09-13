# %% [markdown]
# ## Tạo bộ dữ liệu pairwise từ process_data_1_labeled.xlsx
# - Sinh cặp ảnh cùng nhóm (label=1)
# - Sinh số lượng tương đương cặp khác nhóm (label=0)
# - Lưu ra pairs_labeled.csv

import pandas as pd
import itertools
import random

INPUT_XLSX  = "process_data_1.xlsx"
OUTPUT_CSV  = "pairs_labeled.csv"
MAX_POS_PER_GROUP = 50   # giới hạn số cặp dương mỗi nhóm để tránh quá lớn
NEG_RATIO          = 1.0 # số lượng cặp âm ≈ NEG_RATIO * số lượng cặp dương
RANDOM_SEED        = 42

# ---- Load dữ liệu ----
df = pd.read_excel(INPUT_XLSX)
if "dup_group_id" not in df.columns:
    raise ValueError("File không có cột dup_group_id!")

# Lọc chỉ các ảnh có gán nhóm
df_dup = df[df["dup_group_id"] != -1].copy()
groups = df_dup.groupby("dup_group_id")["path"].apply(list)

pairs_pos = []
for gid, paths in groups.items():
    if len(paths) < 2:
        continue
    cnt = 0
    for a, b in itertools.combinations(paths, 2):
        pairs_pos.append((a, b, 1))
        cnt += 1
        if cnt >= MAX_POS_PER_GROUP:
            break

print(f"✅ Sinh được {len(pairs_pos)} cặp dương (label=1)")

# ---- Sinh cặp âm (negative) ----
all_paths = df["path"].tolist()
dup_set = set(p for paths in groups for p in paths)
unique_paths = [p for p in all_paths if p not in dup_set]

pairs_neg = []
random.seed(RANDOM_SEED)

while len(pairs_neg) < int(len(pairs_pos) * NEG_RATIO):
    a, b = random.sample(all_paths, 2)
    # bỏ nếu cùng nhóm
    ga = df.loc[df["path"] == a, "dup_group_id"].iloc[0]
    gb = df.loc[df["path"] == b, "dup_group_id"].iloc[0]
    if ga != gb:
        pairs_neg.append((a, b, 0))

print(f"✅ Sinh được {len(pairs_neg)} cặp âm (label=0)")

# ---- Gộp & shuffle ----
pairs_all = pairs_pos + pairs_neg
random.shuffle(pairs_all)

df_pairs = pd.DataFrame(pairs_all, columns=["path_a", "path_b", "label"])
df_pairs.to_csv(OUTPUT_CSV, index=False)

print(f"💾 Đã lưu pairs_labeled.csv với {len(df_pairs)} cặp ảnh")
print(df_pairs.head())
