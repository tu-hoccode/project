import os, itertools, json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ==== CONFIG ====
INPUT_XLSX   = "pre_data_3.xlsx"
EMB_NPY      = "embeddings_resnet50.npy"
PATHS_JSON   = "image_paths.json"
OUTPUT_XLSX  = "process_data_1.xlsx"
COS_NEAR     = 0.89
BLOCK        = 2048
MIN_GROUP_SIZE = 2

# ==== LOAD DATA ====
print("🔄 Đang load dữ liệu và embeddings...")
df = pd.read_excel(INPUT_XLSX)
with open(PATHS_JSON, "r") as f:
    paths_valid = json.load(f)
emb = np.load(EMB_NPY)

idx_map = {p:i for i,p in enumerate(paths_valid)}
df["valid_idx"] = df["path"].astype(str).map(idx_map)

df_valid = df[df["valid_idx"].notna()].copy()
n = len(df_valid)
if n == 0:
    raise ValueError("❌ Không có ảnh hợp lệ để gán nhãn!")

# ==== DSU ====
class DSU:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0]*n
    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1

dsu = DSU(n)

# ==== 1) Gom nhóm exact duplicate bằng md5 (nếu có) ====
if "md5" in df_valid.columns:
    grp = df_valid.dropna(subset=["md5"]).groupby("md5")["valid_idx"].apply(list)
    for ids in grp:
        ids = [int(i) for i in ids if pd.notna(i)]
        if len(ids) >= 2:
            root = ids[0]
            for i in ids[1:]:
                dsu.union(root, i)

# ==== 2) Near-duplicate bằng cosine ====
valid_indices = df_valid["valid_idx"].astype(int).tolist()
emb_valid = emb[valid_indices]

for s in range(0, n, BLOCK):
    e = min(n, s+BLOCK)
    block = emb_valid[s:e]
    sims = cosine_similarity(block, emb_valid)
    for i in range(e - s):
        sims[i, s+i] = -1  # bỏ self-sim
    rows, cols = np.where(sims >= COS_NEAR)
    for r, c in zip(rows, cols):
        dsu.union(s + int(r), int(c))

# ==== Gán nhóm & đếm ====
groups = {}
for idx in range(n):
    root = dsu.find(idx)
    groups.setdefault(root, []).append(idx)

dup_group_id = [-1] * len(df)
dup_count = [1] * len(df)

for gid, members in enumerate(groups.values()):
    if len(members) < MIN_GROUP_SIZE:
        continue
    for idx in members:
        orig_idx = df_valid.index[idx]
        dup_group_id[orig_idx] = gid
        dup_count[orig_idx] = len(members)

df["dup_group_id"] = dup_group_id
df["dup_count"] = dup_count
df["is_duplicate"] = df["dup_group_id"].apply(lambda x: 1 if x != -1 else 0)

# ==== Xuất file ====
front_cols = ["is_duplicate", "dup_group_id", "dup_count", "path"]
front_cols += [c for c in ["category", "category_id", "md5"] if c in df.columns]
others = [c for c in df.columns if c not in set(front_cols)]
df_out = df[front_cols + others]

df_out.to_excel(OUTPUT_XLSX, index=False)
print(f"✅ Đã lưu kết quả gọn nhẹ → {OUTPUT_XLSX}")
print(f"📊 Ảnh trùng: {df_out['is_duplicate'].sum()} / {len(df_out)} ({df_out['is_duplicate'].mean()*100:.2f}%)")
