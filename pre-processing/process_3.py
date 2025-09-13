import os, json, itertools
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ==== CONFIG ====
LABELED_XLSX       = "process_data_1.xlsx"  # từ bước Labeled Data
EMB_NPY            = "embeddings_resnet50.npy"
PATHS_JSON         = "image_paths.json"
COS_CLUSTER        = 0.90        # ngưỡng nối cạnh để gom nhóm
BLOCK              = 2048        # tính cosine theo block tránh tràn RAM
ONLY_DUPLICATES    = True        # True: chỉ gom ảnh is_duplicate=1
MIN_CLUSTER_SIZE   = 2           # cụm phải >= 2 ảnh mới coi là cluster hợp lệ

OUT_CLUSTERS_XLSX  = "process_data_3.xlsx"
OUT_CLUSTER_STATS  = "cluster_stats.csv"
OUT_CLUSTER_REPRS  = "cluster_representatives.xlsx"

# ==== LOAD ====
df = pd.read_excel(LABELED_XLSX)
paths_all = df["path"].astype(str).tolist()

with open(PATHS_JSON, "r") as f:
    paths_valid = json.load(f)
emb_all = np.load(EMB_NPY)  # đã L2-normalized khi trích xuất

# map path -> idx embedding
path2idx = {p:i for i,p in enumerate(paths_valid)}
df["emb_idx"] = df["path"].map(path2idx)

# lọc đối tượng để gom cụm
mask = df["emb_idx"].notna()
if ONLY_DUPLICATES and "is_duplicate" in df.columns:
    mask = mask & (df["is_duplicate"] == 1)

df_sub = df[mask].copy().reset_index(drop=True)

if df_sub.empty:
    raise ValueError("Không còn ảnh hợp lệ để gom cụm (kiểm tra is_duplicate và emb_idx)!")

emb_idx = df_sub["emb_idx"].astype(int).tolist()
emb     = emb_all[emb_idx]
n = len(df_sub)
print(f"🔎 Số ảnh đưa vào clustering: {n}")

# ==== DSU (Union-Find) ====
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

# (tuỳ chọn) nối exact duplicate theo md5 nếu có
if "md5" in df_sub.columns:
    grp_md5 = df_sub.dropna(subset=["md5"]).groupby("md5").indices
    for _, idxs in grp_md5.items():
        if len(idxs) >= 2:
            root = idxs[0]
            for j in idxs[1:]:
                dsu.union(root, j)

# ==== Nối near-duplicate theo cosine ====
for s in range(0, n, BLOCK):
    e = min(n, s+BLOCK)
    block = emb[s:e]              # (B, d)
    sims  = cosine_similarity(block, emb)  # (B, n)
    # bỏ self-sim trong block
    for i in range(e - s):
        sims[i, s+i] = -1.0
    rows, cols = np.where(sims >= COS_CLUSTER)
    for r, c in zip(rows, cols):
        dsu.union(s + int(r), int(c))

# ==== Assign cluster_id (connected components) ====
root2cid = {}
cid_counter = 0
cluster_id = [-1]*n

# gom theo root
groups = {}
for i in range(n):
    r = dsu.find(i)
    groups.setdefault(r, []).append(i)

# chỉ gán cluster_id cho cụm >= MIN_CLUSTER_SIZE
for r, members in groups.items():
    if len(members) < MIN_CLUSTER_SIZE:
        continue
    cid = cid_counter
    cid_counter += 1
    for i in members:
        cluster_id[i] = cid

df_sub["cluster_id"] = cluster_id
df_sub["cluster_size"] = df_sub.groupby("cluster_id")["cluster_id"].transform("count")

# ghép ngược vào df gốc
df_out = df.copy()
df_out = df_out.merge(
    df_sub[["path","cluster_id","cluster_size"]],
    on="path", how="left"
)
df_out["cluster_id"]   = df_out["cluster_id"].fillna(-1).astype(int)
df_out["cluster_size"] = df_out["cluster_size"].fillna(1).astype(int)

# ==== Xuất clusters.xlsx (mặc định đưa cột cluster lên đầu) ====
front = ["cluster_id", "cluster_size", "is_duplicate", "dup_group_id", "dup_count", "path"]
front = [c for c in front if c in df_out.columns]
others = [c for c in df_out.columns if c not in set(front)]
df_out = df_out[front + others]
df_out.to_excel(OUT_CLUSTERS_XLSX, index=False)
print(f"✅ Saved clusters → {OUT_CLUSTERS_XLSX}")

# ==== Thống kê cụm ====
stats = (df_out[df_out["cluster_id"]!=-1]
        .groupby("cluster_id")
        .agg(cluster_size=("cluster_id","size"),
            n_unique_path=("path","nunique"))
        .reset_index()
        .sort_values("cluster_size", ascending=False))
stats.to_csv(OUT_CLUSTER_STATS, index=False)
print(f"✅ Saved cluster stats → {OUT_CLUSTER_STATS}")
print(stats.head())

# ==== Chọn ảnh đại diện cho mỗi cluster (representative) ====
# tiêu chí: ưu tiên ảnh xuất hiện trước (hoặc có thể thay bằng ảnh lớn nhất, hoặc theo 'dup_count')
repr_rows = []
for cid, g in df_out[df_out["cluster_id"]!=-1].groupby("cluster_id"):
    # đại diện: hàng đầu tiên (bạn có thể thay bằng logic khác)
    rep = g.iloc[0]
    repr_rows.append({
        "cluster_id": cid,
        "cluster_size": int(g.shape[0]),
        "rep_path": rep["path"]
    })
repr_df = pd.DataFrame(repr_rows).sort_values("cluster_size", ascending=False)
repr_df.to_excel(OUT_CLUSTER_REPRS, index=False)
print(f"✅ Saved cluster representatives → {OUT_CLUSTER_REPRS}")
print(repr_df.head())

print(f"📊 Tổng số cluster >= {MIN_CLUSTER_SIZE}: {stats.shape[0]}")
print(f"📊 Ảnh thuộc cluster: {(df_out['cluster_id']!=-1).sum()} / {len(df_out)}")