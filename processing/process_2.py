# group_duplicates_and_build_pairs_keep_columns.py
import json, itertools, random
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ===================== CONFIG =====================
INPUT_XLSX = "pre_data_2.xlsx"
EMB_NPY = "embeddings_resnet50.npy"
PATHS_JSON = "image_paths.json"

COS_NEAR = 0.90
BLOCK = 2048
MIN_GROUP_SIZE = 2

OUTPUT_XLSX = "process_data_1.xlsx"
GROUP_REPORT_CSV = "duplicate_groups_report.csv"

GENERATE_PAIRS = True
PAIRS_CSV = "pairs_labeled.csv"
RANDOM_SEED = 42
NEG_RATIO = 1.0
MAX_POS_PER_GROUP = 200

# ===================== UTILS =====================
class DSU:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n
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

def limited_combinations(paths, max_pairs=None, rng=None):
    combs = list(itertools.combinations(paths, 2))
    if max_pairs and len(combs) > max_pairs:
        rng.shuffle(combs)
        combs = combs[:max_pairs]
    return combs

# ===================== MAIN =====================
def main():
    df = pd.read_excel(INPUT_XLSX)
    if "path" not in df.columns:
        raise ValueError("Không tìm thấy cột 'path' trong file đầu vào.")

    with open(PATHS_JSON, "r", encoding="utf-8") as f:
        paths_valid = json.load(f)
    emb = np.load(EMB_NPY)

    idx_map = {p: i for i, p in enumerate(paths_valid)}
    df["embedding_index"] = df["path"].astype(str).map(idx_map)

    df_valid = df[df["embedding_index"].notna()].copy()
    n = len(df_valid)
    if n == 0:
        raise ValueError("Không có ảnh hợp lệ để gán nhóm!")

    dsu = DSU(n)

    # Exact duplicates by md5
    if "md5" in df_valid.columns:
        grp_md5 = df_valid.dropna(subset=["md5"]).groupby("md5")["embedding_index"].apply(list)
        for ids in grp_md5:
            ids = [int(i) for i in ids if pd.notna(i)]
            if len(ids) >= 2:
                root = ids[0]
                for i in ids[1:]:
                    dsu.union(root, i)

    # Near duplicates by cosine similarity
    valid_indices = df_valid["embedding_index"].astype(int).tolist()
    emb_valid = emb[valid_indices]
    for s in range(0, n, BLOCK):
        e = min(n, s + BLOCK)
        block = emb_valid[s:e]
        sims = cosine_similarity(block, emb_valid)
        for i in range(e - s):
            sims[i, s + i] = -1
        rows, cols = np.where(sims >= COS_NEAR)
        for r, c in zip(rows, cols):
            dsu.union(s + int(r), int(c))

    # Assign groups
    root_to_members = {}
    for idx in range(n):
        root = dsu.find(idx)
        root_to_members.setdefault(root, []).append(idx)

    df["duplicate_group_id"] = -1
    df["duplicate_group_size"] = 1
    gid_counter = 0
    for members in root_to_members.values():
        if len(members) < MIN_GROUP_SIZE:
            continue
        for local_idx in members:
            orig_idx = df_valid.index[local_idx]
            df.at[orig_idx, "duplicate_group_id"] = gid_counter
            df.at[orig_idx, "duplicate_group_size"] = len(members)
        gid_counter += 1

    df["duplicate"] = df["duplicate_group_id"].apply(lambda x: x != -1)

    # Xuất file
    df.to_excel(OUTPUT_XLSX, index=False)
    print(f"Saved grouped file → {OUTPUT_XLSX}")
    print(f"Duplicate images: {df['duplicate'].sum()} / {len(df)}")

    # Group report
    df[df["duplicate"]].groupby("duplicate_group_id").size().reset_index(name="group_size") \
        .sort_values("group_size", ascending=False).to_csv(GROUP_REPORT_CSV, index=False)
    print(f"Group report saved → {GROUP_REPORT_CSV}")

    # Sinh pairs
    if GENERATE_PAIRS:
        rng = random.Random(RANDOM_SEED)
        groups = df[df["duplicate_group_id"] != -1].groupby("duplicate_group_id")["path"].apply(list)
        pos_pairs = []
        for plist in groups:
            if len(plist) < 2:
                continue
            combs = limited_combinations(plist, MAX_POS_PER_GROUP, rng)
            for a, b in combs:
                a_, b_ = (a, b) if a <= b else (b, a)
                pos_pairs.append((a_, b_, 1))
        print(f"Positive pairs: {len(pos_pairs)}")

        path2gid = dict(zip(df["path"], df["duplicate_group_id"]))
        all_paths = df["path"].tolist()
        target_neg = int(len(pos_pairs) * NEG_RATIO)
        neg_pairs = set()
        attempts = 0
        while len(neg_pairs) < target_neg and attempts < max(50000, 20 * target_neg):
            a, b = rng.sample(all_paths, 2)
            if path2gid.get(a, -1) != path2gid.get(b, -1):
                a_, b_ = (a, b) if a <= b else (b, a)
                neg_pairs.add((a_, b_, 0))
            attempts += 1

        pairs_all = pos_pairs + list(neg_pairs)
        pairs_all = list({(a, b, y) for (a, b, y) in pairs_all})
        rng.shuffle(pairs_all)

        pd.DataFrame(pairs_all, columns=["path_a", "path_b", "label"]).to_csv(PAIRS_CSV, index=False)
        print(f"Saved pairs → {PAIRS_CSV} ({len(pairs_all)} rows)")

if __name__ == "__main__":
    main()
