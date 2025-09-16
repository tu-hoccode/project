# build_pairs_from_groups_keep_columns_with_duplicate.py
import itertools, random
import pandas as pd

# ========= CONFIG =========
INPUT_XLSX = "process_data_1.xlsx"
OUTPUT_PAIRS_CSV = "data_final.csv"

MAX_POS_PER_GROUP = 200
SAMPLE_POS_UNIFORM = True
NEG_RATIO = 1.0
AVOID_SAME_TITLE = False
RANDOM_SEED = 42

# ========= LOAD =========
df = pd.read_excel(INPUT_XLSX)

req_cols = ["path", "duplicate_group_id", "duplicate"]
for c in req_cols:
    if c not in df.columns:
        raise ValueError(f"Thiếu cột '{c}' trong file {INPUT_XLSX}")

df = df[df["path"].astype(str).str.len() > 0].copy()
df["duplicate_group_id"] = df["duplicate_group_id"].fillna(-1).astype(int)
df["duplicate"] = df["duplicate"].astype(bool)

# ========= BUILD POSITIVE PAIRS =========
groups = (
    df[df["duplicate_group_id"] != -1]
    .groupby("duplicate_group_id")["path"]
    .apply(list)
    .to_dict()
)

rng = random.Random(RANDOM_SEED)

def limited_combinations(paths, max_pairs=None, rng=None):
    combs = list(itertools.combinations(paths, 2))
    if max_pairs and len(combs) > max_pairs:
        rng.shuffle(combs)
        combs = combs[:max_pairs]
    return combs

pos_pairs = []
for gid, plist in groups.items():
    if len(plist) < 2:
        continue
    combs = limited_combinations(plist, MAX_POS_PER_GROUP, rng)
    for a, b in combs:
        a_, b_ = (a, b) if a <= b else (b, a)
        pos_pairs.append((a_, b_, True))  # True cho cặp positive

print(f"Positive pairs: {len(pos_pairs)}")

# ========= BUILD NEGATIVE PAIRS =========
path2gid = dict(zip(df["path"], df["duplicate_group_id"]))
all_paths = df["path"].tolist()
title_map = dict(zip(df["path"], df["title"])) if "title" in df.columns else {}

target_neg = int(len(pos_pairs) * NEG_RATIO)
neg_pairs = set()
attempts = 0
MAX_ATTEMPTS = max(50000, 20 * max(1, target_neg))

while len(neg_pairs) < target_neg and attempts < MAX_ATTEMPTS:
    a, b = rng.sample(all_paths, 2)
    if path2gid.get(a, -1) == path2gid.get(b, -1):
        attempts += 1
        continue
    if AVOID_SAME_TITLE and title_map and title_map.get(a) == title_map.get(b):
        attempts += 1
        continue
    pair = (a, b) if a <= b else (b, a)
    neg_pairs.add((pair[0], pair[1], False))  # False cho cặp negative
    attempts += 1

neg_pairs = list(neg_pairs)
print(f"Negative pairs: {len(neg_pairs)} (target ~ {target_neg})")

# ========= MERGE, DEDUP, SHUFFLE =========
pairs_all = pos_pairs + neg_pairs
pairs_all = list({(a, b, y) for (a, b, y) in pairs_all})
rng.shuffle(pairs_all)

pairs_df = pd.DataFrame(pairs_all, columns=["path_a", "path_b", "duplicate"])

# Nếu có cột title, thêm luôn title_a và title_b
if "title" in df.columns:
    title_map = df.set_index("path")["title"]
    pairs_df["title_a"] = pairs_df["path_a"].map(title_map)
    pairs_df["title_b"] = pairs_df["path_b"].map(title_map)
    pairs_df = pairs_df[["path_a", "title_a", "path_b", "title_b", "duplicate"]]

pairs_df.to_csv(OUTPUT_PAIRS_CSV, index=False)
print(f"Saved: {OUTPUT_PAIRS_CSV} ({len(pairs_df)} rows)")

# ========= STATS =========
n_pos = int((pairs_df["duplicate"] == True).sum())
n_neg = int((pairs_df["duplicate"] == False).sum())
print(f"Stats → pos={n_pos}, neg={n_neg}, pos_ratio={n_pos/len(pairs_df):.3f}")
