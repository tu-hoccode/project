#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phát hiện ảnh trùng / gần trùng / trùng hình dạng và gộp nhãn vào encoded_data.

Cấu trúc:
- pre-processing/
    ├─ images_png/                <--- ảnh .png ở đây
    ├─ encoded_data.xlsx          <--- chứa cột image_path (đường dẫn tương đối 'images_png/...') và/hoặc 'id'
- processing/
    ├─ dup_image/                 <--- nơi lưu kết quả phát hiện trùng (cặp, label)
    ├─ process_1.py               <--- file này (chạy từ thư mục processing)
    └─ processing_data_1.xlsx     <--- KẾT QUẢ GỘP NHÃN VỚI DỮ LIỆU GỐC

Thuật toán: pHash + ORB + Hu Moments. Đã siết ngưỡng để chính xác hơn.
"""

import os, sys, math, hashlib
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import imagehash
import cv2
import networkx as nx
from tqdm import tqdm

# ============== CẤU HÌNH THƯ MỤC ==============
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# Input
INPUT_XLSX     = ROOT / "pre-processing" / "encoded_data.xlsx"
# LƯU Ý: trong encoded_data.xlsx, image_path có dạng "images_png/<...>.png"
# Ta đặt base dir là "pre-processing" để ghép trực tiếp.
IMAGE_BASE_DIR = ROOT / "pre-processing"

# Output (label & cặp)
OUTPUT_DIR     = HERE / "dup_image"
OUTPUT_XLSX    = OUTPUT_DIR / "data_with_dups.xlsx"
PAIRS_XLSX     = OUTPUT_DIR / "duplicate_pairs.xlsx"
TXT_REPORT     = OUTPUT_DIR / "duplicates.txt"

# Output gộp nhãn vào dữ liệu gốc:
MERGED_XLSX    = HERE / "processing_data_1.xlsx"

# Cột dữ liệu
IMAGE_COL      = "image_path"   # cột đường dẫn ảnh trong Excel
ID_COL         = "id"           # nếu không có sẽ tạo từ index

# Chấp nhận đuôi ảnh (dữ liệu của bạn là .png; thêm đuôi khác nếu cần)
VALID_EXTS     = {".png"}  # có thể đổi thành {".png",".jpg",".jpeg",".webp",".bmp",".tiff"}

# ============== THAM SỐ (ĐÃ SIẾT NGƯỠNG CHÍNH XÁC HƠN) ==============
PHASH_CANDIDATE_MAX_HD  = 12   # giảm từ 14 -> 12 (lọc ứng viên chặt hơn)
PHASH_DUP_MAX_HD        = 6    # giảm từ 8  -> 6  (coi gần-trùng khi HD nhỏ)

ORB_N_FEATURES          = 1000 # tăng từ 800 -> 1000 (nhiều keypoints hơn)
ORB_GOOD_RATIO          = 0.75
ORB_MIN_GOOD_MATCHES    = 20   # tăng 15 -> 20
ORB_MIN_SHAPE_SCORE     = 0.40 # tăng 0.32 -> 0.40

HU_MAX_DISTANCE         = 0.010 # giảm 0.015 -> 0.010

WEIGHT_PHASH            = 0.5
WEIGHT_SHAPE            = 0.5
UNIFIED_THRESHOLD       = 0.80  # tăng 0.72 -> 0.80

RESIZE_LONG_EDGE        = 640    # 0 để tắt resize
USE_PROGRESS            = True


# ===================== TIỆN ÍCH =====================
def ensure_outdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def file_md5(path: Path, chunk: int = 8192) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def file_sha1(path: Path, chunk: int = 8192) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def safe_open_image(path: Path) -> Optional[Image.Image]:
    try:
        img = Image.open(path)
        img = img.convert("RGB")
        return img
    except Exception:
        return None

def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def resize_long_edge_cv(img_bgr: np.ndarray, long_edge: int) -> np.ndarray:
    if long_edge <= 0: return img_bgr
    h, w = img_bgr.shape[:2]
    m = max(h, w)
    if m <= long_edge: return img_bgr
    scale = long_edge / float(m)
    nw, nh = int(round(w*scale)), int(round(h*scale))
    return cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

def compute_phash(img_pil: Image.Image) -> imagehash.ImageHash:
    return imagehash.phash(img_pil)

def compute_hu_moments(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        m = cv2.moments(gray)
    else:
        cnt = max(contours, key=cv2.contourArea)
        m = cv2.moments(cnt)
    hu = cv2.HuMoments(m).flatten()
    with np.errstate(all='ignore'):
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    return hu

def hu_distance(hu1: np.ndarray, hu2: np.ndarray) -> float:
    return float(np.linalg.norm(hu1 - hu2))

def compute_orb_des(img_bgr: np.ndarray, n_features: int = ORB_N_FEATURES):
    orb = cv2.ORB_create(nfeatures=n_features)
    kps, des = orb.detectAndCompute(img_bgr, None)
    return kps, des

def orb_match_score(des1, des2, kps1_count: int, kps2_count: int) -> Tuple[int, float]:
    if des1 is None or des2 is None or len(des1)==0 or len(des2)==0:
        return 0, 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < ORB_GOOD_RATIO * n.distance:
            good.append(m)
    min_kps = max(1, min(kps1_count, kps2_count))
    score = len(good) / float(min_kps)
    return len(good), float(score)

def unified_similarity(p_sim: float, shape_score: float, hu_dist: float) -> float:
    hu_sim = math.exp(- (hu_dist / max(1e-6, HU_MAX_DISTANCE)) )
    shape_sim = min(1.0, shape_score / max(1e-6, ORB_MIN_SHAPE_SCORE))
    return WEIGHT_PHASH * p_sim + WEIGHT_SHAPE * (0.5*shape_sim + 0.5*hu_sim)

def is_duplicate(hd_phash: int, p_sim: float, good_matches: int, shape_score: float, hu_dist: float) -> bool:
    if hd_phash <= PHASH_DUP_MAX_HD:
        return True
    if (good_matches >= ORB_MIN_GOOD_MATCHES) and (shape_score >= ORB_MIN_SHAPE_SCORE) and (hu_dist <= HU_MAX_DISTANCE):
        return True
    u = unified_similarity(p_sim, shape_score, hu_dist)
    return u >= UNIFIED_THRESHOLD

def resolve_path(rel_or_abs: str) -> Path:
    """
    image_path trong Excel là tương đối: "images_png/<...>.png"
    Ta ghép với IMAGE_BASE_DIR = <ROOT>/pre-processing
    """
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return IMAGE_BASE_DIR / rel_or_abs

def log_signature_stats(sig_df: pd.DataFrame):
    n = len(sig_df)
    n_exist = int(sig_df["exists"].fillna(False).sum()) if "exists" in sig_df else 0
    n_open  = int(sig_df["open_ok"].fillna(False).sum()) if "open_ok" in sig_df else 0
    n_phash = int(sig_df["phash"].notna().sum()) if "phash" in sig_df else 0
    print(f"[STATS] images: {n} | exists: {n_exist} | open_ok: {n_open} | phash_ok: {n_phash}")


# ===================== PIPELINE =====================
def load_image_list_from_excel(xlsx_path: Path) -> pd.DataFrame:
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Không thấy file Excel: {xlsx_path}")
    df = pd.read_excel(xlsx_path)
    if IMAGE_COL not in df.columns:
        raise ValueError(f"Không thấy cột '{IMAGE_COL}' trong {xlsx_path.name}")
    if ID_COL not in df.columns:
        df[ID_COL] = df.index.astype(str)

    df[IMAGE_COL] = df[IMAGE_COL].astype(str)

    # Lọc theo phần mở rộng .png
    def ok_ext(p):
        try:
            return Path(p).suffix.lower() in VALID_EXTS
        except Exception:
            return False

    before = len(df)
    df = df[df[IMAGE_COL].map(ok_ext)].drop_duplicates(subset=[IMAGE_COL]).reset_index(drop=True)
    after = len(df)
    if after < before:
        print(f"[INFO] Lọc theo phần mở rộng: {before} -> {after}")

    return df[[ID_COL, IMAGE_COL]]

def compute_signatures(items: pd.DataFrame) -> pd.DataFrame:
    records = []
    it = items.itertuples(index=False)
    if USE_PROGRESS:
        it = tqdm(it, total=len(items), desc="Tính chữ ký ảnh")

    for row in it:
        rec = {ID_COL: getattr(row, ID_COL), IMAGE_COL: getattr(row, IMAGE_COL)}
        p = resolve_path(getattr(row, IMAGE_COL))

        if not p.exists():
            rec.update(dict(exists=False, open_ok=False, width=None, height=None,
                            md5=None, sha1=None, phash=None, phash_hex=None,
                            hu=None, orb_kps=0, orb_des=None))
            records.append(rec); continue

        rec["exists"] = True
        try:
            rec["md5"] = file_md5(p)
            rec["sha1"] = file_sha1(p)
        except Exception:
            rec["md5"] = None; rec["sha1"] = None

        img_pil = safe_open_image(p)
        if img_pil is None:
            rec.update(dict(open_ok=False, width=None, height=None,
                            phash=None, phash_hex=None, hu=None, orb_kps=0, orb_des=None))
            records.append(rec); continue

        rec["open_ok"] = True
        rec["width"], rec["height"] = img_pil.size

        # pHash
        try:
            ph = imagehash.phash(img_pil)
            rec["phash"] = int(str(ph), 16)
            rec["phash_hex"] = str(ph)
        except Exception:
            rec["phash"] = None; rec["phash_hex"] = None

        # ORB + Hu
        try:
            cv = pil_to_cv(img_pil)
            if RESIZE_LONG_EDGE > 0:
                cv = resize_long_edge_cv(cv, RESIZE_LONG_EDGE)
            hu = compute_hu_moments(cv)
            kps, des = compute_orb_des(cv)
            rec["hu"] = hu.tolist()
            rec["orb_kps"] = len(kps) if kps is not None else 0
            rec["orb_des"] = des
        except Exception:
            rec["hu"] = None; rec["orb_kps"] = 0; rec["orb_des"] = None

        records.append(rec)

    sig = pd.DataFrame(records)
    log_signature_stats(sig)
    return sig

def build_pairs(sig: pd.DataFrame) -> pd.DataFrame:
    EXPECTED_COLS = [
        "id1","id2","path1","path2",
        "hamming_phash","p_sim","orb_good","shape_score",
        "hu_dist","unified_sim","duplicate","reason"
    ]
    pair_records = []

    if "phash" not in sig:
        return pd.DataFrame(columns=EXPECTED_COLS)

    valid = sig[(sig.get("exists", False)==True) & (sig.get("open_ok", False)==True) & sig["phash"].notna()]
    if valid.empty:
        print("[WARN] Không có ảnh hợp lệ (exists & open_ok & có phash). Trả về pairs rỗng.")
        return pd.DataFrame(columns=EXPECTED_COLS)

    ids  = valid[ID_COL].tolist()
    phs  = valid["phash"].tolist()
    hus  = [np.array(v) if isinstance(v, list) else None for v in valid["hu"].tolist()]
    dess = valid["orb_des"].tolist()
    kpss = valid["orb_kps"].tolist()
    paths= valid[IMAGE_COL].tolist()

    # Trùng tuyệt đối theo md5/sha1
    for col in ["md5", "sha1"]:
        if col in sig.columns:
            tmp = sig[sig[col].notna()].groupby(col)[ID_COL].apply(list)
            for _, group in tmp.items():
                if len(group) >= 2:
                    for i in range(len(group)):
                        for j in range(i+1, len(group)):
                            a, b = group[i], group[j]
                            ra = sig.loc[sig[ID_COL]==a].iloc[0]
                            rb = sig.loc[sig[ID_COL]==b].iloc[0]
                            pair_records.append({
                                "id1": a, "id2": b,
                                "path1": ra[IMAGE_COL], "path2": rb[IMAGE_COL],
                                "hamming_phash": 0, "p_sim": 1.0,
                                "orb_good": 999, "shape_score": 1.0,
                                "hu_dist": 0.0, "unified_sim": 1.0,
                                "duplicate": True, "reason": f"exact_{col}"
                            })

    # Ứng viên theo pHash: sort và mở cửa sổ lân cận
    order = np.argsort(phs)
    phs_sorted   = [phs[i]   for i in order]
    ids_sorted   = [ids[i]   for i in order]
    hu_sorted    = [hus[i]   for i in order]
    des_sorted   = [dess[i]  for i in order]
    kps_sorted   = [kpss[i]  for i in order]
    path_sorted  = [paths[i] for i in order]

    def hamming64(a: int, b: int) -> int:
        return bin(a ^ b).count("1")

    it = range(len(phs_sorted))
    if USE_PROGRESS:
        it = tqdm(it, total=len(phs_sorted), desc="So khớp ứng viên (pHash→ORB)")

    for i in it:
        ph_i, id_i, hu_i, des_i, kps_i = phs_sorted[i], ids_sorted[i], hu_sorted[i], des_sorted[i], kps_sorted[i]
        for j in range(i+1, min(len(phs_sorted), i+1+64)):
            ph_j, id_j, hu_j, des_j, kps_j = phs_sorted[j], ids_sorted[j], hu_sorted[j], des_sorted[j], kps_sorted[j]

            hd = hamming64(ph_i, ph_j)
            if hd > PHASH_CANDIDATE_MAX_HD:
                if (ph_j - ph_i) > 0 and hd >= 32:
                    break
                continue

            p_sim = 1.0 - (hd/64.0)
            if des_i is not None and des_j is not None and kps_i>0 and kps_j>0:
                good, shape_sc = orb_match_score(des_i, des_j, kps_i, kps_j)
            else:
                good, shape_sc = 0, 0.0
            hu_dist = hu_distance(hu_i, hu_j) if (hu_i is not None and hu_j is not None) else 1e9

            dup_flag = is_duplicate(hd, p_sim, good, shape_sc, hu_dist)
            u_sim = unified_similarity(p_sim, shape_sc, hu_dist)

            pair_records.append({
                "id1": id_i, "id2": id_j,
                "path1": path_sorted[i], "path2": path_sorted[j],
                "hamming_phash": int(hd), "p_sim": float(p_sim),
                "orb_good": int(good), "shape_score": float(shape_sc),
                "hu_dist": float(hu_dist), "unified_sim": float(u_sim),
                "duplicate": bool(dup_flag), "reason": "phash_orb_hu"
            })

    return pd.DataFrame(pair_records, columns=EXPECTED_COLS)

def label_groups(items: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    # Không có pairs → không có nhóm
    if pairs is None or pairs.empty or ("duplicate" not in pairs.columns):
        out = items.copy()
        out["dup_group_id"] = -1
        out["dup_representative"] = None
        out["duplicated"] = False
        return out

    G = nx.Graph()
    for r in items.itertuples(index=False):
        G.add_node(getattr(r, ID_COL))

    for r in pairs[pairs["duplicate"] == True].itertuples(index=False):
        G.add_edge(r.id1, r.id2, weight=r.unified_sim)

    comps = list(nx.connected_components(G))
    group_map: Dict[str,int] = {}
    gid = 0
    for comp in comps:
        gid += 1
        for node in comp:
            group_map[node] = gid

    out = items.copy()
    out["dup_group_id"] = out[ID_COL].map(group_map).fillna(-1).astype(int)

    # tạm chọn đại diện theo path; sẽ thay bằng ảnh có độ phân giải lớn nhất
    rep_map: Dict[int,str] = {}
    for gid_val, sub in out.groupby("dup_group_id"):
        if gid_val == -1: continue
        rep_map[gid_val] = sub[IMAGE_COL].sort_values().iloc[0]
    out["dup_representative"] = out["dup_group_id"].map(rep_map)
    out["duplicated"] = out.duplicated(subset=["dup_group_id"]) & (out["dup_group_id"] != -1)
    return out

def pick_representative_by_resolution(labeled: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    wh = sig[[ID_COL, "width", "height"]].copy()
    wh["area"] = wh["width"].fillna(0) * wh["height"].fillna(0)
    merged = labeled.merge(wh, on=ID_COL, how="left")

    rep_map: Dict[int,str] = {}
    for gid_val, sub in merged.groupby("dup_group_id", dropna=False):
        if gid_val == -1: continue
        idx = sub["area"].fillna(0).astype(float).idxmax()
        rep_map[gid_val] = merged.loc[idx, IMAGE_COL]

    merged["dup_representative"] = merged["dup_group_id"].map(rep_map).fillna(merged.get("dup_representative"))
    merged["duplicated"] = merged.duplicated(subset=["dup_group_id"]) & (merged["dup_group_id"] != -1)
    return merged.drop(columns=["area"], errors="ignore")

def enrich_with_dup_info(labeled: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Thêm:
    - is_duplicate (1/0)
    - dup_id_1, dup_path_1, dup_id_2, dup_path_2, ...
    """
    labeled = labeled.copy()
    labeled["is_duplicate"] = 0

    if pairs is None or pairs.empty or ("duplicate" not in pairs.columns):
        return labeled

    dup_pairs = pairs[pairs["duplicate"] == True]
    if dup_pairs.empty:
        return labeled

    # gom theo id: tất cả cặp trùng của từng ảnh
    mapping: Dict[str, list] = {}
    for r in dup_pairs.itertuples(index=False):
        mapping.setdefault(r.id1, []).append((r.id2, r.path2))
        mapping.setdefault(r.id2, []).append((r.id1, r.path1))

    max_refs = max((len(v) for v in mapping.values()), default=0)
    for i in range(1, max_refs+1):
        labeled[f"dup_id_{i}"] = None
        labeled[f"dup_path_{i}"] = None

    for idx, row in labeled.iterrows():
        idv = row[ID_COL]
        if idv in mapping:
            labeled.at[idx, "is_duplicate"] = 1
            for j, (oid, opath) in enumerate(mapping[idv], start=1):
                labeled.at[idx, f"dup_id_{j}"] = oid
                labeled.at[idx, f"dup_path_{j}"] = opath

    return labeled

def merge_labels_into_encoded(labeled: pd.DataFrame, input_xlsx: Path) -> pd.DataFrame:
    """
    Gộp nhãn vào encoded_data.xlsx đầy đủ cột (không chỉ ID & image_path).
    Ưu tiên join theo 'id', fallback 'image_path'.
    """
    enc_full = pd.read_excel(input_xlsx)
    # quyết định khóa join
    if (ID_COL in enc_full.columns) and (ID_COL in labeled.columns):
        keys = [ID_COL]
    elif (IMAGE_COL in enc_full.columns) and (IMAGE_COL in labeled.columns):
        keys = [IMAGE_COL]
    else:
        # cố gắng join theo cả hai nếu có
        keys = [c for c in [ID_COL, IMAGE_COL] if (c in enc_full.columns and c in labeled.columns)]
        if not keys:
            raise RuntimeError("Không tìm được khóa join chung để gộp nhãn (cần 'id' hoặc 'image_path').")

    # căn kiểu an toàn
    for k in keys:
        enc_full[k] = enc_full[k].astype(str)
        labeled[k]  = labeled[k].astype(str)

    merged = enc_full.merge(labeled.drop_duplicates(subset=keys), on=keys, how="left", validate="m:1")
    return merged

def run_pipeline():
    print(f"[INFO] Đọc từ Excel: {INPUT_XLSX}")
    ensure_outdir(OUTPUT_DIR)

    # 1) Đọc danh sách ảnh (ID & image_path)
    items = load_image_list_from_excel(INPUT_XLSX)
    print(f"[INFO] Số dòng: {len(items)} | IMAGE_BASE_DIR: {IMAGE_BASE_DIR}")

    # 2) Tính chữ ký
    sig = compute_signatures(items)
    if sig.empty or (sig["phash"].notna().sum() == 0):
        print("[STOP] Không có ảnh hợp lệ để so khớp (không mở được hoặc không tính được pHash).")
        # Lưu rỗng hợp lệ
        pairs = pd.DataFrame(columns=["id1","id2","path1","path2","hamming_phash","p_sim","orb_good","shape_score","hu_dist","unified_sim","duplicate","reason"])
        labeled = label_groups(items, pairs)
        labeled = enrich_with_dup_info(labeled, pairs)
        # ghi nhãn rỗng ra dup_image/ và gộp vào processing_data_1.xlsx
        labeled.to_excel(OUTPUT_XLSX, index=False)
        pairs.to_excel(PAIRS_XLSX, index=False)
        merged = merge_labels_into_encoded(labeled, INPUT_XLSX)
        merged.to_excel(MERGED_XLSX, index=False)
        with open(TXT_REPORT, "w", encoding="utf-8") as f:
            f.write("Không có ảnh hợp lệ để so khớp.\n")
        print(f"[DONE] Lưu rỗng: {OUTPUT_XLSX} | {PAIRS_XLSX} | {TXT_REPORT}")
        print(f"[DONE] Lưu dữ liệu gộp nhãn: {MERGED_XLSX}")
        return

    # 3) Sinh cặp và chấm điểm
    pairs = build_pairs(sig)

    # 4) Gán nhãn nhóm
    labeled = label_groups(items, pairs)
    labeled = pick_representative_by_resolution(labeled, sig)

    # 5) Điểm giống cao nhất mỗi ảnh (tham khảo)
    if not pairs.empty:
        m1 = pairs.groupby("id1")["unified_sim"].max().rename("dup_score_max1")
        m2 = pairs.groupby("id2")["unified_sim"].max().rename("dup_score_max2")
        labeled = labeled.merge(m1, left_on=ID_COL, right_index=True, how="left")
        labeled = labeled.merge(m2, left_on=ID_COL, right_index=True, how="left")
        labeled["dup_score_max"] = labeled[["dup_score_max1", "dup_score_max2"]].max(axis=1)
        labeled = labeled.drop(columns=["dup_score_max1","dup_score_max2"])
    else:
        labeled["dup_score_max"] = np.nan

    # 6) Làm giàu thông tin trùng: is_duplicate, dup_id_*, dup_path_*
    labeled = enrich_with_dup_info(labeled, pairs)

    # 7) Lưu file kết quả phát hiện trùng (trong dup_image/)
    labeled.to_excel(OUTPUT_XLSX, index=False)
    pairs.sort_values(["duplicate","unified_sim","p_sim"], ascending=[False, False, False]).to_excel(PAIRS_XLSX, index=False)

    with open(TXT_REPORT, "w", encoding="utf-8") as f:
        groups = labeled[labeled["dup_group_id"]!=-1]["dup_group_id"]
        total_groups = groups.nunique()
        multi_groups = (labeled["dup_group_id"].value_counts(dropna=True)>=2).sum()
        f.write(f"Tổng ảnh: {len(labeled)}\n")
        f.write(f"Số nhóm trùng (>=2 ảnh): {multi_groups} / tổng group {total_groups}\n\n")
        for gid_val, sub in labeled.groupby("dup_group_id"):
            if gid_val == -1: continue
            sub = sub.sort_values(IMAGE_COL)
            if len(sub) < 2: continue
            f.write(f"Nhóm {gid_val} (n={len(sub)}):\n")
            for r in sub.itertuples(index=False):
                f.write(f"  - {getattr(r, IMAGE_COL)}\n")
            f.write("\n")

    # 8) Gộp nhãn vào encoded_data.xlsx đầy đủ cột -> processing_data_1.xlsx
    merged = merge_labels_into_encoded(labeled, INPUT_XLSX)
    merged.to_excel(MERGED_XLSX, index=False)

    print(f"[DONE] Lưu nhãn: {OUTPUT_XLSX}")
    print(f"[DONE] Lưu cặp:  {PAIRS_XLSX}")
    print(f"[DONE] Báo cáo:  {TXT_REPORT}")
    print(f"[DONE] Dữ liệu gộp nhãn: {MERGED_XLSX}")


if __name__ == "__main__":
    run_pipeline()
