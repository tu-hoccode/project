import os, time, hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from PIL import Image, ImageOps

# ===== CONFIG =====
INPUT_XLSX   = "pre_data_1.xlsx"
PATH_COL     = "path"                    # cột sẽ được ghi đè
BASE_DIR     = Path("../crawl_data")     # thư mục gốc ảnh
OUTPUT_ROOT  = Path("images_png")
OUTPUT_XLSX  = "pre_data_2.xlsx"
MANIFEST_XLSX= "image_manifest.xlsx"
ERRORS_CSV   = "image_errors.csv"
MAX_WORKERS  = 8

TARGET_SIZE   = (224, 224)
RESIZE_METHOD = "padding"   # 'padding' | 'crop' | 'stretch'
RESIZE_ENABLED = True

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ===== UTILS =====
def clean_rel_path(raw: str) -> str:
    s = str(raw).strip().replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    for prefix in ("crawl_data/", "./crawl_data/"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    if s.startswith("/"):
        s = s[1:]
    return s

def resolve_src_path(p_from_excel: str):
    raw = Path(str(p_from_excel).strip())
    cleaned = clean_rel_path(p_from_excel)
    cand1 = BASE_DIR / cleaned
    if cand1.exists():
        return cand1
    if not cleaned.startswith("images/"):
        cand2 = BASE_DIR / "images" / cleaned
        if cand2.exists():
            return cand2
    if raw.exists():
        return raw
    return None

def ensure_rgb(img):
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB") if img.mode != "RGB" else img

def resize_padding(img, target):
    tw, th = target
    iw, ih = img.size
    img_ratio = iw / ih
    tgt_ratio = tw / th
    if img_ratio > tgt_ratio:
        new_w = tw
        new_h = int(new_w / img_ratio)
    else:
        new_h = th
        new_w = int(new_h * img_ratio)
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (tw, th), (255, 255, 255))
    canvas.paste(img_resized, ((tw - new_w)//2, (th - new_h)//2))
    return canvas

def resize_crop(img, target):
    tw, th = target
    iw, ih = img.size
    img_ratio = iw / ih
    tgt_ratio = tw / th
    if img_ratio > tgt_ratio:
        new_h = th
        new_w = int(new_h * img_ratio)
        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        left = (new_w - tw)//2
        return img_resized.crop((left, 0, left + tw, th))
    else:
        new_w = tw
        new_h = int(new_w / img_ratio)
        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        top = (new_h - th)//2
        return img_resized.crop((0, top, tw, top + th))

def apply_resize(img):
    if not RESIZE_ENABLED:
        return img
    if RESIZE_METHOD == "padding":
        return resize_padding(img, TARGET_SIZE)
    elif RESIZE_METHOD == "crop":
        return resize_crop(img, TARGET_SIZE)
    return img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

def build_out_path(rel_from_excel: str):
    cleaned = clean_rel_path(rel_from_excel)
    if cleaned.startswith("images/"):
        cleaned = cleaned[len("images/"):]
    subdir = Path(cleaned).parent
    out_dir = OUTPUT_ROOT / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / (Path(cleaned).stem + ".png")

def file_md5(p: Path):
    try:
        return hashlib.md5(p.read_bytes()).hexdigest()
    except:
        return ""

# ===== WORKER =====
def convert_worker(args):
    idx, rel = args
    try:
        if pd.isna(rel) or str(rel).strip() == "":
            return idx, rel, "empty"
        src = resolve_src_path(str(rel))
        if src is None or not src.exists():
            return idx, rel, f"missing:{rel}"
        with Image.open(src) as im:
            im = ensure_rgb(im)
            im = apply_resize(im)
            out_path = build_out_path(str(rel))
            im.save(out_path, "PNG", optimize=True)
        return idx, str(out_path.as_posix()), ""
    except Exception as e:
        return idx, rel, f"error:{type(e).__name__}:{e}"

# ===== MAIN =====
start = time.time()
df = pd.read_excel(INPUT_XLSX)
if PATH_COL not in df.columns:
    raise ValueError(f"File {INPUT_XLSX} không có cột '{PATH_COL}'")

tasks = [(i, v) for i, v in df[PATH_COL].items()]
new_paths = [pd.NA] * len(df)
errors = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futures = {ex.submit(convert_worker, t): t[0] for t in tasks}
    for fut in as_completed(futures):
        idx, new_path, err = fut.result()
        new_paths[idx] = new_path
        if err:
            errors.append({"index": idx, "path_src": df.at[idx, PATH_COL], "error": err})

# Ghi đè cột path
df[PATH_COL] = new_paths
df.to_excel(OUTPUT_XLSX, index=False)

elapsed = time.time() - start
ok_count = sum(isinstance(p, str) and len(p) > 0 for p in new_paths)
print(f"✅ Converted & resized: {ok_count}/{len(df)}")
print(f"⚠️ Missing/failed: {len(errors)}")
print(f"📁 Output: {OUTPUT_ROOT}/")
print(f"💾 Excel saved: {OUTPUT_XLSX}")
print(f"⏱️ Time: {elapsed:.2f}s | method={RESIZE_METHOD} | size={TARGET_SIZE}")

# Lưu log lỗi & manifest
if errors:
    pd.DataFrame(errors).to_csv(ERRORS_CSV, index=False)
    print(f"📝 Lưu log lỗi vào: {ERRORS_CSV}")

rows = []
for p in OUTPUT_ROOT.rglob("*.png"):
    try:
        from PIL import Image
        with Image.open(p) as im:
            w, h = im.size
            mode = im.mode
        rows.append({"path": str(p.as_posix()), "width": w, "height": h, "mode": mode, "md5": file_md5(p)})
    except Exception as e:
        rows.append({"path": str(p.as_posix()), "width": None, "height": None, "mode": None, "md5": "", "error": str(e)})
pd.DataFrame(rows).to_excel(MANIFEST_XLSX, index=False)
print(f"📊 Manifest lưu vào: {MANIFEST_XLSX}")