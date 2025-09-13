import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
import concurrent.futures
import csv

# ===================== CONFIG =====================
INPUT_XLSX    = "D:\DataMining\project\pre-processing\data_pre_2.xlsx"
PATH_COL      = "path"
BASE_DIR      = Path("../crawl_data")
OUTPUT_ROOT   = Path("images_png")
OUTPUT_XLSX   = "data_pre_3.xlsx"
MANIFEST_XLSX = "image_manifest.xlsx"
ERRORS_CSV    = "image_errors.csv"
MAX_WORKERS   = 8

TARGET_SIZE   = (224, 224)
RESIZE_METHOD = "padding"  # 'padding' | 'crop' | 'stretch'
RESIZE_ENABLED = True

# Normalization parameters (ImageNet)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Ensure output directory exists
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ===================== IMAGE PROCESSING =====================
def resize_image(img):
    """Resize an image based on the chosen method"""
    if not RESIZE_ENABLED:
        return img

    if RESIZE_METHOD == 'stretch':
        return img.resize(TARGET_SIZE, Image.LANCZOS)
    elif RESIZE_METHOD == 'crop':
        return ImageOps.fit(img, TARGET_SIZE, method=Image.LANCZOS, centering=(0.5, 0.5))
    elif RESIZE_METHOD == 'padding':
        img.thumbnail(TARGET_SIZE, Image.LANCZOS)
        delta_w = TARGET_SIZE[0] - img.size[0]
        delta_h = TARGET_SIZE[1] - img.size[1]
        padding = (delta_w//2, delta_h//2, delta_w - delta_w//2, delta_h - delta_h//2)
        return ImageOps.expand(img, padding, fill=(255, 255, 255))
    else:
        return img


def process_image(row, index):
    original_path = BASE_DIR / row[PATH_COL]
    new_path = OUTPUT_ROOT / f"{Path(row[PATH_COL]).stem}_{index}.png"

    try:
        with Image.open(original_path) as img:
            img = img.convert("RGB")  # ensure RGB
            img_resized = resize_image(img)

            # Convert to numpy & normalize (kept for ML usage)
            arr = np.array(img_resized, dtype=np.float32) / 255.0
            arr_norm = (arr - IMAGENET_MEAN) / IMAGENET_STD

            # Save normalized array separately (.npy), not as PNG
            np.save(new_path.with_suffix(".npy"), arr_norm)

            # Save processed PNG (resized only, for human inspection/debug)
            img_resized.save(new_path, "PNG")

            manifest_info = {
                "index": index,
                "original_path": str(original_path),
                "processed_path": str(new_path),
                "normalized_array": str(new_path.with_suffix(".npy")),
                "original_size": img.size,
                "processing_method": RESIZE_METHOD if RESIZE_ENABLED else "none"
            }
            return manifest_info, None

    except Exception as e:
        return None, {
            "index": index,
            "original_path": str(original_path),
            "error_message": str(e)
        }

# ===================== MAIN =====================
def main():
    df = pd.read_excel(INPUT_XLSX)

    manifests, errors = [], {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_image, row, idx): idx for idx, row in df.iterrows()}
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            manifest, error = future.result()
            if manifest:
                manifests.append(manifest)
                df.at[idx, PATH_COL] = manifest["processed_path"]
            elif error:
                errors[idx] = error

    # Save updated dataframe
    df.to_excel(OUTPUT_XLSX, index=False)

    # Save manifest
    if manifests:
        pd.DataFrame(manifests).to_excel(MANIFEST_XLSX, index=False)

    # Save errors
    if errors:
        with open(ERRORS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["index", "original_path", "error_message"])
            writer.writeheader()
            writer.writerows(errors.values())

    print(f"✅ Processing complete. Successful: {len(manifests)}, Errors: {len(errors)}")


if __name__ == "__main__":
    main()
