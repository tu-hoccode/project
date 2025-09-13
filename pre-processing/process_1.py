import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# ---- Configuration ----
INPUT_XLSX = "pre_data_3.xlsx"
EMB_NPY = "embeddings_resnet50.npy"
PATHS_JSON = "image_paths.json"
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

# ---- Load metadata ----
df = pd.read_excel(INPUT_XLSX)
assert "path" in df.columns, "pre_data_3.xlsx thiếu cột 'path'"

# Kiểm tra và lọc file tồn tại
paths = df["path"].astype(str).tolist()
paths_valid = []
missing_files = []

for p in paths:
    if os.path.exists(p):
        paths_valid.append(p)
    else:
        missing_files.append(p)

print(f"✅ Found {len(paths_valid)} / {len(paths)} existing files.")
if missing_files:
    print(f"⚠️  Missing {len(missing_files)} files. First 5: {missing_files[:5]}")

# Kiểm tra nếu không có file nào tồn tại
if not paths_valid:
    raise ValueError("Không tìm thấy file ảnh nào để xử lý!")

# ---- tf.data pipeline với error handling ----
def load_and_preprocess(p):
    try:
        img = tf.io.read_file(p)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32)
        img = preprocess_input(img)
        return img
    except Exception as e:
        print(f"Error processing {p}: {e}")
        # Trả về tensor zeros để không làm gián đoạn batch
        return tf.zeros(IMG_SIZE + (3,), dtype=tf.float32)

# Tạo dataset
ds = tf.data.Dataset.from_tensor_slices(paths_valid)
ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ---- Load model ----
print("🔄 Loading ResNet50 model...")
base = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# ---- Extract embeddings ----
embs = []
for i, batch in enumerate(ds):
    if i % 10 == 0:
        print(f"Processing batch {i}...")
    
    feat = base(batch, training=False).numpy()
    
    # L2-normalize
    norms = np.linalg.norm(feat, axis=1, keepdims=True)
    feat = feat / (norms + 1e-8)
    embs.append(feat)

embeddings = np.concatenate(embs, axis=0)
print(f"✅ Extracted embeddings: {embeddings.shape}")

# ---- Save results ----
np.save(EMB_NPY, embeddings)
with open(PATHS_JSON, "w", encoding='utf-8') as f:
    json.dump(paths_valid, f, ensure_ascii=False)

print(f"✅ Saved embeddings -> {EMB_NPY}")
print(f"✅ Saved valid paths -> {PATHS_JSON}")
print("🎉 Extraction completed successfully!")