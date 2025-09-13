import pandas as pd
import numpy as np
from PIL import Image

df = pd.read_excel("process_data_4.xlsx")

# Bổ sung kích thước ảnh
def get_image_info(path):
    try:
        with Image.open(path) as im:
            w, h = im.size
            return w, h, w/h if h != 0 else None
    except:
        return None, None, None

df[["width", "height", "aspect_ratio"]] = df["path"].apply(lambda x: pd.Series(get_image_info(x)))

# Bổ sung màu trung bình
def mean_rgb(path):
    try:
        with Image.open(path) as im:
            arr = np.array(im.convert("RGB"))
            return arr.mean(axis=(0,1))  # [R,G,B]
    except:
        return [None, None, None]

df[["mean_R", "mean_G", "mean_B"]] = pd.DataFrame(df["path"].apply(mean_rgb).tolist())

# (Nếu có category/brand/title) thì normalize text
for col in ["category", "brand", "title"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

df.to_excel("process_data_5.xlsx", index=False)
print("✅ Saved enriched dataset → process_data_5.xlsx")
