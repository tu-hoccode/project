# Bước integration: gộp các file đã làm sạch thành 1 file duy nhất
# Đảm bảo file đầu ra có đúng 5 cột: category, id, title, url, path
# Nếu file nào thiếu cột thì thêm NA vào
import pandas as pd
from pathlib import Path

# Danh sách file cần gộp
FILES = [
    "data_cleaned/phone.xlsx",
    "data_cleaned/laptop.xlsx",
    "data_cleaned/camera.xlsx",
    "data_cleaned/speaker.xlsx",
    "data_cleaned/tv.xlsx",
]

OUTPUT_PATH = "data_pre_1.xlsx"
NEEDED = ["category", "id", "title", "url", "path"]

def run_integration(files, out_path):
    frames = []
    for f in files:
        p = Path(f)
        if not p.exists():
            print(f"Bỏ qua (không thấy file): {f}")
            continue

        df_temp = pd.read_excel(p)
        frames.append(df_temp)

    if not frames:
        raise ValueError("Không có file hợp lệ để gộp.")

    # Gộp tất cả
    df = pd.concat(frames, ignore_index=True)

    # Đảm bảo đủ 5 cột, cột nào thiếu thì thêm NA
    for col in NEEDED:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[NEEDED]

    # Lưu file
    df.to_excel(out_path, index=False)
    print(f"Đã lưu: {out_path} | Shape: {df.shape}")
    return df

if __name__ == "__main__":
    run_integration(FILES, OUTPUT_PATH)
