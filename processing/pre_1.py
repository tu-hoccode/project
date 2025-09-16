import pandas as pd
from pathlib import Path

files = [
    "data_cleaned/phone.xlsx",
    "data_cleaned/laptop.xlsx",
    "data_cleaned/camera.xlsx",
    "data_cleaned/speaker.xlsx",
    "data_cleaned/tv.xlsx",
]

output = "pre_data_1.xlsx"

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

    # Gộp data
    df = pd.concat(frames, ignore_index=True)
    # Lưu file
    df.to_excel(out_path, index=False)
    print(f"Đã lưu: {out_path} | Shape: {df.shape}")
    return df

if __name__ == "__main__":
    run_integration(files, output)
