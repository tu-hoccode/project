# %% [markdown]
# ## Normalization-only (PyTorch) — read from pre_data_2.xlsx
# - Không encoding, chỉ chuẩn hóa pixel khi load ảnh
# - ImageNet mean/std (chuẩn cho ResNet/VGG/EfficientNet)
# - Kiểm tra 1 batch: shape & range sau normalize

import pandas as pd
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ========= CONFIG =========
META_FILE   = "pre_data_2.xlsx"   # file sau Transformation
PATH_COL    = "path"              # cột đường dẫn ảnh
BATCH_SIZE  = 32
NUM_WORKERS = 0                   # an toàn trên macOS/Windows; tăng lên 2/4 khi ổn

# ImageNet normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# Chỉ normalization (không augmentation)
normalize_tfms = transforms.Compose([
    transforms.ToTensor(),                          # [0,1]
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

class ImageCSVDataset(Dataset):
    def __init__(self, xlsx_path, path_col="path", transform=None):
        # đọc excel (không encode gì cả)
        self.df = pd.read_excel(xlsx_path)
        if path_col not in self.df.columns:
            raise ValueError(f"'{path_col}' column not found in {xlsx_path}")
        self.path_col = path_col
        self.transform = transform

        # lọc các dòng path rỗng để tránh lỗi
        self.df = self.df[self.df[self.path_col].astype(str).str.len() > 0].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        p = str(self.df.iloc[idx][self.path_col])
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Không mở được ảnh: {p} | {e}")
        if self.transform:
            img = self.transform(img)
        # Không trả label vì chưa encoding
        return img

def main():
    ds = ImageCSVDataset(META_FILE, path_col=PATH_COL, transform=normalize_tfms)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=NUM_WORKERS, persistent_workers=False)

    # Lấy 1 batch để kiểm tra
    batch = next(iter(loader))
    print(f"Batch shape: {batch.shape} (N, C, H, W)")
    print(f"dtype: {batch.dtype}")
    print(f"value range ≈ [{batch.min().item():.3f}, {batch.max().item():.3f}]  (đã Normalize)")

if __name__ == "__main__":
    main()
