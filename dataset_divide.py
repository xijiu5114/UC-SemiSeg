import os
import shutil
import random

# 已整理好的数据集路径
images_dir = r"PH2_processed\images"
masks_dir = r"PH2_processed\masks"

# 输出路径
output_dir = r"PH2_dataset_final"

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

os.makedirs(output_dir, exist_ok=True)
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, "masks"), exist_ok=True)

# 获取所有文件名（不带后缀）
all_files = [f[:-4] for f in os.listdir(images_dir) if f.endswith(".bmp")]
random.shuffle(all_files)

num_total = len(all_files)
num_train = int(train_ratio * num_total)
num_val = int(val_ratio * num_total)

train_files = all_files[:num_train]
val_files = all_files[num_train:num_train+num_val]
test_files = all_files[num_train+num_val:]

splits = {"train": train_files, "val": val_files, "test": test_files}

for split, files in splits.items():
    for f in files:
        # 复制图片
        shutil.copy(os.path.join(images_dir, f + ".bmp"),
                    os.path.join(output_dir, split, "images", f + ".bmp"))
        # 复制 mask
        shutil.copy(os.path.join(masks_dir, f + ".bmp"),
                    os.path.join(output_dir, split, "masks", f + ".bmp"))

print("数据集划分完成！")