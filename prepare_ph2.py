import os
import shutil

source_dir = r"PH2 Dataset images"

target_images = r"D:\tkk\PH2_processed\images"
target_masks = r"D:\tkk\PH2_processed\masks"

os.makedirs(target_images, exist_ok=True)
os.makedirs(target_masks, exist_ok=True)

for folder in os.listdir(source_dir):

    folder_path = os.path.join(source_dir, folder)

    if not os.path.isdir(folder_path):
        continue

    # 读取 Dermoscopic_Image 文件夹
    image_dir = os.path.join(folder_path, f"{folder}_Dermoscopic_Image")
    mask_dir = os.path.join(folder_path, f"{folder}_lesion")

    if os.path.exists(image_dir):
        for file in os.listdir(image_dir):
            src = os.path.join(image_dir, file)
            dst = os.path.join(target_images, folder + ".bmp")
            shutil.copy(src, dst)

    if os.path.exists(mask_dir):
        for file in os.listdir(mask_dir):
            src = os.path.join(mask_dir, file)
            dst = os.path.join(target_masks, folder + ".bmp")
            shutil.copy(src, dst)

print("PH2 数据整理完成！")