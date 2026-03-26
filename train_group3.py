import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# ==============================
# 0. 环境与随机种子设置
# ==============================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# ==============================
# 1. PH2 数据集定义 (修复了 ID 匹配逻辑)
# ==============================
class PH2Dataset(Dataset):
    def __init__(self, images_dir, masks_dir=None, augment=False):
        self.images = sorted([f for f in os.listdir(images_dir) if f.endswith('.bmp')])
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.augment = augment
        self.resize = transforms.Resize((256, 256))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(self.images_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img = self.resize(img)

        # 同步随机增强
        h_flip = self.augment and random.random() > 0.5
        v_flip = self.augment and random.random() > 0.5
        if h_flip: img = transforms.functional.hflip(img)
        if v_flip: img = transforms.functional.vflip(img)
        img = self.to_tensor(img)

        if self.masks_dir is None:
            return img, img_id

        # 兼容 .png 或 .bmp 的 Mask
        mask_path = os.path.join(self.masks_dir, img_id + ".png")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.masks_dir, img_id + ".bmp")
            
        mask = Image.open(mask_path).convert("L")
        mask = self.resize(mask)
        if h_flip: mask = transforms.functional.hflip(mask)
        if v_flip: mask = transforms.functional.vflip(mask)
        mask = self.to_tensor(mask)
        mask = torch.where(mask > 0.5, 1.0, 0.0)
        return img, mask, img_id

# ==============================
# 2. Attention U-Net 网络架构
# ==============================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))

class AttentionUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv1 = DoubleConv(in_ch, 64); self.conv2 = DoubleConv(64, 128)
        self.conv3 = DoubleConv(128, 256); self.conv4 = DoubleConv(256, 512)
        self.dropout = nn.Dropout2d(0.5)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2); self.att3 = AttentionBlock(256, 256, 128); self.upconv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2); self.att2 = AttentionBlock(128, 128, 64); self.upconv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2); self.att1 = AttentionBlock(64, 64, 32); self.upconv1 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x); p1 = self.maxpool(c1); c2 = self.conv2(p1); p2 = self.maxpool(c2)
        c3 = self.conv3(p2); p3 = self.maxpool(c3); c4 = self.dropout(self.conv4(p3))
        x3 = self.upconv3(torch.cat([self.up3(c4), self.att3(self.up3(c4), c3)], dim=1))
        x2 = self.upconv2(torch.cat([self.up2(x3), self.att2(self.up2(x3), c2)], dim=1))
        x1 = self.upconv1(torch.cat([self.up1(x2), self.att1(self.up1(x2), c1)], dim=1))
        return torch.sigmoid(self.out(x1))

# ==============================
# 3. 损失函数与工具
# ==============================
class DiceBCELoss(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, pred, target, smooth=1e-6):
        pred = pred.view(-1); target = target.view(-1)
        intersection = (pred * target).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        bce_loss = F.binary_cross_entropy(pred, target)
        return bce_loss + dice_loss

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(alpha).add_(p.data, alpha=1 - alpha)

def get_metrics(y_true, y_pred):
    y_pred = (y_pred > 0.5).float()
    tp = (y_true * y_pred).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    return dice.item(), iou.item()

def mc_dropout_inference(model, img, T=8):
    model.train() # 激活 Dropout 采样
    preds = torch.stack([model(img) for _ in range(T)])
    return preds.mean(0), preds.std(0)

# ==============================
# 4. 主训练流程 (优化权重分配)
# ==============================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res_dir = "Proposed_Optimized_Results"
    os.makedirs(res_dir, exist_ok=True)

    # 数据加载
    train_img, train_mask = "PH2_dataset_final/train/images", "PH2_dataset_final/train/masks"
    val_img, val_mask = "PH2_dataset_final/val/images", "PH2_dataset_final/val/masks"
    
    full_train_ds = PH2Dataset(train_img, train_mask, augment=True)
    indices = np.random.RandomState(42).choice(len(full_train_ds), 20, replace=False)
    unlabeled_indices = list(set(range(len(full_train_ds))) - set(indices))
    
    l_loader = DataLoader(Subset(full_train_ds, indices), batch_size=4, shuffle=True)
    u_loader = DataLoader(Subset(PH2Dataset(train_img, augment=True), unlabeled_indices), batch_size=4, shuffle=True)
    v_loader = DataLoader(PH2Dataset(val_img, val_mask), batch_size=4)

    # 网络初始化
    student = AttentionUNet().to(device)
    teacher = AttentionUNet().to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters(): p.requires_grad = False

    optimizer = optim.Adam(student.parameters(), lr=1e-4)
    # 增加余弦退火调度器以优化后期指标
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = DiceBCELoss()

    best_dice, global_step = 0, 0
    history = {"loss": [], "dice": []}

    print(">>> 启动优化版 Proposed 训练 (Soft-weighting Uncertainty)...")
    for epoch in range(100):
        student.train()
        epoch_loss, u_iter = 0, iter(u_loader)
        consist_weight = 0.5 * min(1.0, epoch / 40.0)

        for imgs_l, masks_l, _ in l_loader:
            imgs_l, masks_l = imgs_l.to(device), masks_l.to(device)
            try: imgs_u, _ = next(u_iter)
            except: u_iter = iter(u_loader); imgs_u, _ = next(u_iter)
            imgs_u = imgs_u.to(device)

            # 1. 有标签监督
            optimizer.zero_grad()
            loss_l = criterion(student(imgs_l), masks_l)

            # 2. 软加权一致性 (核心优化)
            with torch.no_grad():
                mu_u, sigma_u = mc_dropout_inference(teacher, imgs_u, T=8)
                # 优化点：计算软置信度权重，不确定性越大，权重越低
                # 使用指数衰减：w = exp(-sigma^2 / beta)
                conf_weight = torch.exp(-torch.pow(sigma_u, 2) / 0.01)
                pseudo_label = (mu_u > 0.5).float()

            loss_u = (conf_weight * F.mse_loss(student(imgs_u), pseudo_label, reduction='none')).mean()
            
            total_loss = loss_l + consist_weight * loss_u
            total_loss.backward()
            optimizer.step()
            
            global_step += 1
            update_ema_variables(student, teacher, 0.99, global_step)
            epoch_loss += total_loss.item()

        # 验证
        teacher.eval()
        v_d = 0
        with torch.no_grad():
            for im, mk, _ in v_loader:
                d, _ = get_metrics(mk.to(device), teacher(im.to(device)))
                v_d += d
        avg_v = v_d / len(v_loader)
        scheduler.step()
        
        history["loss"].append(epoch_loss / len(l_loader)); history["dice"].append(avg_v)
        if avg_v > best_dice:
            best_dice = avg_v
            torch.save(teacher.state_dict(), "SOTA_Proposed.pth")
            print(f"Epoch {epoch}: New Best Val Dice {best_dice:.4f}")

    # --- 最终评估与出图 ---
    print("\n>>> 开始最终测试集评估...")
    test_ds = PH2Dataset("PH2_dataset_final/test/images", "PH2_dataset_final/test/masks")
    test_loader = DataLoader(test_ds, batch_size=1)
    teacher.load_state_dict(torch.load("SOTA_Proposed.pth"))
    teacher.eval()

    final_dice, final_iou = [], []
    for i, (img, mask, img_id) in enumerate(test_loader):
        img_gpu = img.to(device)
        with torch.no_grad():
            pred = teacher(img_gpu)
        d, iou = get_metrics(mask.to(device), pred)
        final_dice.append(d); final_iou.append(iou)

        # 保存定性分析图
        img_np = (img[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        mask_np = (pred[0,0].cpu().numpy() > 0.5).astype(np.uint8) * 255
        overlay = cv2.addWeighted(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), 0.7, 
                                  cv2.applyColorMap(mask_np, cv2.COLORMAP_JET), 0.3, 0)
        cv2.imwrite(f"{res_dir}/{img_id[0]}_pred.png", overlay)

    print(f"\n======== Proposed 最终成绩 ========")
    print(f"Average Dice: {np.mean(final_dice):.4f}")
    print(f"Average IoU: {np.mean(final_iou):.4f}")

if __name__ == "__main__":
    main()