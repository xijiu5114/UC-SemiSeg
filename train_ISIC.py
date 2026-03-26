import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
from tqdm import tqdm

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
# 1. ISIC 2017 数据集定义
# ==============================
class ISIC2017Dataset(Dataset):
    def __init__(self, images_dir, masks_dir=None, augment=False):
        self.images = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.augment = augment
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(self.images_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        
        h_flip = self.augment and random.random() > 0.5
        v_flip = self.augment and random.random() > 0.5
        if h_flip: img = TF.hflip(img)
        if v_flip: img = TF.vflip(img)
        
        img_tensor = transforms.ToTensor()(img)
        img_tensor = self.normalize(img_tensor)

        if self.masks_dir is None:
            return img_tensor, img_id

        mask_path = os.path.join(self.masks_dir, img_id + ".png")
        mask = Image.open(mask_path).convert("L")
        if h_flip: mask = TF.hflip(mask)
        if v_flip: mask = TF.vflip(mask)
        
        mask_tensor = transforms.ToTensor()(mask)
        mask_tensor = torch.where(mask_tensor > 0.5, 1.0, 0.0)
        return img_tensor, mask_tensor, img_id

# ==============================
# 2. Attention U-Net 架构
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
# 3. 损失函数与 EMA
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
    model.train()
    preds = torch.stack([model(img) for _ in range(T)])
    return preds.mean(0), preds.std(0)

# ==============================
# 4. 主训练流程 (10% 标签设定)
# ==============================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res_dir = "ISIC2017_10Percent_Results"
    os.makedirs(res_dir, exist_ok=True)

    data_root = "ISIC2017_Dataset_Processed_256"
    train_img = os.path.join(data_root, "train/images")
    train_mask = os.path.join(data_root, "train/masks")
    test_img = os.path.join(data_root, "test/images")
    test_mask = os.path.join(data_root, "test/masks")

    # 加载数据集
    full_train_ds = ISIC2017Dataset(train_img, train_mask, augment=True)
    n_val = int(len(full_train_ds) * 0.1)
    n_train_total = len(full_train_ds) - n_val
    train_set, val_set = random_split(full_train_ds, [n_train_total, n_val], generator=torch.Generator().manual_seed(42))

    # 【关键修改】：选择 200 张作为有标签数据 (10% 比例)
    num_labeled = 200 
    indices = np.random.RandomState(42).choice(len(train_set), num_labeled, replace=False)
    unlabeled_indices = list(set(range(len(train_set))) - set(indices))
    
    # Batch Size 调整：有标签和无标签各取 4，总 Batch 为 8
    l_loader = DataLoader(Subset(train_set, indices), batch_size=4, shuffle=True, drop_last=True)
    u_loader = DataLoader(Subset(ISIC2017Dataset(train_img, augment=True), unlabeled_indices), batch_size=4, shuffle=True, drop_last=True)
    v_loader = DataLoader(val_set, batch_size=4)
    t_loader = DataLoader(ISIC2017Dataset(test_img, test_mask), batch_size=1)

    student = AttentionUNet().to(device)
    teacher = AttentionUNet().to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters(): p.requires_grad = False

    optimizer = optim.Adam(student.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = DiceBCELoss()

    best_dice, global_step = 0, 0
    print(f">>> 开始 10% 标签训练 | 有标签: {num_labeled}张 | 无标签: {len(unlabeled_indices)}张")

    for epoch in range(100):
        student.train()
        u_iter = iter(u_loader)
        consist_weight = 0.5 * min(1.0, epoch / 40.0)

        pbar = tqdm(l_loader, desc=f"Epoch {epoch}")
        for imgs_l, masks_l, _ in pbar:
            imgs_l, masks_l = imgs_l.to(device), masks_l.to(device)
            try: imgs_u, _ = next(u_iter)
            except: u_iter = iter(u_loader); imgs_u, _ = next(u_iter)
            imgs_u = imgs_u.to(device)

            # 1. 监督损失
            optimizer.zero_grad()
            preds_l = student(imgs_l)
            loss_l = criterion(preds_l, masks_l)

            # 2. SRF 一致性损失
            with torch.no_grad():
                mu_u, sigma_u = mc_dropout_inference(teacher, imgs_u, T=8)
                # 软权重映射
                conf_weight = torch.exp(-torch.pow(sigma_u, 2) / 0.02)
                pseudo_label = (mu_u > 0.5).float()

            preds_u = student(imgs_u)
            loss_u = (conf_weight * F.mse_loss(preds_u, pseudo_label, reduction='none')).mean()
            
            total_loss = loss_l + consist_weight * loss_u
            total_loss.backward()
            optimizer.step()
            
            global_step += 1
            update_ema_variables(student, teacher, 0.999, global_step) # 调高 EMA Alpha 增加稳定性
            pbar.set_postfix({"Loss": f"{total_loss.item():.4f}"})

        # 验证
        teacher.eval()
        val_dice_total = 0
        with torch.no_grad():
            for im, mk, _ in v_loader:
                d, _ = get_metrics(mk.to(device), teacher(im.to(device)))
                val_dice_total += d
        avg_v = val_dice_total / len(v_loader)
        scheduler.step()
        
        print(f"--- Epoch {epoch} | Val Dice: {avg_v:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if avg_v > best_dice:
            best_dice = avg_v
            torch.save(teacher.state_dict(), "ISIC2017_Best_10Percent.pth")
            print(f"!!! 发现更好的模型，已保存")

    # 测试
    teacher.load_state_dict(torch.load("ISIC2017_Best_10Percent.pth"))
    teacher.eval()
    t_dice, t_iou = [], []
    with torch.no_grad():
        for i, (img, mask, img_id) in enumerate(t_loader):
            pred = teacher(img.to(device))
            d, iou = get_metrics(mask.to(device), pred)
            t_dice.append(d); t_iou.append(iou)
    print(f"\n[10% 标签最终测试成绩] Dice: {np.mean(t_dice):.4f} | IoU: {np.mean(t_iou):.4f}")

if __name__ == "__main__":
    main()