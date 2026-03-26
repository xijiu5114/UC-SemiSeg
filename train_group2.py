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
# 固定随机种子
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
# 1. 数据集定义
# ==============================
class PH2Dataset(Dataset):
    def __init__(self, images_dir, masks_dir, augment=False):
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.augment = augment
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        img = self.img_transform(img)
        mask = self.mask_transform(mask)
        mask = torch.where(mask > 0, 1.0, 0.0)
        return img, mask

# ==============================
# 2. Attention UNet 结构
# ==============================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
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
        g1 = self.W_g(g); x1 = self.W_x(x)
        psi = self.psi(self.relu(g1 + x1))
        return x * psi

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
        c1 = self.conv1(x); p1 = self.maxpool(c1)
        c2 = self.conv2(p1); p2 = self.maxpool(c2)
        c3 = self.conv3(p2); p3 = self.maxpool(c3)
        c4 = self.conv4(p3); c4 = self.dropout(c4)
        up3 = self.up3(c4); att3 = self.att3(up3, c3); x3 = self.upconv3(torch.cat([up3, att3], dim=1))
        up2 = self.up2(x3); att2 = self.att2(up2, c2); x2 = self.upconv2(torch.cat([up2, att2], dim=1))
        up1 = self.up1(x2); att1 = self.att1(up1, c1); x1 = self.upconv1(torch.cat([up1, att1], dim=1))
        return torch.sigmoid(self.out(x1))

# ==============================
# 3. 损失与评价指标
# ==============================
class DiceBCELoss(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, pred, target, smooth=1):
        pred = pred.view(-1); target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        bce = F.binary_cross_entropy(pred, target)
        return bce + (1 - dice)

def get_metrics(y_true, y_pred):
    y_pred = (y_pred > 0.5).float()
    tp = (y_true * y_pred).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    return dice.item(), iou.item()

def mc_dropout_inference(model, img, T=10):
    model.train() # 开启 Dropout
    preds = torch.stack([model(img) for _ in range(T)])
    return preds.mean(0), preds.std(0)

# ==============================
# 4. Group 2 训练主流程
# ==============================
def run_group2_complete():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res_dir = "group2_results"
    os.makedirs(res_dir, exist_ok=True)

    # 1. 加载并抽样 (20张图)
    full_train = PH2Dataset("PH2_dataset_final/train/images", "PH2_dataset_final/train/masks", augment=True)
    indices = np.random.RandomState(seed=42).choice(len(full_train), 20, replace=False)
    small_train_dataset = Subset(full_train, indices)
    
    val_ds = PH2Dataset("PH2_dataset_final/val/images", "PH2_dataset_final/val/masks")
    test_ds = PH2Dataset("PH2_dataset_final/test/images", "PH2_dataset_final/test/masks")

    train_loader = DataLoader(small_train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)
    test_loader = DataLoader(test_ds, batch_size=1)

    model = AttentionUNet().to(device)
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_losses, val_dices = [], []
    best_dice = 0

    # 2. 训练
    print(f">>> 开始 Group 2 训练 (仅使用 {len(small_train_dataset)} 张标注图)...")
    for epoch in range(50):
        model.train()
        running_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward(); optimizer.step()
            running_loss += loss.item()
        
        # 验证
        model.eval()
        v_dice = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                d, _ = get_metrics(masks, model(imgs))
                v_dice += d
        
        avg_v_dice = v_dice / len(val_loader)
        train_losses.append(running_loss / len(train_loader))
        val_dices.append(avg_v_dice)
        
        if avg_v_dice > best_dice:
            best_dice = avg_v_dice
            torch.save(model.state_dict(), "best_model_group2.pth")
            print(f"Epoch {epoch+1}: Best Val Dice Updated -> {best_dice:.4f}")

    # 3. 绘制 Loss/Metric 曲线
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.title('Group 2 Training Loss')
    plt.subplot(1, 2, 2)
    plt.plot(val_dices, label='Val Dice', color='green')
    plt.title('Group 2 Validation Dice')
    plt.tight_layout()
    plt.savefig(f"{res_dir}/loss_curve_group2.png"); plt.close()

    # 4. 测试与可视化 (Overlay + Uncertainty)
    print("\n>>> 正在生成测试集可视化结果...")
    model.load_state_dict(torch.load("best_model_group2.pth"))
    
    test_dices, test_ious = [], []

    for idx, (img, mask) in enumerate(test_loader):
        img_gpu = img.to(device)
        # 先用 MC Dropout 生成不确定性
        mu, sigma = mc_dropout_inference(model, img_gpu, T=10)
        # 清晰指标由 eval 模式输出
        with torch.no_grad():
            model.eval()
            eval_pred = model(img_gpu)
        d, i = get_metrics(mask.to(device), eval_pred)
        test_dices.append(d); test_ious.append(i)

        img_np = img[0].permute(1, 2, 0).numpy()
        img_bgr = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # Overlay 使用 eval_pred
        mask_np = (eval_pred[0,0].cpu().detach().numpy() > 0.5).astype(np.uint8) * 255
        overlay = img_bgr.copy()
        overlay[mask_np > 127] = [255, 0, 0]
        img_overlay = cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0)
        
        # Uncertainty
        sig_np = sigma[0,0].cpu().detach().numpy()
        sig_norm = (sig_np - sig_np.min()) / (sig_np.max() - sig_np.min() + 1e-6)
        heatmap = cv2.applyColorMap((sig_norm * 255).astype(np.uint8), cv2.COLORMAP_HOT)

        cv2.imwrite(f"{res_dir}/{idx}_image.png", img_bgr)
        cv2.imwrite(f"{res_dir}/{idx}_overlay.png", img_overlay)
        cv2.imwrite(f"{res_dir}/{idx}_uncertainty.png", heatmap)

    print("-" * 30)
    print(f"Group 2 最终成绩 (eval 模式) -> Dice: {np.mean(test_dices):.4f}, IoU: {np.mean(test_ious):.4f}")

if __name__ == "__main__":
    run_group2_complete()