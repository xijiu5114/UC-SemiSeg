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
import cv2

# ==========================================
# 1. 实验总控 (在此切换消融实验)
# ==========================================
config = {
    # 实验模式: 
    # 'no_MT'   -> 关掉 MT (纯监督 Baseline)
    # 'no_MCD'  -> 关掉 MCD (保持 MT, 但采样次数 T=1)
    # 'no_SRF'  -> 关掉 SRF (保持 MT+MCD, 但权重固定为 1)
    # 'FULL'    -> 完整模型 (全开)
    "mode": "no_SRF", 
    
    "seed": 42,
    "epochs": 100,
    "batch_size": 4,
    "lr": 1e-4,          # 如果 no_MT 跑不动，尝试 5e-5
    "T_samples": 8,      # MCD 采样次数
    "beta": 0.01,        # SRF 敏感度
    "rampup_epoch": 40,  # 一致性权重预热
    "ema_alpha": 0.999,  # 教师网络更新系数
    "consist_max": 0.5   # 一致性损失最大权重
}

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(config["seed"])

# ==========================================
# 2. 数据集类 (PH2 路径适配)
# ==========================================
class PH2Dataset(Dataset):
    def __init__(self, images_dir, masks_dir, augment=False):
        self.images = sorted([f for f in os.listdir(images_dir) if f.endswith('.bmp')])
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.augment = augment
        self.resize = transforms.Resize((256, 256))
        self.to_tensor = transforms.ToTensor()

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_id = os.path.splitext(img_name)[0]
        
        # 加载图像
        img = Image.open(os.path.join(self.images_dir, img_name)).convert("RGB")
        
        # 加载 Mask (处理 PH2 常见的命名: 可能是 .png 或 .bmp)
        mask_path = os.path.join(self.masks_dir, img_id + ".png")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.masks_dir, img_id + ".bmp")
        mask = Image.open(mask_path).convert("L")

        # 调整大小
        img = self.resize(img)
        mask = self.resize(mask)

        # 简单增强
        if self.augment and random.random() > 0.5:
            img = transforms.functional.hflip(img)
            mask = transforms.functional.hflip(mask)

        img = self.to_tensor(img)
        mask = torch.where(self.to_tensor(mask) > 0.5, 1.0, 0.0)
        
        return img, mask, img_id

# ==========================================
# 3. 网络架构 (Attention U-Net)
# ==========================================
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
    def forward(self, g, x):
        return x * self.psi(F.relu(self.W_g(g) + self.W_x(x), inplace=True))

class AttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv1 = DoubleConv(3, 64); self.conv2 = DoubleConv(64, 128)
        self.conv3 = DoubleConv(128, 256); self.conv4 = DoubleConv(256, 512)
        self.dropout = nn.Dropout2d(0.5)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2); self.att3 = AttentionBlock(256, 256, 128); self.upconv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2); self.att2 = AttentionBlock(128, 128, 64); self.upconv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2); self.att1 = AttentionBlock(64, 64, 32); self.upconv1 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        c1 = self.conv1(x); p1 = self.maxpool(c1); c2 = self.conv2(p1); p2 = self.maxpool(c2)
        c3 = self.conv3(p2); p3 = self.maxpool(c3); c4 = self.dropout(self.conv4(p3))
        x3 = self.upconv3(torch.cat([self.up3(c4), self.att3(self.up3(c4), c3)], dim=1))
        x2 = self.upconv2(torch.cat([self.up2(x3), self.att2(self.up2(x3), c2)], dim=1))
        x1 = self.upconv1(torch.cat([self.up1(x2), self.att1(self.up1(x2), c1)], dim=1))
        return torch.sigmoid(self.out(x1))

# ==========================================
# 4. 训练与消融逻辑
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 路径配置 (请确保文件夹存在)
    train_img = "dataset/PH2_dataset_final/train/images"
    train_mask = "dataset/PH2_dataset_final/train/masks"
    
    full_dataset = PH2Dataset(train_img, train_mask, augment=True)
    
    # 划分 20 张有标签，其余无标签
    indices = np.random.RandomState(42).choice(len(full_dataset), 20, replace=False)
    unlabeled_idx = list(set(range(len(full_dataset))) - set(indices))
    
    l_loader = DataLoader(Subset(full_dataset, indices), batch_size=config["batch_size"], shuffle=True)
    u_loader = DataLoader(Subset(full_dataset, unlabeled_idx), batch_size=config["batch_size"], shuffle=True)
    # 验证集使用无标签中的前 20 张
    v_loader = DataLoader(Subset(full_dataset, unlabeled_idx[:20]), batch_size=1)

    student = AttentionUNet().to(device)
    teacher = AttentionUNet().to(device)
    teacher.load_state_dict(student.state_dict())
    
    optimizer = optim.Adam(student.parameters(), lr=config["lr"], weight_decay=1e-5)
    
    print(f"--- 正在运行模式: {config['mode']} ---")

    best_dice = 0.0
    patience = 20
    patience_counter = 0
    global_step = 0

    for epoch in range(config["epochs"]):
        student.train()
        u_iter = iter(u_loader)
        
        # 一致性损失权重预热
        w_consist = config["consist_max"] * min(1.0, epoch / config["rampup_epoch"]) if config["mode"] != "no_MT" else 0

        for img_l, mask_l, _ in l_loader:
            img_l, mask_l = img_l.to(device), mask_l.to(device)
            optimizer.zero_grad()

            # 1. 监督损失 (BCE + Dice)
            pred_l = student(img_l)
            bce = F.binary_cross_entropy(pred_l, mask_l)
            dice_loss = 1 - (2.*(pred_l*mask_l).sum()+1e-7)/(pred_l.sum()+mask_l.sum()+1e-7)
            loss_sup = bce + dice_loss

            # 2. 一致性损失 (受消融开关控制)
            loss_consist = torch.tensor(0.0).to(device)
            if config["mode"] != "no_MT":
                try: img_u, _, _ = next(u_iter)
                except: u_iter = iter(u_loader); img_u, _, _ = next(u_iter)
                img_u = img_u.to(device)
                
                teacher.train() # 开启 Dropout
                t_count = 1 if config["mode"] == "no_MCD" else config["T_samples"]
                
                with torch.no_grad():
                    t_preds = torch.stack([teacher(img_u) for _ in range(t_count)])
                    mu = t_preds.mean(0)
                    # 权重计算
                    weight = torch.ones_like(mu)
                    if config["mode"] == "FULL" and t_count > 1:
                        sigma = t_preds.std(0)
                        weight = torch.exp(-torch.pow(sigma, 2) / config["beta"])
                
                pred_u = student(img_u)
                pseudo_label = (mu > 0.5).float()
                loss_consist = (weight * F.mse_loss(pred_u, pseudo_label, reduction='none')).mean()

            # 总损失反传
            (loss_sup + w_consist * loss_consist).backward()
            optimizer.step()

            # EMA 更新教师网络
            if config["mode"] != "no_MT":
                alpha = min(1 - 1 / (global_step + 1), config["ema_alpha"])
                for tp, sp in zip(teacher.parameters(), student.parameters()):
                    tp.data.mul_(alpha).add_(sp.data, alpha=1-alpha)
            global_step += 1

        # --- 验证与可视化诊断 ---
        student.eval()
        d_scores, i_scores = [], []
        with torch.no_grad():
            for i, (v_img, v_mask, _) in enumerate(v_loader):
                v_img, v_mask = v_img.to(device), v_mask.to(device)
                pred = student(v_img)
                
                # 每一轮存一张图看一眼，防止锁死 0.47
                if i == 0:
                    p_np = (pred[0,0].cpu().numpy()*255).astype(np.uint8)
                    m_np = (v_mask[0,0].cpu().numpy()*255).astype(np.uint8)
                    debug_img = np.hstack([p_np, (p_np > 127).astype(np.uint8)*255, m_np])
                    cv2.imwrite(f"debug_{epoch}.png", debug_img)

                # 指标计算
                p_bin = (pred > 0.5).float()
                inter = (p_bin * v_mask).sum()
                dice = (2.*inter + 1e-7) / (p_bin.sum() + v_mask.sum() + 1e-7)
                iou = (inter + 1e-7) / ((p_bin + v_mask).sum() - inter + 1e-7)
                d_scores.append(dice.item()); i_scores.append(iou.item())

        avg_dice = np.mean(d_scores)
        if avg_dice > best_dice:
            best_dice = avg_dice
            patience_counter = 0
        else:
            patience_counter += 1
        
        print(f"Epoch {epoch} | Loss: {loss_sup.item():.4f} | Dice: {avg_dice:.4f} | IoU: {np.mean(i_scores):.4f} | PredMax: {pred.max().item():.3f} | Patience: {patience_counter}/{patience}")
        
        # 早停检查
        if patience_counter >= patience:
            print(f"早停触发! 连续 {patience} 个 epoch 没有改进，停止训练。")
            print(f"最佳 Dice: {best_dice:.4f}")
            break

if __name__ == "__main__":
    main()