import matplotlib.pyplot as plt
import numpy as np

# --- 学术级绘图配置 ---
plt.rcParams['font.family'] = 'Consolas'  # 使用专业等宽字体
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# --- 数据准备 (来自图4表格) ---
methods = ['o/w MT', 'o/w MCD', 'o/w SRF', 'Ours']
dice_scores = [0.879, 0.879, 0.870, 0.895]  # Dice 值
iou_scores = [0.800, 0.795, 0.781, 0.821]   # IoU 值

x_labels = ['(a) w/o MT', '(b) w/o MCD', '(c) w/o SRF', '(d) Ours (UC-SemiSeg)']

# 计算增益 (Ours 相对于最高的 w/o 组)
dice_gain = (dice_scores[3] - max(dice_scores[:3])) * 100
iou_gain = (iou_scores[3] - max(iou_scores[:3])) * 100

# --- 绘图逻辑 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=100)  # 创建 1 行 2 列的图

bar_width = 0.6
x = np.arange(len(methods))

# 配色：基准组用淡蓝/淡橙，Ours 用深蓝/深橙
colors_dice = ['#b3cde3', '#b3cde3', '#b3cde3', '#2b6fbe']  # Ours 使用深蓝色，其他淡色
colors_iou = ['#fddaec', '#fddaec', '#fddaec', '#ff7f0e']   # Ours 使用深橙色，其他淡色

# --- 子图1: Dice 结果 ---
ax_dice = axes[0]
rects_dice = ax_dice.bar(x, dice_scores, bar_width, color=colors_dice, edgecolor='#2f4f4f', linewidth=1.2)

# 设置 Dice 轴属性
ax_dice.set_ylabel('Dice Coefficient ↑', fontsize=13)
ax_dice.set_title('Ablation Study on Dice Score', fontsize=14, fontweight='bold')
ax_dice.set_xticks(x)
ax_dice.set_xticklabels(x_labels, rotation=15, ha='right')  # 旋转标签避免重叠
ax_dice.set_ylim(0.70, 0.95)  # 设定 Y 轴范围以突出差异
ax_dice.grid(axis='y', linestyle='--', alpha=0.5)

# --- 子图2: IoU 结果 ---
ax_iou = axes[1]
rects_iou = ax_iou.bar(x, iou_scores, bar_width, color=colors_iou, edgecolor='#2f4f4f', linewidth=1.2)

# 设置 IoU 轴属性
ax_iou.set_ylabel('IoU Coefficient ↑', fontsize=13)
ax_iou.set_title('Ablation Study on IoU Score', fontsize=14, fontweight='bold')
ax_iou.set_xticks(x)
ax_iou.set_xticklabels(x_labels, rotation=15, ha='right')
ax_iou.set_ylim(0.70, 0.90)  # IoU 值通常更低，范围略有不同
ax_iou.grid(axis='y', linestyle='--', alpha=0.5)


# --- 在柱子上方自动添加数据标注和增益 ---
def autolabel_and_gain(rects, ax, gain, target_index=3):
    """自动给柱子添加数据和增益标注"""
    for i, rect in enumerate(rects):
        height = rect.get_height()
        # 标注具体数值
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 垂直偏移
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 专门针对 "Ours" 标注增益
        if i == target_index:
            ax.annotate(f'(+{gain:.1f}%)',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 20),  # 更高偏移
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2ca02c') # 绿色增益


# 执行数据标注
autolabel_and_gain(rects_dice, ax_dice, dice_gain)
autolabel_and_gain(rects_iou, ax_iou, iou_gain)

# --- 优化布局并保存 ---
plt.tight_layout()  # 自动调整布局，防止标签被截断
plt.savefig('ablation_barchart.png', dpi=300, bbox_inches='tight')  # 保存为高分辨率图片
plt.show()  # 显示图片