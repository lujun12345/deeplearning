import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===================== 1. 基础配置 =====================
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 你的CSV文件路径（请改成你自己的路径）
CSV_PATH = r'E:\python\deep-sleep\experiments\shuju4_xsleep_exp\0517_115859\log.csv'

# 平滑窗口：每N个epoch取平均
SMOOTH_WINDOW = 5

# ===================== 2. 读取数据 =====================
df = pd.read_csv(CSV_PATH)
epochs = df["epoch"].values

loss = df["loss"].values
val_loss = df["val_loss"].values

# ===================== 3. 平滑函数 =====================
def smooth_curve(data, window=5):
    """
    移动平均平滑
    """
    kernel = np.ones(window) / window
    smoothed = np.convolve(data, kernel, mode="valid")
    # 计算平滑后对应的epoch（取窗口中间）
    smooth_epochs = np.arange(window//2, len(data) - window//2)
    return smooth_epochs, smoothed

# 执行平滑
epochs_smooth, loss_smooth = smooth_curve(loss, SMOOTH_WINDOW)
_, val_loss_smooth = smooth_curve(val_loss, SMOOTH_WINDOW)

# ===================== 4. 绘图 =====================
plt.figure(figsize=(12, 7))

# 原始曲线（浅、虚线）
plt.plot(epochs, loss, color="#a0c4ff", alpha=0.4, linestyle="--", linewidth=1, label="原始训练loss")
plt.plot(epochs, val_loss, color="#ffadad", alpha=0.4, linestyle="--", linewidth=1, label="原始验证loss")

# 平滑曲线（粗、实线、带标记）
plt.plot(epochs_smooth, loss_smooth, color="#3d85c6", linewidth=3, marker="o", markersize=4, label=f"训练loss（{SMOOTH_WINDOW}epoch平滑）")
plt.plot(epochs_smooth, val_loss_smooth, color="#ff6b6b", linewidth=3, marker="s", markersize=4, label=f"验证loss（{SMOOTH_WINDOW}epoch平滑）")

# 标题与标签
plt.title(f"训练/验证损失曲线（{SMOOTH_WINDOW}个epoch平滑）", fontsize=16, pad=15)

plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)

# 网格与图例
plt.grid(alpha=0.2)
plt.legend(fontsize=12)
plt.tight_layout()

# 保存图片
plt.savefig("loss_curve_smooth.png", dpi=300, bbox_inches="tight")
plt.close()

print("✅ 绘图完成！已保存为：loss_curve_smooth.png")
print(f"📊 原始epoch：{len(epochs)} 个")
print(f"📊 平滑后点：{len(epochs_smooth)} 个")