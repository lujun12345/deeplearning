
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os

# ====================== 路径配置（自动匹配你的predict输出）======================
PREDICTION_DIR = os.environ.get(
    "PREDICTION_DIR",
    r'E:\python\deep-sleep\experiments\runs\predictions-checkpoint-epoch86'
) # 改成你自己的预测文件夹，或设置 PREDICTION_DIR 环境变量
CONFUSION_PATH = os.path.join(PREDICTION_DIR, "confusionmatrix.pkl")
CSV_PATH = os.path.join(PREDICTION_DIR, "predictions.csv")

# 睡眠阶段名称（固定5类）
STAGE_NAMES = ['W', 'N1', 'N2', 'N3', 'REM']
WINDOWS = [1, 3, 5, 10, 15, 30]
colors = LinearSegmentedColormap.from_list('sleep_cmap', ['#FFFFFF', '#5A9BD3'])

plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.unicode_minus'] = False

def get_window_epochs(df):
    if 'Epoch stride' in df.columns:
        return pd.to_numeric(df['Epoch stride'])
    return df['Window'].str.extract(r'(\d+)').astype(int)[0]

# ==============================================================================
# 图1：多窗口性能曲线（OA / F1 / Kappa）
# ==============================================================================
def plot_window_performance():
    df = pd.read_csv(CSV_PATH)
    df['WindowEpochs'] = get_window_epochs(df)
    mean_df = df.groupby('WindowEpochs').mean(numeric_only=True).sort_index()

    plt.figure(figsize=(9, 5))
    plt.plot(mean_df.index, mean_df['Overall accuracy'], marker='o', linewidth=2.5, label='Accuracy')
    plt.plot(mean_df.index, mean_df['F1'], marker='s', linewidth=2.5, label='Macro F1')
    plt.plot(mean_df.index, mean_df['Kappa'], marker='^', linewidth=2.5, label='Kappa')

    plt.title('Performance across different epoch windows', fontsize=14)
    plt.xlabel('Window size (epochs)', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(WINDOWS)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PREDICTION_DIR, "window_performance.png"), bbox_inches='tight')
    plt.close()
    print("Figure 1 saved: window_performance.png")

# ==============================================================================
# 图2：各睡眠阶段 F1 分数柱状图
# ==============================================================================
def plot_stage_f1():
    df = pd.read_csv(CSV_PATH)
    df['WindowEpochs'] = get_window_epochs(df)
    df_30epoch = df[df['WindowEpochs'] == 30]

    f1_scores = [df_30epoch[f'F1 - {s}'].mean() for s in STAGE_NAMES]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(STAGE_NAMES, f1_scores, color='#5A9BD3', edgecolor='black', linewidth=1.2)
    plt.bar_label(bars, fmt='%.2f', fontsize=11)
    plt.ylim(0, 1)
    plt.title('F1 Score for each sleep stage', fontsize=14)
    plt.ylabel('F1 Score', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PREDICTION_DIR, "stage_f1_bar.png"), bbox_inches='tight')
    plt.close()
    print("Figure 2 saved: stage_f1_bar.png")

# ==============================================================================
# 图3：混淆矩阵热力图（30 epochs）
# ==============================================================================
def plot_confusion_matrix():
    confmat = pickle.load(open(CONFUSION_PATH, 'rb'))

    total_dict = confmat['total']
    best_window = 30 if 30 in total_dict else max(total_dict.keys())
    C = total_dict[best_window]

    C = C / C.sum(axis=1, keepdims=True)

    plt.figure(figsize=(7, 5.5))
    sns.heatmap(C, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=STAGE_NAMES, yticklabels=STAGE_NAMES)
    plt.title(f'Confusion Matrix ({best_window}-epoch window)', fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(PREDICTION_DIR, "confusion_matrix_30epoch.png"), bbox_inches='tight')
    plt.close()
    print("Figure 3 saved: confusion_matrix_30epoch.png")

# ==============================================================================
# 图4：单受试者睡眠时序图（真实 vs 预测）
# ==============================================================================
def plot_sleep_sequence():
    example_subject = [f for f in os.listdir(PREDICTION_DIR) if f.endswith('.pkl')][1]
    with open(os.path.join(PREDICTION_DIR, example_subject), 'rb') as f:
        data = pickle.load(f)

    t = np.concatenate(data['targets'])
    p = np.concatenate(data['predictions'], axis=1).argmax(axis=0)

    L = 600  # 只画前10分钟
    t = t[:L]
    p = p[:L]

    plt.figure(figsize=(14, 4))
    plt.plot(t, label='Ground Truth', linewidth=2.5, color='#FF5733')
    plt.plot(p, label='Prediction', linewidth=1.8, alpha=0.8, color='#3366FF')
    plt.yticks([0,1,2,3,4], STAGE_NAMES)
    plt.title(f'Sleep Stage Sequence - {example_subject[:-4]}', fontsize=14)
    plt.xlabel('Epoch (30 s each)')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PREDICTION_DIR, "sleep_sequence.png"), bbox_inches='tight')
    plt.close()
    print("Figure 4 saved: sleep_sequence.png")

# ==============================================================================
# 图5：真实标签分布 vs 预测标签分布
# ==============================================================================
def plot_label_distribution():
    df = pd.read_csv(CSV_PATH)
    df['WindowEpochs'] = get_window_epochs(df)
    df_30epoch = df[df['WindowEpochs'] == 30]

    support = [df_30epoch[f'Support - {s}'].mean() for s in STAGE_NAMES]
    pred_counts = support / sum(support)

    plt.figure(figsize=(8, 5))
    plt.bar(STAGE_NAMES, pred_counts, color='#E4989A', edgecolor='black', linewidth=1.2)
    plt.title('Sleep Stage Distribution', fontsize=14)
    plt.ylabel('Proportion', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PREDICTION_DIR, "label_distribution.png"), bbox_inches='tight')
    plt.close()
    print("Figure 5 saved: label_distribution.png")


# ==============================================================================
# 主函数：一次性画出所有图
# ==============================================================================
if __name__ == "__main__":
    import os
    print("Generating figures...\n")

    # plot_window_performance()
    # plot_stage_f1()
    # plot_confusion_matrix()
    # plot_sleep_sequence()
    plot_label_distribution()
    print("\nAll 5 figures generated.")
    # Figure 1: Performance of the model on different epoch windows.
    # Figure 2: Macro F1-score for each sleep stage with 30-epoch window.
    # Figure 3: Normalized confusion matrix of the proposed model.
    # Figure 4: An example of overnight sleep stage prediction (ground truth vs prediction).
    # Figure 5: Distribution of sleep stages in the test set.
    # Figure 6: Overall architecture of the proposed sleep staging model.
