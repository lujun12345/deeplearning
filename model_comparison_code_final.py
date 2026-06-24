"""
睡眠分析模型性能比较绘图代码
功能：生成多个子图展示不同模型在各数据集上的性能表现
包括：总体准确率、平均损失值、F1分数、参数与准确率关系
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def setup_chinese_font():
    """设置中文字体支持"""
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.facecolor'] = 'white'

def load_and_clean_data(file_path):
    """
    加载并清理Excel数据
    参数: file_path - Excel文件路径
    返回: 清理后的DataFrame
    """
    # 读取原始数据
    df_raw = pd.read_excel(file_path, header=None)
    
    # 处理数据，填充数据集信息
    data_rows = []
    current_dataset = None
    
    # 从第2行开始处理实际数据（跳过前两行标题）
    for i in range(2, len(df_raw)):
        row = df_raw.iloc[i]
        if pd.notna(row.iloc[0]):  # 更新当前数据集
            current_dataset = row.iloc[0]
        else:  # 填充空的数据集字段
            row.iloc[0] = current_dataset
        
        # 只保留有模型名称的有效行
        if pd.notna(row.iloc[1]):
            data_rows.append(row.tolist()[:9])  # 取前9列数据
    
    # 定义列名
    columns = ['数据集', '模型', '实验训练轮数', '参数', '最佳训练轮数', 'val_loss', 
               'val_overall_accuracy', 'val_balanced_accuracy', 'val_f1']
    
    # 创建并清理数据框
    df_final = pd.DataFrame(data_rows, columns=columns)
    
    # 转换数值列类型
    numeric_columns = ['实验训练轮数', '参数', '最佳训练轮数', 'val_loss', 
                       'val_overall_accuracy', 'val_balanced_accuracy', 'val_f1']
    
    for col in numeric_columns:
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
    
    # 去除无效行
    df_final = df_final.dropna(subset=numeric_columns)
    
    # 清理模型名称中的特殊字符
    df_final['模型'] = df_final['模型'].str.replace('‑', '-', regex=False)
    
    return df_final

def generate_model_comparison_plots(df_final, save_path='/mnt/model_comparison_analysis.png'):
    """
    生成模型比较综合图表
    参数:
        df_final - 清理后的DataFrame
        save_path - 图片保存路径
    """
    # 创建图形和子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('睡眠分析模型性能综合比较', fontsize=20, fontweight='bold', y=0.98)
    
    # 定义颜色方案和基本信息
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']  # 专业配色
    models = df_final['模型'].unique()
    datasets = df_final['数据集'].unique()
    
    # ---------------------- 子图1: 各数据集上的总体准确率比较 ----------------------
    ax1 = axes[0, 0]
    x = np.arange(len(models))
    width = 0.2  # 柱子宽度
    
    # 为每个数据集绘制柱状图
    for i, dataset in enumerate(datasets):
        data = df_final[df_final['数据集'] == dataset]['val_overall_accuracy'].values
        ax1.bar(x + i*width, data, width, label=dataset, color=colors[i], 
                alpha=0.8, edgecolor='white', linewidth=1)
    
    # 设置子图1属性
    ax1.set_xlabel('模型', fontsize=12, fontweight='bold')
    ax1.set_ylabel('总体准确率', fontsize=12, fontweight='bold')
    ax1.set_title('各数据集上的模型总体准确率比较', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(x + width * 1.5)  # 居中显示x轴标签
    ax1.set_xticklabels(models, rotation=15, ha='right')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例放在右侧
    ax1.grid(True, alpha=0.3, axis='y')  # 只显示水平网格线
    ax1.set_ylim(0, 1)  # 准确率范围0-1
    
    # 添加数值标签
    for i, dataset in enumerate(datasets):
        data = df_final[df_final['数据集'] == dataset]['val_overall_accuracy'].values
        for j, v in enumerate(data):
            ax1.text(j + i*width, v + 0.02, f'{v:.3f}', ha='center', va='bottom', 
                    fontsize=9, fontweight='bold')
    
    # ---------------------- 子图2: 模型平均损失值比较 ----------------------
    ax2 = axes[0, 1]
    
    # 计算每个模型的平均损失
    loss_data = []
    model_labels = []
    for model in models:
        model_data = df_final[df_final['模型'] == model]
        avg_loss = model_data['val_loss'].mean()
        loss_data.append(avg_loss)
        model_labels.append(f'{model}\n(平均)')
    
    # 绘制横向柱状图
    y_pos = np.arange(len(model_labels))
    bars = ax2.barh(y_pos, loss_data, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    
    # 设置子图2属性
    ax2.set_xlabel('平均损失值', fontsize=12, fontweight='bold')
    ax2.set_ylabel('模型', fontsize=12, fontweight='bold')
    ax2.set_title('各模型平均损失值比较\n(值越小越好)', fontsize=14, fontweight='bold', pad=20)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(model_labels)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 添加数值标签
    for i, (bar, value) in enumerate(zip(bars, loss_data)):
        ax2.text(value + 0.02, bar.get_y() + bar.get_height()/2, f'{value:.3f}', 
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    # ---------------------- 子图3: F1分数比较（折线图） ----------------------
    ax3 = axes[1, 0]
    
    # 为每个模型绘制折线
    for i, model in enumerate(models):
        model_data = df_final[df_final['模型'] == model].sort_values('数据集')
        f1_scores = model_data['val_f1'].values
        
        ax3.plot(datasets, f1_scores, marker='o', linewidth=3, markersize=8, 
                label=model, color=colors[i], alpha=0.9, 
                markerfacecolor='white', markeredgewidth=2, markeredgecolor=colors[i])
    
    # 设置子图3属性
    ax3.set_xlabel('数据集', fontsize=12, fontweight='bold')
    ax3.set_ylabel('F1分数', fontsize=12, fontweight='bold')
    ax3.set_title('各模型在不同数据集上的F1分数比较', fontsize=14, fontweight='bold', pad=20)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=15, ha='right')  # 旋转x轴标签
    
    # ---------------------- 子图4: 参数数量与平衡准确率关系 ----------------------
    ax4 = axes[1, 1]
    
    # 绘制散点图
    for i, model in enumerate(models):
        model_data = df_final[df_final['模型'] == model]
        params = model_data['参数'].values
        balanced_acc = model_data['val_balanced_accuracy'].values
        
        # 散点图
        ax4.scatter(params, balanced_acc, s=120, alpha=0.8, color=colors[i], 
                   label=model, edgecolors='white', linewidth=2)
        
        # 添加数据集标签
        for j, (p, acc, dataset) in enumerate(zip(params, balanced_acc, model_data['数据集'])):
            ax4.annotate(dataset, (p, acc), xytext=(5, 5), textcoords='offset points', 
                        fontsize=9, fontweight='bold', alpha=0.8)
    
    # 设置子图4属性
    ax4.set_xlabel('参数数量', fontsize=12, fontweight='bold')
    ax4.set_ylabel('平衡准确率', fontsize=12, fontweight='bold')
    ax4.set_title('模型参数数量与平衡准确率关系', fontsize=14, fontweight='bold', pad=20)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # 为总标题留出空间
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"模型比较图表已保存至: {save_path}")

def generate_performance_summary(df_final):
    """
    生成模型性能汇总表
    参数: df_final - 清理后的DataFrame
    返回: 汇总表DataFrame
    """
    # 按模型分组计算统计指标
    summary_table = df_final.groupby('模型').agg({
        'val_overall_accuracy': ['mean', 'std'],
        'val_loss': ['mean', 'std'],
        'val_f1': ['mean', 'std'],
        '参数': ['mean', 'min', 'max']
    }).round(3)
    
    # 重命名列名
    summary_table.columns = ['平均准确率', '准确率标准差', '平均损失', '损失标准差', 
                            '平均F1', 'F1标准差', '平均参数', '最小参数', '最大参数']
    
    return summary_table

def save_performance_summary(summary_table, save_path='/mnt/model_performance_summary.xlsx'):
    """
    保存性能汇总表到Excel文件
    参数:
        summary_table - 性能汇总表DataFrame
        save_path - 保存路径
    """
    # 使用ExcelWriter保存，确保格式正确
    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        summary_table.to_excel(writer, sheet_name='模型性能汇总', index=True)
    
    print(f"性能汇总表已保存至: {save_path}")

def main():
    """主函数：执行完整的数据分析流程"""
    # 1. 初始化设置
    setup_chinese_font()
    
    # 2. 加载和清理数据
    file_path = '/mnt/比较_2.xlsx'  # 请根据实际路径修改
    print("正在加载和清理数据...")
    df_final = load_and_clean_data(file_path)
    
    # 3. 生成可视化图表
    print("正在生成模型比较图表...")
    generate_model_comparison_plots(df_final)
    
    # 4. 生成性能汇总表
    print("\n生成模型性能汇总表:")
    summary_table = generate_performance_summary(df_final)
    print(summary_table)
    
    # 5. 保存汇总表到Excel
    save_performance_summary(summary_table)
    
    # 6. 输出关键发现
    print("\n=== 关键发现 ===")
    best_accuracy_model = summary_table['平均准确率'].idxmax()
    best_accuracy_value = summary_table['平均准确率'].max()
    print(f"1. 平均准确率最高的模型: {best_accuracy_model} ({best_accuracy_value:.3f})")
    
    best_loss_model = summary_table['平均损失'].idxmin()
    best_loss_value = summary_table['平均损失'].min()
    print(f"2. 平均损失最低的模型: {best_loss_model} ({best_loss_value:.3f})")
    
    best_f1_model = summary_table['平均F1'].idxmax()
    best_f1_value = summary_table['平均F1'].max()
    print(f"3. 平均F1分数最高的模型: {best_f1_model} ({best_f1_value:.3f})")
    
    least_params_model = summary_table['平均参数'].idxmin()
    least_params_value = summary_table['平均参数'].min()
    print(f"4. 平均参数最少的模型: {least_params_model} ({least_params_value:.0f}个参数)")

if __name__ == "__main__":
    main()
