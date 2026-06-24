import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. Excel文件配置
file_path = r"C:\Users\HP\Desktop\比较_2.xlsx"   # Excel文件路径，替换为你的文件路径
sheet_name = 'Sheet1'  # 要读取的工作表名称

# 2. 数据列配置
category_col = '模型'  # 分类列（X轴，比如不同模型、不同组别）
value_cols = [
    'val_overall_accuracy',
]  # 数值列（Y轴，要对比的指标，可填1个或多个）

# 3. 颜色配置（可自定义修改，数量和数值列数量对应）
# 颜色支持：十六进制色值、RGB元组、matplotlib内置颜色名
custom_colors = [
    '#BCDDA3',
    '#34A4BE',
    '#5E3680',
    '#F4B183',
    '#C2A439',
    '#35A76C'
]

# 4. 图表样式配置
title = '各模型性能对比'  # 图表标题
x_label = '模型名称'  # X轴标签
y_label = '指标数值'  # Y轴标签
fig_size = (14, 8)  # 图表大小（宽, 高），单位英寸
bar_width = 0.7  # 单根柱子的宽度，多指标时建议0.2-0.3
show_data_label = True  # 是否在柱子上显示数值标签
data_label_precision = 3  # 数值标签保留的小数位数
y_limit = (0.6, 0.8) # Y轴范围，比如(0, 1)，None为自动适配
grid = True  # 是否显示Y轴网格线
# ===================== 【用户可修改配置区】结束 =====================

# 5. 数据读取与预处理
# 读取Excel数据
df = pd.read_excel(file_path, sheet_name=sheet_name)

# 数据清洗：去除分类列为空的行，去除数值列全为空的行
df = df.dropna(subset=[category_col])
df = df.dropna(how='all', subset=value_cols)

# 重置索引
df = df.reset_index(drop=True)

# 6. 绘制柱状图
# 创建画布
plt.figure(figsize=fig_size, dpi=100)

# 计算X轴位置
x = range(len(df[category_col]))

# 多指标分组柱状图绘制
for i, (value_col, color) in enumerate(zip(value_cols, custom_colors[:len(value_cols)])):
    # 计算每组柱子的X偏移
    bar_x = [pos + bar_width * i for pos in x]
    # 绘制柱子
    bars = plt.bar(
        bar_x,
        df[value_col],
        width=bar_width,
        label=value_col,
        color=color,
        edgecolor='white',
        linewidth=0.8
    )

    # 显示数值标签
    if show_data_label:
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + (max(df[value_cols].max()) * 0.01),  # 标签在柱子上方1%的位置
                f'{height:.{data_label_precision}f}',
                ha='center',
                va='bottom',
                fontsize=9,
                color=color,
                fontweight='medium'
            )

# 7. 图表美化与配置
# 设置标题和轴标签
plt.title(title, fontsize=16, fontweight='bold', pad=20)
plt.xlabel(x_label, fontsize=12, fontweight='medium', labelpad=15)
plt.ylabel(y_label, fontsize=12, fontweight='medium', labelpad=15)

# 设置X轴刻度
plt.xticks(
    [pos + bar_width * (len(value_cols) - 1) / 2 for pos in x],
    df[category_col],
    fontsize=10,
    rotation=45,  # X轴标签旋转45度，避免重叠
    ha='right'
)

# 设置Y轴范围
if y_limit is not None:
    plt.ylim(y_limit)

# 显示网格线
if grid:
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

# 显示图例
plt.legend(
    fontsize=10,
    loc='upper right',
    bbox_to_anchor=(1.15, 1),  # 图例放在图表右侧，避免遮挡
    borderaxespad=0
)

# 调整布局，避免标签被截断
plt.tight_layout()

# 8. 保存与显示图表
# 保存为PNG图片，分辨率300DPI
plt.savefig('消融实验对比柱状图1.png', dpi=300, bbox_inches='tight')
# 显示图表
plt.show()

# 打印数据预览，确认数据读取正确
print("===== 数据预览 =====")
print(df[[category_col] + value_cols].head(10))
print("\n===== 图表已生成并保存 =====")