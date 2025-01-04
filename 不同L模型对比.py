import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
import colorsys

# 加载本地字体文件
font_path = 'C:/Windows/Fonts/simhei.ttf'
font_prop = fm.FontProperties(fname=font_path)

# 读取CSV文件
df = pd.read_csv('tau_results1.csv')

# 设置调色板
palette = sns.color_palette("Set2", len(df['L'].unique()))

# 打印调色板和L值信息以进行调试
print("Palette length:", len(palette))
print("L values:", df['L'].unique())

# 设置图表样式
plt.style.use('seaborn-v0_8-whitegrid')

# 设置中文字体
plt.rcParams['font.sans-serif'] = [font_path]
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

def darken_color(color, amount=0.3):
    """
    Darkens the given color by the specified amount.
    """
    try:
        c = colorsys.rgb_to_hls(*colorsys.hex_to_rgb(color))
        return colorsys.hls_to_rgb(c[0], max(0, c[1] * (1 - amount)), c[2])
    except:
        r, g, b = colorsys.rgb_to_hls(*color[:3])
        return tuple([max(0, x * (1 - amount)) for x in colorsys.hls_to_rgb(r, g, b)])

# 创建绘图对象
fig, ax = plt.subplots(figsize=(12, 8))

# 获取唯一的Dataset和L值
datasets = df['Dataset'].unique()
l_values = df['L'].unique()

# 定义位置参数
width = 0.15  # 每个箱线图的宽度
gap = 0.4  # 每个组之间的间隙
positions = []
current_pos = 0

# 计算每个箱线图的位置
for dataset in datasets:
    dataset_positions = []
    for i, l in enumerate(l_values):
        dataset_positions.append(current_pos + i * (width + 0.05))  # 增加模型之间的间隙
    positions.append(dataset_positions)
    current_pos += len(l_values) * (width + 0.05) + gap  # 增加每组之间的间隙

# 绘制箱线图
for dataset, dataset_positions in zip(datasets, positions):
    subset = df[df['Dataset'] == dataset]
    for pos, l in zip(dataset_positions, l_values):
        index = l_values.tolist().index(l)
        if index >= len(palette):
            print(f"Index {index} out of range for palette with length {len(palette)}")
            continue
        color = palette[index]
        darker_color = darken_color(color, amount=0.3)
        bp = ax.boxplot(subset[subset['L'] == l]['Kendalls_tau'], positions=[pos], widths=width, patch_artist=True,
                        boxprops=dict(facecolor=color, color=darker_color),
                        medianprops=dict(color=darker_color, linewidth=2),
                        whiskerprops=dict(color=darker_color),
                        capprops=dict(color=darker_color),
                        flierprops=dict(markeredgecolor=color))
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_edgecolor(darker_color)
        for median in bp['medians']:
            median.set_color(darker_color)

# 设置x轴标签
ax.set_xticks([np.mean(pos) for pos in positions])
ax.set_xticklabels(datasets, fontsize=11, rotation=0, fontproperties=font_prop)

# 调整标签和标题
ax.set_xlabel('datasets', fontsize=13, weight='bold', fontproperties=font_prop)
ax.set_ylabel("Kendall's τ ", fontsize=13, weight='bold', fontproperties=font_prop)
# ax.set_title("邻域网络大小对LCNN模型影响的性能分析", fontsize=14, weight='bold', fontproperties=font_prop)

# 自定义图例
handles = [plt.Line2D([0], [0], color=palette[i], lw=4) for i in range(len(l_values))]
labels = [f'{l}' for l in l_values]
ax.legend(handles, labels, title_fontsize='11', bbox_to_anchor=(0.5, 1.1), loc='upper center', prop={'family': font_prop.get_name(), 'size': 14}, ncol=4)

# 减少网格线
ax.grid(True, linestyle='--', linewidth=0.6, which='major', axis='y')
ax.grid(False, which='major', axis='x')

# 显示图表
plt.tight_layout()
plt.show()
