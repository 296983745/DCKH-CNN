import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 加载 CSV 文件
file_path = 'Kernel_size_MaxPooling_tau_results2.csv'  # 请替换为你的实际路径
df = pd.read_csv(file_path)

# 提取 "MaxPool" 和 "Kernel" 的值
df['MaxPool'] = df['Model'].str.extract(r'MaxPool=(\d+)').astype(int)
df['Kernel'] = df['Model'].str.extract(r'Kernel=(\d+)').astype(int)

# 获取不同的数据集名称
datasets = df['Dataset'].unique()

# 为每个数据集生成热力图
for dataset in datasets:
    df_dataset = df[df['Dataset'] == dataset]
    df_aggregated = df_dataset.groupby(['Kernel', 'MaxPool'], as_index=False)['Kendalls_tau'].mean()

    # 将Kendalls_tau采取4舍5入 保留4位小数
    df_aggregated['Kendalls_tau'] = df_aggregated['Kendalls_tau'].round(4)

    heatmap_data = df_aggregated.pivot(index='Kernel', columns='MaxPool', values='Kendalls_tau')

    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap='coolwarm', cbar=True, annot_kws={"size": 10}, linewidths=.5,
                linecolor='grey')
    plt.title(f'Kendalls_tau Heatmap for {dataset}')
    plt.xlabel('Max-Pooling')
    plt.ylabel('Kernel size')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.show()
