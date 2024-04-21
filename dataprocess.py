import pandas as pd
import numpy as np

# 读取原始CSV文件
data = pd.read_csv("data_12k_10c.csv")

# 创建一个空的列表来存储处理后的数据
processed_data = []

# 定义每个标签的名称
labels = ['de_normal', 'de_7_inner', 'de_7_ball', 'de_7_outer', 'de_14_inner', 'de_14_ball', 'de_14_outer',
          'de_21_inner', 'de_21_ball', 'de_21_outer']

# 循环遍历每一列数据
for label in labels:
    # 从每一列中按要求取样本
    samples = []
    for i in range(0, len(data[label]), 512):
        sample = data[label][i:i + 1024]
        if len(sample) == 1024:
            samples.append(sample.values)

    # 将标签值添加到每个样本中
    label_values = [labels.index(label)] * len(samples)
    samples_with_label = np.column_stack((samples, label_values))

    # 将处理后的样本添加到处理后的数据列表中
    processed_data.extend(samples_with_label)

# 转换为DataFrame
processed_data_df = pd.DataFrame(processed_data)

# 添加列名
feature_columns = [f"feature{i + 1}" for i in range(1024)]
processed_data_df.columns = feature_columns + ["label"]

# 将处理后的数据保存为新的CSV文件
processed_data_df.to_csv("processed_data.csv", index=False)

print("处理后的数据已保存到 processed_data.csv 文件中。")
