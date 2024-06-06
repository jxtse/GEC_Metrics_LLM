import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import pingouin as pg

# 读取数据
with open('fine-tuned-llama2-13b.json', 'r') as file:
    data = [json.loads(line) for line in file]

# 提取label和predict数据
labels = []
predictions = []

for item in data:
    # 去除多余空格并转换为字典
    label = {k.strip(): int(v.strip()) for k, v in (pair.split(':') for pair in item['label'].split(','))}
    
    # label = {}
    # for pair in item['label'].split(','):
    #     k, v = pair.split(':')
    #     k = k.strip()
    #     v = v.strip()
    #     if v:  # 检查v是否为空
    #         label[k] = int(v)
    predict = {k.strip(): int(v.strip()) for k, v in (pair.split(':') for pair in item['predict'].split(','))}
    labels.append(label)
    predictions.append(predict)

# 转换为DataFrame
df_labels = pd.DataFrame(labels)
df_predictions = pd.DataFrame(predictions)

# 计算一致性指标
metrics = {}
for column in df_labels.columns:
    mae = mean_absolute_error(df_labels[column], df_predictions[column])
    mse = mean_squared_error(df_labels[column], df_predictions[column])
    pearson_corr, _ = pearsonr(df_labels[column], df_predictions[column])
    metrics[column] = {'MAE': mae, 'MSE': mse, 'Pearson': pearson_corr}

# 转换为DataFrame便于展示
df_metrics = pd.DataFrame(metrics).T
print("一致性指标：")
print(df_metrics)

# 设置图形风格
sns.set(style="whitegrid")

# 散点图比较label和predict
# for column in df_labels.columns:
#     plt.figure(figsize=(8, 6))
#     sns.scatterplot(x=df_labels[column], y=df_predictions[column])
#     plt.xlabel('Label')
#     plt.ylabel('Predict')
#     plt.title(f'Scatter Plot for {column}')
#     plt.plot([0, 10], [0, 10], 'r--')  # 添加y=x的参考线
#     plt.show()

# 创建一张图表，绘制所有评分的散点图
plt.figure(figsize=(8, 10),)

# 设置颜色和标记
markers = {'semantic coherence': 'o', 'edit level': 's', 'fluency': 'D', 'overall': '^'}
colors = {'semantic coherence': 'b', 'edit level': 'g', 'fluency': 'r', 'overall': 'm'}

# 绘制每个评分的散点图
for column in df_labels.columns:
    sns.scatterplot(x=df_labels[column], y=df_predictions[column], label=column, 
                    marker=markers[column], color=colors[column], s=100)

# 添加y=x的参考线
plt.plot([0, 10], [0, 10], 'r--', linewidth=2)

# 设置图表标题和标签
plt.xlabel('Label', fontsize=14)
plt.ylabel('Predict', fontsize=14)
plt.title('Scatter Plot for All Scores', fontsize=16)
plt.legend(title='Scores', fontsize=12, title_fontsize=14)

# 添加网格线
plt.grid(True, linestyle='--', linewidth=0.5)

plt.show()