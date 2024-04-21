import json

import torch
from joblib import dump, load
import torch.utils.data as Data
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F
from sklearn import preprocessing

torch.manual_seed(100)  # 设置随机种子，以使实验结果具有可重复性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 有GPU先用GPU训练
batch_size = 32
input_dim = 32   # 输入维度为一维信号序列堆叠为 32 * 32
hidden_layer_sizes = [256, 128, 64]
output_dim = 10


class LSTMclassifier(nn.Module):
    def __init__(self, batch_size, input_dim, hidden_layer_sizes, output_dim, dropout_rate=0.5):
        """
        LSTM 分类任务  params:
        batch_size       : 批次量大小
        input_dim        : 输入数据的维度
        hidden_layer_size:隐层的数目和维度
        output_dim       : 输出的维度
        dropout_rate     : 随机丢弃神经元的概率
        """
        super().__init__()
        # 批次量大小
        self.batch_size = batch_size
        # lstm层数
        self.num_layers = len(hidden_layer_sizes)
        self.lstm_layers = nn.ModuleList()  # 用于保存LSTM层的列表

        # 定义第一层LSTM
        self.lstm_layers.append(nn.LSTM(input_dim, hidden_layer_sizes[0], batch_first=True))

        # 定义后续的LSTM层
        for i in range(1, self.num_layers):
            self.lstm_layers.append(nn.LSTM(hidden_layer_sizes[i - 1], hidden_layer_sizes[i], batch_first=True))

        # 定义全连接层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_layer_sizes[-1], 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, output_dim)
        )

    def forward(self, input_seq):  # torch.Size([16, 512])
        # 前向传播的过程是输入->LSTM层->全连接层->输出
        # 在观察查看LSTM输入的维度，LSTM的第一个输入input_size维度是(batch, seq_length, H_in) batch是batch size , seq_length是序列长度，H_in是输入维度，也就是变量个数
        # LSTM的第二个输入是一个元组，包含了h0,c0两个元素，这两个元素的维度都是（D∗num_layers,N,H_out)，
        # D=1表示单向网络，num_layers表示多少个LSTM层叠加，N是batch size，H_out表示隐层神经元个数

        # 数据预处理
        # 改变输入形状，适应网络输入[batch, seq_length, H_in]
        # 注意：这里是 把数据进行了堆叠 把一个1*1024 的序列 进行 划分堆叠成形状为 32 * 32， 就使输入序列的长度降下来了
        # 序列如果 以1024 长度输入进网络，LSTM 容易发生 梯度消失或者爆炸，不容易训练出好结果
        # 当然， 还可以 堆叠 为其他形状的矩阵
        # input_seq = input_seq.view(self.batch_size, 32, 32)
        lstm_out = input_seq

        for lstm in self.lstm_layers:
            lstm_out, _ = lstm(lstm_out)  ## 进行一次LSTM层的前向传播
        # print(lstm_out.size())  # torch.Size([32, 32, 32])
        out = self.classifier(lstm_out[:, -1, :])  # torch.Size([32, 10]  # 仅使用最后一个时间步的输出
        return out

class CustomDataset(Dataset):
    def __init__(self, csv_file):
        # 从CSV文件加载数据
        self.data = pd.read_csv(csv_file)
        self.data=self.data[0:1280]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取数据和标签
        data_values = torch.tensor(self.data.iloc[idx, :1024].values, dtype=torch.float32)
        label = torch.tensor(self.data.iloc[idx, 1024], dtype=torch.long)  # 假设标签是整数类型

        return data_values, label

# # 使用示例
file_name="samples_data_10c.csv"
dataset = CustomDataset(file_name)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

model_path="best_model_lstm.pt"
model=torch.load(model_path).to("cuda")
model.eval()
predicted=torch.tensor([]).to("cuda")
for batch_data, batch_labels in data_loader:
    batch_data=batch_data.view(batch_size, 32, 32).to("cuda")
    test_output =model(batch_data)
    probabilities = F.softmax(test_output, dim=1)
    predicted_new = torch.argmax(probabilities, dim=1)
    predicted=torch.cat([predicted,predicted_new],dim=0)

predicted=predicted.cpu()
predicted=predicted.numpy()

df=pd.DataFrame(predicted,columns=['label'])

label_counts = df['label'].value_counts()

replacement_dict = {
    0: '正常运行',
    1: '7英寸内圈故障',
    2: '7英寸滚珠故障',
    3: '7英寸外圈故障',
    4: '14英寸内圈故障',
    5: '14英寸滚珠故障',
    6: '14英寸外圈故障',
    7: '21英寸内圈故障',
    8: '21英寸滚珠故障',
    9: '21英寸外圈故障',
}

# 使用replace函数将label列中的值根据预定义的字典进行替换
df['label'] = df['label'].replace(replacement_dict)
label_counts = df['label'].value_counts()
label_counts_df = label_counts.reset_index()
label_counts_df.columns = ['label', 'count']


# 将 DataFrame 转换为 JSON 格式并保存为文件
label_counts_df.to_json("label_counts.json", orient="records")

print("标签计数已保存到 label_counts.json 文件中。")

# 将 DataFrame 保存为 CSV 文件
# label_counts_df.to_csv("label_counts.csv", index=False)
# label_counts=pd.DataFrame(label_counts,columns='label')

# df.to_csv("label_counts.csv", index=False)
# # 将修改后的DataFrame保存到CSV文件中
# df.to_csv("label_counts.csv", index=False)
# json_data=json.dumps(predicted.tolist())
# print(json_data)




