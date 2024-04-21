import pandas as pd
import torch
from torch import nn


# 定义 Generator 类
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(36, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 36),
            nn.Tanh()
        )

    def forward(self, x):
        data = self.main(x)
        return data

# 实例化 Generator
generator = Generator()

# 加载保存好的模型参数
generator.load_state_dict(torch.load('best_generator_model.pth'))

# 生成3000条数据
num_samples = 3000
generated_data = []

with torch.no_grad():
    for _ in range(num_samples):
        random_noise = torch.randn(1, 36)  # 生成随机噪声
        generated_sample = generator(random_noise)  # 通过生成器生成样本
        generated_data.append(generated_sample.numpy()[0])  # 将生成的样本添加到列表中

# 将生成的数据转换为 DataFrame
generated_df = pd.DataFrame(generated_data)
generated_df['label']=0
# 保存为 CSV 文件
generated_df.to_csv('generated_data.csv', index=False)
