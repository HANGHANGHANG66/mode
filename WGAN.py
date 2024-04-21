import warnings
from matplotlib import pyplot as plt
from sklearn import preprocessing
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch.utils.data.dataset as Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyDataset(Dataset.Dataset):
    def __init__(self, path):
        values=pd.read_csv(path)
        values=values.iloc[:,:36]
        min_max_normalizer = preprocessing.MinMaxScaler(feature_range=(0, 1))
        scaled_data = min_max_normalizer.fit_transform(values)
        data = pd.DataFrame(scaled_data)
        self.data = torch.tensor(data.values)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        g_data = torch.Tensor(self.data[index])
        return g_data

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

class Critic(nn.Module):  # Critic代替了原来的Discriminator
    def __init__(self):
        super(Critic, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(36, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.main(x)
        return x

data = "normal.csv"
H_data = MyDataset(data)
H_iter = DataLoader(H_data, batch_size=50, shuffle=True)

Critic = Critic().to(device)
Generator = Generator().to(device)

# Critic和Generator的优化器不同于原来的GAN
c_optim = torch.optim.RMSprop(Critic.parameters(), lr=0.00005)
g_optim = torch.optim.RMSprop(Generator.parameters(), lr=0.00005)

# WGAN不需要损失函数，但需要对输出进行裁剪
clip_value = 0.01

critic_losses = []
generator_losses = []
best_generator_loss = float('inf')

for epoch in range(1000):
    c_epoch_loss = 0
    g_epoch_loss = 0
    for step, real in enumerate(H_iter):
        real = real.to(device)
        size = real.size(0)
        random_noise = torch.randn(size, 36).to(device)
        fake = Generator(random_noise)

        # 训练 Critic
        for p in Critic.parameters():  # 对 Critic 的参数进行裁剪
            p.data.clamp_(-clip_value, clip_value)

        c_optim.zero_grad()
        real_output = Critic(real.float())
        fake_output = Critic(fake.detach())
        c_loss = -torch.mean(real_output) + torch.mean(fake_output)  # 使用负号因为是最大化问题
        c_loss.backward()
        c_optim.step()

        # 训练 Generator
        if step % 5 == 0:  # 每训练 Critic 5次再训练
            g_optim.zero_grad()
            fake = Generator(random_noise)
            fake_output = Critic(fake)
            g_loss = -torch.mean(fake_output)
            g_loss.backward()
            g_optim.step()

            g_epoch_loss += g_loss.item()

        c_epoch_loss += c_loss.item()

    c_epoch_loss /= len(H_iter)
    g_epoch_loss /= len(H_iter)
    if g_epoch_loss < best_generator_loss:
        best_generator_loss = g_epoch_loss
        torch.save(Generator.state_dict(), 'best_generator_model.pth')
    print(f'Epoch {epoch}: Critic Loss: {c_epoch_loss}, Generator Loss: {g_epoch_loss}')
    critic_losses.append(c_epoch_loss)
    generator_losses.append(g_epoch_loss)

plt.plot(critic_losses, label='Critic Loss')
plt.plot(generator_losses, label='Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('WGAN Critic and Generator Losses')
plt.legend()
plt.show()



