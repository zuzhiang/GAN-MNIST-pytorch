'''
pytorch进行神经网络训练的流程：

一.首先定义神经网络模型 model=Net()

二.定义优化器  optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

三.训练阶段，循环每个epoch，对于一个epoch：
    1.将模型设置为训练模式  model.train()
    2.将训练数据分为batch_size大小的组，对每一组中所有的样本：
        1).将梯度初始化为0  optimizer.zero_grad()
        2).对这组样本进行训练，并得到输出  output=model(data)
        3).根据预测输出和真实类别计算损失函数  loss=F.nll_loss(output,target)
        4).反向传播，计算梯度  loss.backward()
        5).更新模型参数  optimizer.step()
    3.保存模型（可省略）  torch.save(model,"mnist_torch.pkl")

四.加载已训练好的模型（可省略）

五.测试阶段，对于所有的测试样本：
    1.将模型设置为评价模式  model.eval()
    2.根据建好的模型对测试数据进行预测，得到预测结果  output=model(data)
    3.计算损失函数  test_loss+=F.nll_loss(output,target).data

上述中 model.train()和model.eval()的不同在于，model.eval()去除神经网络中的随机性，而model.train()
则保留了神经网络中的随机性。这是因为神经网络中的某些操作（如dropout等），会存在一定的随机性（dropout
会随机让神经网络中的节点失活），所以为了保证预测结果是可复现的，所以在测试阶段要设置为model.eval()。

'''
import os
import cv2
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
from keras.datasets import mnist
from torchvision.utils import save_image

batch_size = 100
epoch_num = 30
lr = 0.0002
input_dim = 100


class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 56 * 56)
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True) # inplace设为True，让操作在原地进行
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 50, 3, stride=1, padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(50, 25, 3, stride=1, padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(25, 1, 2, stride=2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 1, 56, 56)
        x = self.br(x)
        x = self.conv1(x)
        x = self.conv2(x)
        output = self.conv3(x)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=1, padding=2),
            nn.LeakyReLU(0.2,True)
        )
        self.pl1 = nn.AvgPool2d(2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.LeakyReLU(0.2,True)
        )
        self.pl2 = nn.AvgPool2d(2, stride=2)
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 7 * 7, 1024),
            nn.LeakyReLU(0.2,True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pl1(x)
        x = self.conv2(x)
        x = self.pl2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        output = self.fc2(x)
        return output


def G_train(input_dim):
    G_optimizer.zero_grad()

    noise = torch.randn(batch_size, input_dim).to(device)
    real_label = torch.ones(batch_size).to(device)
    fake_img = G(noise)
    D_output = D(fake_img)
    G_loss = criterion(D_output, real_label)

    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()


def D_train(real_img, input_dim):
    D_optimizer.zero_grad()

    real_label = torch.ones(real_img.shape[0]).to(device)
    D_output = D(real_img)
    D_real_loss = criterion(D_output, real_label)

    noise = torch.randn(batch_size, input_dim, requires_grad=False).to(device)
    fake_label = torch.zeros(batch_size).to(device)
    fake_img = G(noise)
    D_output = D(fake_img)
    D_fake_loss = criterion(D_output, fake_label)

    D_loss = D_real_loss + D_fake_loss

    D_loss.backward()
    D_optimizer.step()

    return D_loss.data.item()


def save_img(img, img_name):
    img = 0.5 * (img + 1)
    img = img.clamp(0, 1)
    save_image(img, "./imgs/" + img_name)
    # print("image has saved.")


if __name__ == "__main__":

    if not os.path.exists("./checkpoint"):
        os.makedirs("./checkpoint")

    if not os.path.exists("./imgs"):
        os.makedirs("./imgs")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=torchvision.transforms.ToTensor(),
                                   download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # 构建生成器和判别器网络
    if os.path.exists('./checkpoint/Generator.pkl') and os.path.exists('./checkpoint/Discriminator.pkl'):
        G=torch.load("./checkpoint/Generator.pkl").to(device)
        D=torch.load("./checkpoint/Discriminator.pkl").to(device)
    else:
        G = Generator(input_dim).to(device)
        D = Discriminator().to(device)

    # 指明损失函数和优化器
    criterion = nn.BCELoss()
    G_optimizer = optim.Adam(G.parameters(), lr=lr)
    D_optimizer = optim.Adam(D.parameters(), lr=lr)

    print("Training...........")
    for epoch in range(1, epoch_num + 1):
        print("epoch: ", epoch)
        for batch, (x, _) in enumerate(train_loader):
            # 对判别器和生成器分别进行训练，注意顺序不能反
            D_loss=D_train(x.to(device), input_dim)
            G_loss=G_train(input_dim)

            #if batch % 20 == 0:

            print("[ %d / %d ]  g_loss: %.6f  d_loss: %.6f" % (batch, 600, float(G_loss), float(D_loss)))

            if batch % 50 == 0:
                fake_img = torch.randn(128, input_dim)
                fake_img = G(fake_img)
                save_img(fake_img, "img_" + str(epoch) + "_" + str(batch) + ".png")
                # 保存模型
                torch.save(G, "./checkpoint/Generator.pkl")
                torch.save(D, "./checkpoint/Discriminator.pkl")