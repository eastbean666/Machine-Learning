"""
案例：
    演示CNN的综合案例，图像分类
回顾：深度学习项目的步骤
    1.准备数据库
        这里我们用的时候,计算机视觉模块torchvision白带的CIFAR18数据集，包含6W张(32,32,3)的图片，5W训练集，1W张测试集，18个分类，每个分类6K张图片。
        你需要单安装一下 torchvision包,N:pip install torchvision
    2.搭建(卷积)神经网络
    3.模型训练
    4.模型测试
卷积层：
    提取图像的局部特征->特征图(Feature Map),计算方式：N=（W-F+2P）//S+1
    每个卷积核都是一个神经元
池化层:
    降维，有最大池化和平均池化
    池化只在HW上做调整，通道上不改变
"""

#导包
import torch
import torch.nn as nn
from caffe2.python.helpers import train
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader, dataloader
import time
import  matplotlib.pyplot as plt
from torchinfo import summary

#每批次样本数
BATCH_SIZE = 8
#1.准备数据集
def creat_dataset():
    #1.获取训练集
    #参1：数据集路径 参2：是否是训练集 参3：数据预处理->张量数据 参4：是否联网下载(直接用我给的，不用下)
    train_dataset = CIFAR10(root='./data', train=True, transform=ToTensor(), download=True)
    #2.获取测试集
    test_dataset = CIFAR10(root='./data', train=False, transform=ToTensor(), download=True)
    #3.返回数据集
    return train_dataset,test_dataset
#2.搭建卷积神经网络
class ImageClassifier(nn.Module):
    #1.初始化父类成员，搭建神经网络
    def __init__(self):
        #1.1初始化父类成员
        super().__init__()
        #1.2第一个卷积层，输入3个通道，输出6个通道，卷积核大小3*3，步长1，填充0
        self.conv1=nn.Conv2d(3,6,3,1,0)
        #1.3第一个池化层，池化核大小2*2，步长2
        self.pool1=nn.MaxPool2d(2,2)
        #1.4第二个卷积层，输入6个通道，输出16个通道，卷积核大小3*3，步长1，填充0
        self.conv2=nn.Conv2d(6,16,3,1,0)
        #1.5第二个池化层，池化核大小2*2，步长2
        self.pool2=nn.MaxPool2d(2,2)
        #1.6全连接层，输入16*6*6，输出120
        self.fc1=nn.Linear(16*6*6,120)
        #1.7全连接层，输入120，输出84
        self.fc2=nn.Linear(120,84)
        #1.8全连接层，输入84，输出10
        self.output=nn.Linear(84,10)
    #2.定义前向传播
    def forward(self,x):
        #第1层：卷积层（加权求和）->激活函数（Relu）->池化层（降维）
        #分解版
        #x=self.conv1(x)  x=torch.relu(x) x=self.pool1(x)
        #合并版
        x=self.pool1(torch.relu(self.conv1(x)))
        #第2层：卷积层（加权求和）->激活函数（Relu）->池化层（降维）
        x=self.pool2(torch.relu(self.conv2(x)))
        #细节:全连接层只能处理二位数据，所以需要将数据进行拉平(8,16,6,6)->(8,16*6*6)
        #参1:样本数（行数） 参2：列数（特征数），-1表示自动计算
        x=x.reshape(x.shape[0],-1)
        #print(f'x.shape:{x.shape}')
        #第3层：全连接层（加权求和）->激活函数（Relu）
        x=torch.relu(self.fc1(x))
        #第4层：全连接层（加权求和）->激活函数（Relu）
        x=torch.relu(self.fc2(x))
        #第5层：全连接层（加权求和）->激活函数（Relu）   输出层
        return self.output(x)  #后续用多分类交叉熵损失函数CrossEntropyLoss() = softmax()激活函数+损失计算
#3.模型训练
def train01(train_dataset):
    #1.创建数据加载器
    dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
    #2.创建模型对象
    model = ImageClassifier()
    #3.创建损失函数对象
    criterion = nn.CrossEntropyLoss()
    #4.创建优化器对象
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    #5.循环遍历epoch，开始每轮的训练动作
    #5.1定义变量，记录训练的总轮数
    epochs = 50
    #5.2遍历，完成每轮的所以批次的训练动作
    for epoch_idx in range(epochs):
        #5.2.1定义变量，记录：总损失，总样本数据量，预测正确样本数量个数，训练（开始）时间
        total_loss,total_samples,total_correct,start = 0.0,0,0,time.time()
        #5.2.2遍历数据加载器,获取到每批次的数据
        for x,y in dataloader:
            #5.2.3切换训练模式
            model.train()
            #5.2.4模型预测
            y_pred=model(x)
            #5.2.5计算损失
            loss=criterion(y_pred,y)
            #5.2.6梯度清零 +反向传播 +参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #5.2.7统计预测正确的样本个数
            #print(y_pred)
            #argmax() 返回最大值对应的索引，充当->该图片的预测分类
            #tensor([9,1,7,8,2,6,4,1])
            # print(torch.argmax(y_pred,dim=-1))#-1表示行
            # print(y)
            # print((torch.argmax(y_pred)==y))#是否预测正确
            # print((torch.argmax(y_pred) == y).sum())#预测正确的样本数
            total_correct += (torch.argmax(y_pred,dim=-1)==y).sum()
            #5.2.8统计当前批次的总损失
            total_loss += loss.item()*len(y)#第一批总损失+第二批总损失+.....     第一轮平均损失*批次样本数
            #5.2.9 统计当前批次的总样本数
            total_samples += len(y)
            ## break 每轮只训练1批，提高训练效率，减少训练时长，只有测试会这么写，实际开发绝不要这样做。
        #5.2.10走这里 说明一轮训练完毕，打印该轮的训练信息
        print(f'第{epoch_idx+1}轮，总损失:{total_loss/total_samples:.4f},总准确率:{total_correct/total_samples:.4f},训练时长:{time.time()-start:.4f}')
        #break 这里写break，意味着只训练一轮
    #6.保存模型
    torch.save(model.state_dict(),'./model/image_model.pth')

#4.模型测试
def test(test_dataset):
    #1.创建数据加载器
    dataloader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)
    #2.创建模型对象
    model = ImageClassifier()
    #3.加载模型参数
    model.load_state_dict(torch.load('./model/image_model.pth'))
    #4.定义变量统计，预测正确的样本个数，总样本个数
    total_correct,total_samples = 0,0
    #5.遍历数据加载器，获取每批次数据
    for x,y in dataloader:
        #6.切换测试模式
        model.eval()
        #7.模型预测
        y_pred=model(x)
        #8.统计预测正确的样本个数
        #因为训练时用了CrossEntropyLoss(),所以搭建神经网络时没有加入softmax()激活函数,这里要用argmax()来模拟
        total_correct += (torch.argmax(y_pred,dim=-1)==y).sum()
        #9.统计总样本个数
        total_samples += len(y)
    #10.打印测试结果
    print(f'ACC:{total_correct/total_samples:.4f}')

    pass
#5.测试
if __name__ == '__main__':
    #1.获取数据集
    train_dataset,test_dataset = creat_dataset()
    # print(f'训练集:{train_dataset.data.shape}')#训练集:(50000, 32, 32, 3)
    # print(f'测试集:{test_dataset.data.shape}')#测试集:(10000, 32, 32, 3)
    # print(f'数据集类别:{train_dataset.class_to_idx}')
    # #{'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
    # plt.figure(figsize=(2,2))
    # plt.imshow(train_dataset.data[11])
    # plt.title(train_dataset.targets[11])
    # plt.show()
    #2.搭建神经网络
    # model = ImageClassifier()
    # #查看模型参数 参1:模型 参2:输入数据大小 参3:batch_dim
    # summary(model,input_size=(BATCH_SIZE,3,32,32))
    #3.模型训练
    #train01(train_dataset)
    #4.模型评估
    test(test_dataset)
