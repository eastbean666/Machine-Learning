"""
案例：
    ANN(人工神经网络）案例：手机价格分类案例
背景：
    基于手机的20列特征->预测手机的价格区间（4个区间），可以使用机器学习，也可以用深度学习（推荐）
ANN案例的实现步骤：
    1.构建数据集
    2.搭建神经网络
    3.模型训练
    4.模型测试
优化思路：
    1.优化方法 SGD->Adam
    2.学习率 0.001->0.0001
    3.对数据进行标准化
    4.增加网络深度，调整每层神经元数量
    5.调整训练轮数
    ......
"""

#导包
import torch    #pytorch架构，封装了张量的各种操作
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader,TensorDataset  #数据集对象，数据->tensor->数据集->数据加载器
import torch.nn as nn  #neural network，封装了神经网络的操作
import torch.optim as optim #优化器
from sklearn.model_selection import train_test_split #训练集和测试集划分
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt  #绘图
import numpy as np #矩阵运算
import pandas as pd #数据处理
import time #时间模块
from torchinfo import summary


#todo 1.构建数据集
def creat_dataset():
    #1.加载csv文件数据集
    data= pd.read_csv("data/手机价格预测.csv")
    #print(data.head())
    #print(data.shape)

    #2.获取x特征列 和y标签列
    x,y=data.iloc[:,:-1],data.iloc[:,-1]
    #print(x.head())
    #print(y.head())
    #print(x.shape)
    #print(y.shape)

    #3.把特征列转为浮点型
    x=x.astype(np.float32)
    y=y.astype(np.int64)

    #4.切分训练集和测试集
    #参1：特征列 参2：标签列 参3：测试集比例 参4：随机种子 参5：参数用于分层抽样，确保训练集和测试集中各类别的比例与原始数据集保持一致。
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
    #优化（1）：数据标准化
    transfer = StandardScaler()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.transform(x_test)
    #5.把数据集封装成一张数据集 思路：数据->张量->数据集TensorDataSet->数据加载器DataLoader
    train_dataset=TensorDataset(torch.tensor(x_train),torch.tensor(y_train.values))
    test_dataset=TensorDataset(torch.tensor(x_test),torch.tensor(y_test.values))

    #6.返回结果
    return train_dataset,test_dataset,x_train.shape[1],len(np.unique(y_train))
#todo 2.搭建神经网络
class PhonePriceClassifier(nn.Module):
    def __init__(self,input_dim,output_dim):#输入维度，输出维度
        #2.1初始化父类成员
        super().__init__()
        #2.2定义神经网络结构
        #隐藏层1
        self.linear1=nn.Linear(input_dim,128)
        #隐藏层2
        self.linear2=nn.Linear(128,256)
        #输出层
        self.output=nn.Linear(256,output_dim)
    #定义前向传播方法（forward）
    def forward(self,x):
        #2.1隐藏层1：加权求和+激活函数（relu）
        x=torch.relu(self.linear1(x))
        #2.2隐藏层2：加权求和+激活函数（relu）
        x=torch.relu(self.linear2(x))
        #2.3输出层：加权求和+激活函数（softmax）->这里只需要加权求和
        #正常写法，但是不需要，后续用多分类交叉熵损失函数CrssEntropyLoss替代
        #CrossEntropyLoss = softmax（）+损失计算
        #x=torch.softmax(self.output(x),dim=1)
        x=self.output(x)
        #2.4返回结果
        return x
#todo 3.模型训练
def train(train_dataset,input_dim,output_dim):
    #1.创建数据加载器，流程：数据->张量->数据集->数据加载器
    #参1:数据集（1600条） 参2:批次大小 参3:是否打乱数据（训练集打乱，测试集不打乱）
    train_loader=DataLoader(train_dataset,batch_size=16,shuffle=True)
    #2.创建神经网络模型对象
    model=PhonePriceClassifier(input_dim,output_dim)
    #3.定义损失函数，因为是多分类，这里是多分类交叉熵损失函数
    criterion=nn.CrossEntropyLoss()
    #4.创建优化器对象
    optimizer=optim.Adam(model.parameters(),lr=0.0001)
    #5.模型训练
    #5.1定义变量，记录训练的总论数
    epochs=200
    #5.2开始（每轮的）训练
    for epoch in range(epochs):
        #5.2.1 定义变量，记录每次训练的损失值，训练批次数
        total_loss,batch_num=0,0
        #5.2.2 定义变量，表示训练开始的时间
        start_time=time.time()
        #5.2.3 开始本轮的各个批次的训练
        for x,y in train_loader:
            #5.2.3.1 切换模型（状态）训练
            model.train()#训练模型  model.eval()测试模式
            #5.2.3.2 模型预测
            y_pred=model(x)
            #5.2.3.3 计算损失值
            loss=criterion(y_pred,y)
            #5.2.3.4 梯度清零，反向传播，优化参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #5.2.3.5 累计损失值
            total_loss+=loss.item()#把本轮的每批次（16条）的平均损失值累计
            batch_num+=1
        #5.2.4至此本轮训练结束，打印训练信息
        print(f"第{epoch+1}轮训练结束，平均损失值：{total_loss/batch_num:.4f}，用时：{time.time()-start_time:.2f}")
    #6.走到这里说明多轮训练结束，保存模型（参数）
    #参1：模型对象的参数（权重矩阵，偏执矩阵 参2：保存的文件名
    print(f"\n\n模型的参数信息{model.state_dict()}\n\n")
    torch.save(model.state_dict(),"phone_price_classifier.pth")#后缀名用pth,pkl,pickle均可

#todo 4.模型测试
def evaluate(test_dataset,input_dim,output_dim):
    #1.创建神经网络分类对象
    model=PhonePriceClassifier(input_dim,output_dim)
    #2.加载模型参数
    model.load_state_dict(torch.load("phone_price_classifier.pth"))
    #3.创建测试集数据加载器，流程：数据->张量->数据集->数据加载器
    #参1:数据集（400条） 参2:批次大小 参3:是否打乱数据（训练集打乱，测试集不打乱）
    test_loader=DataLoader(test_dataset,batch_size=8,shuffle=False)
    #4.定义变量，记录预测正确的样本数
    correct=0
    #5.从数据加载器中，获取每批次的数据
    for x,y in test_loader:
        #5.1切换模型（状态）测试
        model.eval()
        #5.2模型预测
        y_pred=model(x)
        #print("预测结果1：", y_pred)
        #5.3根据加权求和，得到类别，用argmax()获得最大值对应的下标就是类别
        y_pred=torch.argmax(y_pred,dim=1) #dim=1表示逐行处理
        print("预测结果：",y_pred)
        #5.4统计预测正确的样本数
        #print(y_pred==y) #tensor([ True,  True, False,  True,  True,  True, False, False])
        #print((y_pred==y).sum())#tensor(5)
        correct+=(y_pred==y).sum()
    #6.走到这里，模型预测结束,打印准确率即可
    print(f"准确率：{correct/len(test_dataset):.4f}")

#todo 5.测试
if __name__ == '__main__':
    #1.准备数据集
    train_dataset,test_dataset,input_dim,output_dim=creat_dataset()
    #print("数据集大小：",len(train_dataset),len(test_dataset))
    #print("输入维度：",input_dim)
    #print("输出维度：",output_dim)
    #2.搭建神经网络模型
    model=PhonePriceClassifier(input_dim,output_dim)
    #计算模型参数
    #参1：模型对象 参2：输入数据的形状（批次大小，输入特征数）每批16条每列20个特征
    #summary(model,(16,input_dim))
    #3.模型训练
    train(train_dataset,input_dim,output_dim)
    evaluate(test_dataset,input_dim,output_dim)