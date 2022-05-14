import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

start = time.time()
# 矩阵标准化
def data_tf(data):
    data -= np.mean(data, axis=0)
    data /= np.std(data, axis=0)
    return data
train_set = np.loadtxt(open('./pca_train_matrix.csv'), delimiter=',', skiprows=0)
test_set = np.loadtxt(open('./pca_test_matrix.csv'), delimiter=',', skiprows=0)
train_lable = np.loadtxt(open('./train_lable.csv'), delimiter=',', skiprows=0)
test_lable = np.loadtxt(open('./test_label.csv'), delimiter=',', skiprows=0)
num_label = np.unique(train_lable) # 输出神经元个数276

# 数据预处理

train_set = data_tf(np.transpose(train_set))
test_set = data_tf(np.transpose(test_set))


# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class GetLoader(torch.utils.data.Dataset):
    # 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)


train_set_ = GetLoader(train_set, train_lable)
test_set_ = GetLoader(test_set, test_lable)
train_data = DataLoader(train_set_, batch_size=64, shuffle=True)  # 训练数据
test_data = DataLoader(test_set_, batch_size=32, shuffle=False)  # 测试数据

# 定义一个类，继承自 torch.nn.Module，torch.nn.Module是callable的类
# 在整个类里面重新定义一个标准的BP全连接神经网络，网络一共是四层，
# 层数定义：62, 200， 350， 500， 625
# 其中输入层62个节点，输出层是625个节点，分别代表625个人，其他的层都是隐藏层。
# 我们使用了Relu的激活函数，而不是sigmoid激活函数
# 整个子类需要重写forward函数，

output = len(num_label)


class BPNNModel(nn.Module):
    def __init__(self):
        # 调用父类的初始化函数，必须要的
        super(BPNNModel, self).__init__()

        # 创建四个Sequential对象，Sequential是一个时序容器，将里面的小的模型按照序列建立网络
        self.layer1 = nn.Sequential(nn.Linear(62, 200), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(200, 350), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(350, 500), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(500, 625))

    def forward(self, img):
        # 每一个时序容器都是callable的，因此用法也是一样。
        img = self.layer1(img)
        img = self.layer2(img)
        img = self.layer3(img)
        img = self.layer4(img)
        return img


# 创建和实例化一个整个模型类的对象
model = BPNNModel()
# 打印出整个模型
print(model)

# 定义 loss 函数，这里用的是交叉熵损失函数(Cross Entropy)，这种损失函数之前博文也讲过的。
criterion = nn.CrossEntropyLoss()
# 我们优先使用随机梯度下降，lr是学习率: 0.1
optimizer = torch.optim.SGD(model.parameters(), 1e-1)

# 为了实时观测效果，我们每一次迭代完数据后都会，用模型在测试数据上跑一次，看看此时迭代中模型的效果。
# 用数组保存每一轮迭代中，训练的损失值和精确度，也是为了通过画图展示出来。
train_losses = []
train_acces = []
# 用数组保存每一轮迭代中，在测试数据上测试的损失值和精确度，也是为了通过画图展示出来。
eval_losses = []
eval_acces = []
model = model.double()

for e in range(30):
    # 4.1==========================训练模式==========================
    train_loss = 0
    train_acc = 0
    model.train()   # 将模型改为训练模式

    # 每次迭代都是处理一个小批量的数据，batch_size是64
    for im, label in train_data:
        im = Variable(im)
        label = Variable(label)

        # 计算前向传播，并且得到损失函数的值
        out = model(im)
        loss = criterion(out, label.long())

        # 反向传播，记得要把上一次的梯度清0，反向传播，并且step更新相应的参数。
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录误差
        train_loss += loss.item()

        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        train_acc += acc

    train_losses.append(train_loss / len(train_data))
    train_acces.append(train_acc / len(train_data))

    # 在测试集上检验效果
    eval_loss = 0
    eval_acc = 0
    model.eval()  # 将模型改为预测模式

    # 每次迭代都是处理一个小批量的数据，batch_size是128
    for im, label in test_data:
        im = Variable(im)  # torch中训练需要将其封装即Variable，此处封装100特征值
        label = Variable(label)  # 此处为标签
        out = model(im)  # 经网络输出的结果
        loss = criterion(out, label.long())  # 得到误差
        # 记录误差
        eval_loss += loss.item()
        # 记录准确率
        _, pred = out.max(1)  # 得到出现最大值的位置，也就是预测得到的数即0—624
        num_correct = (pred == label).sum().item()  # 判断是否预测正确
        acc = num_correct / im.shape[0]  # 计算准确率
        eval_acc += acc

    eval_losses.append(eval_loss / len(test_data))
    eval_acces.append(eval_acc / len(test_data))
    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
          .format(e, train_loss / len(train_data), train_acc / len(train_data),
                  eval_loss / len(test_data), eval_acc / len(test_data)))

end = time.time()
print('程序运行时间是：{}'.format(end - start))


plt.title('train loss')
plt.plot(np.arange(len(train_losses)), train_losses)
plt.plot(np.arange(len(train_acces)), train_acces)
plt.title('train acc')
plt.plot(np.arange(len(eval_losses)), eval_losses)
plt.title('test loss')
plt.plot(np.arange(len(eval_acces)), eval_acces)
plt.title('test acc')
plt.show()