'''
@author:Yaqi Zhang
@email:admireseven@163.com
'''
import time
import cv2
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from torch.autograd import Variable

epochs = 100
batch_size = 512
LR = 0.001
download_mnist = True
# 设置训练和测试数据集
# 导入数据集并进行数据增强
# 数据增强是对数据集中的图片进行平移旋转等变换。
# 数据增强只针对训练集，使训练集的图片更具有多样性，让训练出来的模型的适应性更广。
# 使用数据增强会使训练准确率下降，但是可以有效提高测试准确率。
train_data = torchvision.datasets.MNIST(
    root='./data/', train=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        torchvision.transforms.RandomRotation((-10, 10)),#将图片随机旋转（-10,10）度
        torchvision.transforms.ToTensor(), #将PIL图片或者numpy.ndarray转成Tensor类型
        torchvision.transforms.Normalize((0.1307,), (0.3081,))]),
    download=download_mnist
)

test_data = torchvision.datasets.MNIST(
    root='./data/', train=False,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))])
)

# 提取数据集数据
train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size,
                               shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size,
                              shuffle=True)

# 可视化MNIST
# images, lables = next(iter(train_loader))
# img = torchvision.utils.make_grid(images, nrow = 10)
# img = img.numpy().transpose(1, 2, 0)
# cv2.imshow('img', img)
# cv2.waitKey(0)

# 定义神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层
        # input shape : (1,28,28)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=32,  # n_filters
                kernel_size=5,
                stride=1, padding=0
            ),  # output shape : (32,24,24)
            nn.ReLU(),  # 激活函数
        )
        # input shape : (32,24,24)
        self.conv2 = nn.Sequential(  # 24-5/1+1 = 20
            nn.Conv2d(32, 32, 5, 1, 0),  # output shape : (32,20,20)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output shape 20-2/2+1(32,10,10)
            nn.Dropout(0.25)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 0),  # 10-3/1+1 = 11output shape:(64,8,8)
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 0),  # 8-3/1+1 = 6 64*6*6
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 6-2/2+1 = 3 * 3 * 64
            nn.Dropout(0.25)
        )
        # 全连接层, output 10 classes
        self.out = nn.Sequential(
            nn.Linear(64 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # 防止过拟合
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)  # 将参数扁平化，否则输入至全连接层会报错
        output = self.out(x)
        return output


# 训练神经网络
# 实例化模型
cnn = CNN()
# 加载模型
model_path = './mnist.pth'
cnn.load_state_dict(torch.load(model_path))
cnn.eval()
# print(cnn)

# 如果有GPU，将所有模型参数处理为cuda类型
# 否则，不作任何处理
if torch.cuda.is_available():
    print("GPU state:", torch.cuda.is_available())
    cnn.cuda()

# 如果有GPU，将数据处理为cuda类型
# 否则，将数据处理为Variable类型
def Transfer_Variable_or_Cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

# 定义存储数据的列表
train_acces = []
test_acces = []

# 定义模型参数优化方法为Adam
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
# 定义损失函数类型为交叉熵类型
loss_func = nn.CrossEntropyLoss()

print("starting training...")
start_time = time.time()  # 记录训练开始时刻

for epoch in range(epochs):
    running_loss = 0.0  # 初始误差
    train_correct = 0  # 初始分类正确样本数
    for step, (x_train, y_train) in enumerate(train_loader):
        x_train = Transfer_Variable_or_Cuda(x_train)
        y_train = Transfer_Variable_or_Cuda(y_train)
        output = cnn(x_train)  # 前向传播
        # torch.max()[1] 返回最大值对应的索引（最大概率对应的索引，即数字）
        _, pred = torch.max(output.data, 1)
        optimizer.zero_grad()  # 模型参数梯度清零
        loss = loss_func(output, y_train)
        loss.backward()
        optimizer.step()  # 更新梯度
        # 误差累计
        running_loss += loss.item()
        # 训练集中分类正确样本数，用于后续计算模型在训练集上的准确度
        train_correct += torch.sum(pred == y_train.data)
        if step % 32 == 0:
            print(
                "Epoch{}/{} | Loss is:{:.4f} | Train Accuracy is:{:.4f}%".format(
                    epoch + 1, epochs,
                    running_loss / len(train_data),
                    100 * train_correct / len(train_data)))
        train_acc = 100 * train_correct / len(train_data)
        train_acces.append(train_acc.cpu().numpy().tolist())

    testing_correct = 0  # 初始分类正确样本数
    for step, (x_test, y_test) in enumerate(test_loader):
        x_test, y_test = Transfer_Variable_or_Cuda(x_test), \
                         Transfer_Variable_or_Cuda(y_test)
        output = cnn(x_test)
        _, pred = torch.max(output.data, 1)
        # 测试集中分类正确样本数，用于后续计算模型在测试集上的准确度
        testing_correct += torch.sum(pred == y_test.data)
    acc = testing_correct / len(test_data)
    test_acces.append(acc.cpu().numpy().tolist())

    # 保存测试准确率最大的模型
    if test_acces[-1] >= max(test_acces):
        # 每个batch训练完后保存模型
        torch.save(cnn.state_dict(), './mnist.pth')
        # 每个batch训练完后保存优化器
        torch.save(optimizer.state_dict(), './optimizer.pth')

    print("\nEpoch{}/{} | Loss is:{:.4f} | Test Accuracy is:{:.4f}%\n".format(
        epoch + 1, epochs,
        running_loss / len(train_data),
        100 * testing_correct / len(test_data)))

print('\n\033[1;31m The Max Test Accuracy : {:.2f}%\033[0m'.format(
    100. * max(test_acces)))
stop_time = time.time()  # 记录训练结束时刻
print("training time is:{:.4f}s".format(stop_time - start_time))  # 输出训练时长(s)

# test_output = cnn(x_test[:10])
# y_pred = torch.max(test_output,1).cuda()
# print(y_pred,'prediction number')
# print(y_test,'real number')

torch.save(cnn.state_dict(), './CNN_MNIST.pkl')
print("save model successfully!")

plt.figure()
plt.plot(range(1, 101), test_acces, label='Test Accuracy')
# plt.plot(range(1, 101), train_acces, label='Train Accuracy')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.tick_params(axis='both', labelsize=10)
plt.show()
