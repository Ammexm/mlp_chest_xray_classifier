import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import read_image
from PIL import Image

import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# 定义自定义数据集
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # 第2列是图像路径
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 2])
        image = Image.open(img_name).convert('RGB')
        # 第4列是标签
        label = self.dataframe.iloc[idx, 4]       
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 数据预处理和转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),         # 将图像转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])


# 定义CNN模型
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def main():

    # # 读取CSV文件
    # csv_file = "mimic_label.csv"
    # data_df = pd.read_csv(csv_file)
    # # 查看数据的前几行，确保格式正确
    # print(data_df.head())

    # 创建数据集
    csv_file = "mimic_label.csv"
    root_dir = "mimic-cxr-jpg/2.0.0/files/"
    custom_dataset = CustomDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)

    # 定义训练集和测试集的比例
    total_samples = len(custom_dataset)
    train_ratio = 0.7  # 70%用于训练
    test_ratio = 1 - train_ratio  # 30%用于测试

    # 计算划分的样本数量
    train_size = int(train_ratio * total_samples)
    test_size = total_samples - train_size

    # 使用random_split函数划分数据集
    train_dataset, test_dataset = random_split(custom_dataset, [train_size, test_size])

    # 创建数据加载器
    batch_size = 256
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("train_size:", train_size)
    print("test_size:", test_size)
    # train_size: 212953
    # test_size: 5323

    # 初始化模型和损失函数
    num_classes = 2  # 正面和侧面两个类别
    model = CNNModel(num_classes)
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("cuda is available")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    import time
    from datetime import datetime
    from torch.utils.tensorboard import SummaryWriter
    writer_dir = "log/"
    time_stamp = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    writer1 = SummaryWriter(writer_dir + '/' + time_stamp + '/loss')
    writer2 = SummaryWriter(writer_dir + '/' + time_stamp + '/acc')

    print("writer_dir:", writer_dir)

    # 训练模型
    num_epochs = 5

    for epoch in range(num_epochs):
        
        start_time1 = time.time()
        print("epoch", epoch+1, "starting training. train_dataloader:", len(train_dataloader))
        
        model.train()
        running_loss = 0.0
        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)        
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        end_time1 = time.time()
        training_time = end_time1 - start_time1
        # 将训练时间转换为小时、分钟和秒
        hours1, remainder1 = divmod(training_time, 3600)
        minutes1, seconds1 = divmod(remainder1, 60)
        print(f"Training time: {int(hours1)} hours, {int(minutes1)} minutes, {int(seconds1)} seconds")

        model_path = "mlp_lateral_front/results/current_checkpoint.pth"
        print("Saving model in", model_path)
        torch.save(model.state_dict(), model_path)
        
        start_time2 = time.time()
        print("epoch", epoch+1, "starting testing. test_dataloader:", len(test_dataloader))
        # 计算测试准确率
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_dataloader:
                images = images.to(device)
                labels = labels.to(device) 
                outputs = model(images)  
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        end_time2 = time.time()
        testing_time = end_time2 - start_time2
        # 将训练时间转换为小时、分钟和秒
        hours2, remainder2 = divmod(testing_time, 3600)
        minutes2, seconds2 = divmod(remainder2, 60)
            
        test_accuracy = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_dataloader)}, Test Accuracy: {test_accuracy}")
        writer1.add_scalar('loss/epoch_loss', running_loss / len(train_dataloader), epoch+1)
        writer2.add_scalar('acc', test_accuracy, epoch+1)
        
        print(f"Testing time: {int(hours2)} hours, {int(minutes2)} minutes, {int(seconds2)} seconds")
        

    # 保存模型
    torch.save(model.state_dict(), "mlp_lateral_front/results/chest_xray_classifier.pth")


if __name__ == '__main__':
    main()

# nohup python main_chest_xray_classifier.py > output/training.log 2>&1 &