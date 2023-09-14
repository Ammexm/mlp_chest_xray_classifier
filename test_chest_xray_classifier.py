import pandas as pd
import torch
from main_chest_xray_classifier import CNNModel, transform
from PIL import Image

# 初始化模型架构
num_classes = 2  # 正面和侧面两个类别
model = CNNModel(num_classes)
# 检查CUDA是否可用
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("cuda is available")
else:
    device = torch.device("cpu")
model = model.to(device)
# 加载模型权重
model.load_state_dict(torch.load("output/current_checkpoint.pth"))

def classify_image(image_path, model):
    # 加载并预处理图像
    image = Image.open(image_path)
    image = transform(image)

    # 将图像传递给模型进行推断
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0)  # 添加批次维度
        image = image.to(device)
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)  # 获取预测类别

    # 返回分类结果
    return predicted.item()

results = []  # 存储分类结果的列表

# 读取包含图像路径的CSV文件
csv_file = "mimic_labelwithpath.csv"  # 替换为您的CSV文件路径
df = pd.read_csv(csv_file)
for index, row in df.iterrows():
    id = row['dicom_id']    # 获取图像id
    image_path = row['image_path']  # 获取图像路径
    gt = row['ViewPosition']  # 获取图像原label
    
    prediction = classify_image(image_path, model)  # 对图像进行分类

    # 将分类结果和图像路径添加到结果列表
    results.append({'dicom_id': id, 'image_path': image_path, 'prediction': prediction, 'gt': gt})

# 创建包含分类结果的DataFrame
results_df = pd.DataFrame(results)

# 保存结果到新的CSV文件
results_csv_file = "data/prediction_img2label.csv"  # 替换为您要保存结果的CSV文件路径
results_df.to_csv(results_csv_file, index=False)

