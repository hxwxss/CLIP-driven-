import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torchvision.models import vgg16
from torchvision.datasets import ImageFolder
import VGG
from VGG import *
# 设置随机种子，以便结果可复现
torch.manual_seed(42)

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 CIFAR-10 测试集并进行预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_dataset = ImageFolder(root='./train', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 创建 VGG 模型实例并加载预训练的权重
model = VGG(num_classes=10).to(device)  # CIFAR-10 数据集有 10 个类别
model.load_state_dict(torch.load('vgg_model.ckpt'))

# 将模型设置为评估模式
model.eval()

# 在测试集上进行评估
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        # 将数据加载到设备上
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        # 统计预测结果
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# 打印准确率
accuracy = 100 * correct / total
print(correct,total)
print(f'Test Accuracy: {accuracy:.2f}%')
