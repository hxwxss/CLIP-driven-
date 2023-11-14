import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import clip
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import VGG
import os
from PIL import Image
from VGG import *
# 定义标签到文本的映射字典
# 检查是否有可用的GPU
trans={"NGS":"segmented neutrophil","NGB":"band neutrophil","LYT":"typical Lymphocyte",
      "LYA":"atypical Lymphocyte","MON":"Monocyte","EOS":"Eosinophil","BAS":"Basophil",
       "MYO":"Myeloblast","PMO":"Promyelocyte","PMB":"bilobed Promyelocyte","MYB":"Myelocyte",
       "MMZ":"Metamyelocyte","MOB":"Monoblast","EBO":"Erythroblast","KSC":"Smudge cell"
      }
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelTex, preprocess = clip.load("ViT-B-32.pt", device=device)
modelTex = modelTex.eval()
print("CLIP is OK")
# 定义多模态数据集类
class MultiModalDataset(Dataset):
    def __init__(self, image_root, label_root, transform=None):
        self.image_root = image_root
        self.label_root = label_root
        self.transform = transform

        # 获取标签文件列表
        self.label_files = [file for file in os.listdir(label_root) if file.endswith('.txt')]

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, index):
        label_file = self.label_files[index]
        label_path = os.path.join(self.label_root, label_file)
        image_path = os.path.join(self.image_root, label_file.replace('.txt', '.jpg'))

        with open(label_path, 'r') as label_file:
            label = label_file.read()
            #= modelTex.encode_text(label)
            label = clip.tokenize(label).to(device)
            label = modelTex.encode_text(label)
            #print(label)
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label
# 设置随机种子
torch.manual_seed(42)



batch_size = 32
learning_rate = 0.001
num_epochs = 300

# 图像处理转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 假设你有以下数据，需要确保文本标签与图像数据一一对应
image_root = './dataset/images'  # 图像数据的根目录
label_root = './dataset/labels'
# 创建多模态数据集
multimodal_dataset = MultiModalDataset(image_root, label_root, transform)

multimodal_loader = DataLoader(multimodal_dataset, batch_size=batch_size, shuffle=True)
print("Data is OK")

# 加载VGG模型
vgg=VGG(num_classes=1000)

features = vgg.features

# 创建新的分类器部分
num_features = vgg.classifier[0].in_features
classifier = nn.Sequential(
    nn.Linear(num_features, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(),
    nn.Linear(4096, 512)  # 调整为需要的num_classes
)

# 创建新的VGG16模型
modelImg = nn.Sequential(
    features,
    nn.Flatten(),
    classifier
)

# 加载预训练权重文件
state_dict = torch.load('vgg16.pth', map_location='cpu')

# 获取模型的状态字典
model_dict = modelImg.state_dict()

# 选择性加载特征提取部分的权重
pretrained_dict = {k: v for k, v in state_dict.items() if k.startswith('features')}

# 更新模型的状态字典
model_dict.update(pretrained_dict)

# 加载权重
modelImg.load_state_dict(model_dict, strict=False)

modelImg = modelImg.to(device)
print("VGG16 is OK")

criterion = nn.CosineEmbeddingLoss()
optimizer = optim.Adam(list(modelImg.parameters()), lr=learning_rate)  # 只使用图像模型的参数

total_step = len(multimodal_loader)
for epoch in range(num_epochs):
    modelImg.train()  # 设置模型为训练模式
    for i, (images, labels) in enumerate(multimodal_loader):
        images = images.to(device)
        labels = labels.to(device)
        #print("fuckiong",len(labels))
        #with torch.no_grad():
            #text_inputs = torch.tensor(list(labels))
            #text_features = modelTex.encode_text(text_inputs)
        #text_features = text_features.to(device)
        #labels = torch.tensor(labels, dtype=torch.float32).to(device)

        #labels = labels.to(device)

        # 使用您的VGG模型提取图像特征
        #with torch.no_grad():
        image_features = modelImg(images) 
            
            # 使用您的VGG模型提取图像特征

        # 使用CLIP模型提取文本特征
        #with torch.no_grad():
            #text_inputs = labels
        labels = labels.squeeze()
        text_features =labels
            #modelTex.encode_text(text_inputs)

        # 在这里执行图像处理和模型的前向传播

        # 计算损失
        

        #print("image_size: ",image_features.size())
        #print("text_size: ",text_features.size())
        
        #loss = criterion(image_features, text_features, labels)
        mse_loss = nn.MSELoss()
        image_features = image_features.to(torch.float32)
        text_features  = text_features.to(torch.float32)
        loss = mse_loss(image_features, text_features)
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 保存模型的权重
torch.save(modelImg.state_dict(), 'model1.ckpt')
