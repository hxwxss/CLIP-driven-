import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import clip
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os
from PIL import Image
import VGG
from VGG import *
torch.manual_seed(42)


def clip_loss(image,text_pos,text_negs):
    # 计算正确匹配的相似性
    sim_pos = torch.cosine_similarity(image, text_pos, dim=-1)
    # 计算错误匹配的相似性
    sim_negs = torch.cosine_similarity(image.unsqueeze(1), text_negs, dim=-1)
    # 计算损失
    exp_sim_pos = torch.exp(sim_pos)
    exp_sim_negs = torch.exp(sim_negs)
    loss = -torch.log(exp_sim_pos / (exp_sim_pos + torch.sum(exp_sim_negs)))
    return loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trans={"NGS":"segmented neutrophil","NGB":"band neutrophil","LYT":"typical Lymphocyte",
      "LYA":"atypical Lymphocyte","MON":"Monocyte","EOS":"Eosinophil","BAS":"Basophil",
       "MYO":"Myeloblast","PMO":"Promyelocyte","PMB":"bilobed Promyelocyte","MYB":"Myelocyte",
       "MMZ":"Metamyelocyte","MOB":"Monoblast","EBO":"Erythroblast","KSC":"Smudge cell"
      }

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
path="./check"
train_image=[]
train_label=[]
path_wait=[]

modelImg = VGG(num_classes=512)
state_dict = torch.load('model1.ckpt', map_location='cpu')

# 获取模型的状态字典
model_dict = modelImg.state_dict()

# 选择性加载特征提取部分的权重
pretrained_dict = {k: v for k, v in state_dict.items() if k.startswith('features')}

# 更新模型的状态字典
model_dict.update(pretrained_dict)

# 加载权重
modelImg.load_state_dict(model_dict, strict=False)

modelImg = modelImg.to(device)
modelImg = modelImg.to(device)
print("VGG16 is OK")

modelTex, preprocess = clip.load("ViT-B-32.pt", device=device)
modelTex=modelTex.to(device)
print("CLIP is OK")

for root,files,dire in os.walk(path):
    for name in dire:
        train_label.append("This is an image of "+name[0:3])
        img=preprocess(Image.open(path+'/'+name)).unsqueeze(0).to(device)
        train_image.append(img)
print("Data is OK")


pre_text=[]
text_bas=list(set(train_label))
for i in text_bas:
    labels = modelTex.encode_text(clip.tokenize(i).to(device)).float()
    labels = labels.squeeze()
    pre_text.append(labels)



#train_dataset = ImageFolder(root='./train', transform=transform)
#train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)



criterion = clip_loss()


correct = 0
total = 0

total_step = len(train_image)
for i in range(total_step):
    
    print("step: ",i+1,train_label[i])
    
    images=train_image[i].to(device)
    images = modelImg(images).float()
    images=images.squeeze()
    
    loss_tmp=[]
    total+=1
    for j in range(len(pre_text)):
        loss = criterion(images, pre_text[j])
        loss_tmp.append(loss)
        print(text_bas[j],loss)
    ptrdicted=""
    pos=0
    for j in range(len(pre_text)):
        if loss_tmp[j]<loss_tmp[pos]:
            pos=j
    predicted=text_bas[pos]
    correct+=(predicted == train_label[i])
    print("step: ",i+1,predicted)
    
accuracy = 100 * correct / total
print(correct,total)
print(f'Test Accuracy: {accuracy:.2f}%')
