import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from VGG16 import VGG

# Hyperparameters
batch_size = 100
learning_rate = 0.0002
num_epoch = 100

# 데이터 전처리 정의
transform = transforms.Compose( # 여러 transform 과정을 순서대로 적용
    [transforms.ToTensor(), # 이미지를 텐서로 변경
    # [0, 255] -> [0, 1] 범위로 변경됨 
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]) # RGB채널의 평균을 모두 0.5로 설정 # RGB채널의 표준편차를 모두 0.5로 설정 
    # [0, 1] -> [-1, 1] 범위로 변경됨

# CIFAR10 data 로드
cifar10_train = datasets.CIFAR10(root="../Data/", train=True, transform = transform, target_transform=None, download=True)
# DataLoader 정의
train_loader = DataLoader(cifar10_train, batch_size = batch_size, shuffle=True)

# 디바이스 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 모델 정의 -> 디바이스 할당
model = VGG(base_dim=64).to(device)
# 손실 함수 정의
loss_func = nn.CrossEntropyLoss()
# 최적화 함수 정의
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train
loss_arr = []
for i in range(num_epoch):
    for j, [image, label] in enumerate(train_loader):
        x = image.to(device)
        y_ = label.to(device)

        optimizer.zero_grad() # 기울기 초기화
        output = model.forward(x)
        loss = loss_func(output, y_)
        loss.backward()
        optimizer.step() # 파라미터 업데이트

    if i % 10 == 0:
        print(loss)
        loss_arr.append(loss.cpu().detach().numpy()) 

# 학습 후의 모델을 저장
torch.save(model.state_dict(), './vgg16_cifar10.pth')