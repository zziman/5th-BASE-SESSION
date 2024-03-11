import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from VGG16 import VGG

# Hyperparameters
batch_size = 100

# 데이터 전처리 정의
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])

# CIFAR10 data 로드
cifar10_test = datasets.CIFAR10(root="../Data/", train=False, transform = transform, target_transform=None, download=True)
# DataLoader 정의
test_loader = DataLoader(cifar10_test, batch_size = batch_size, shuffle=False)

# 디바이스 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 학습된 모델 불러오기
model = VGG(base_dim=64).to(device)
model.load_state_dict(torch.load('./vgg16_cifar10.pth'))

# Test
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for image, label in test_loader:
        x = image.to(device)
        y = label.to(device)

        output = model.forward(x)
        _, output_index = torch.max(output, 1) # 모델이 출력한 예측값 중 가장 높은 값을 가진 인덱스 저장
        # torch.max -> 최대값, 최대값 인덱스 반환

        total += label.size(0) 
        correct += (output_index == y).sum().float()

    print("Accuracy of Test Data : {}%".format(100*correct/total))
