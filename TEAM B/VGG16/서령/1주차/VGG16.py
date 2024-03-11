# ref
# https://velog.io/@euisuk-chung/%ED%8C%8C%EC%9D%B4%ED%86%A0%EC%B9%98-%ED%8C%8C%EC%9D%B4%ED%86%A0%EC%B9%98%EB%A1%9C-CNN-%EB%AA%A8%EB%8D%B8%EC%9D%84-%EA%B5%AC%ED%98%84%ED%95%B4%EB%B3%B4%EC%9E%90-VGGNet%ED%8E%B8

# Architecture
# 3x3 합성곱 연산 x2 (채널 64)
# 3x3 합성곱 연산 x2 (채널 128)
# 3x3 합성곱 연산 x3 (채널 256)
# 3x3 합성곱 연산 x3 (채널 512)
# 3x3 합성곱 연산 x3 (채널 512)
# FC layer x3

import torch.nn as nn

def conv_2_block(in_dim, out_dim):
    model = nn.Sequential( # 모듈을 순차적으로 정의할 때 사용
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size =3 , padding=1), # 동일한 차원 수 출력 유지
        nn.ReLU(),
        nn.MaxPool2d(2,2) # pooling window, stride
    )
    return model

def conv_3_block(in_dim, out_dim):
    model = nn.Sequential( 
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
    )
    return model

class VGG(nn.Module): # nn.Module을 상속받음으로써 신경망을 구성하는 데 필요한 메서드와 속성 사용 가능 -> 사용자 정의 신경망
    def __init__(self, base_dim, num_classes=10):
        super(VGG, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(3, base_dim), # 입력 채널:RBG # base_dim:64
            conv_2_block(base_dim, 2*base_dim), # 128
            conv_3_block(2*base_dim, 4*base_dim), # 256
            conv_3_block(4*base_dim, 8*base_dim), # 512
            conv_3_block(8*base_dim, 8*base_dim) # 512
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(8*base_dim*1*1, 4096), # conv layer의 출력 특성 맵을 flatten한 벡터의 크기
            nn.ReLU(True), # inplace = True (입력 텐서를 바로 수정 -> 추가 메모리 할당X)
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000,num_classes) # 출력층의 뉴런 수를 CIFAR-10 클래스 수에 맞게 조정
        )
    
    def forward(self, x):
        x = self.feature(x) # conv layer들을 포함하고 있는 nn.Sequential 모듈을 의미
        x = x.view(x.size(0), -1) # 특성 맵을 fc layer에 적합한 1차원 벡터로 flatten
        x = self.fc_layer(x) 
        return x