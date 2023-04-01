
import torch
import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self, init_weights=False):
        super(VGG16, self).__init__()
        self.con1 = nn.Sequential(
            # 第 1 层
            nn.Conv2d(3, 64, 3, 1, 1),      # input [3, 32, 32], output [64, 32, 32]
            nn.BatchNorm2d(64),             # 批归一化操作，用于防止梯度消失或梯度爆炸，参数为卷积后输出的通道数
            nn.ReLU(inplace=True),
            # 第 2 层
            nn.Conv2d(64, 64, 3, 1, 1),     # input [64, 32, 32], output [64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)              # input [64, 32, 32], output [64, 16, 16]
        )
        self.con2 = nn.Sequential(
            # 第 3 层
            nn.Conv2d(64, 128, 3, 1, 1),    # input [64, 16, 16], output [128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 第 4 层
            nn.Conv2d(128, 128, 3, 1, 1),   # input [128, 16, 16], output [128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)              # input [128, 16, 16], output [128, 8, 8]
        )
        self.con3 = nn.Sequential(
            # 第 5 层
            nn.Conv2d(128, 256, 3, 1, 1),   # input [128, 8, 8], output [256, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 第 6 层
            nn.Conv2d(256, 256, 3, 1, 1),   # input [256, 8, 8], output [256, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 第 7 层
            nn.Conv2d(256, 256, 3, 1, 1),   # input [256, 8, 8], output [256, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)              # input [256, 8, 8], output [256, 4, 4]
        )
        self.con4 = nn.Sequential(
            # 第 8 层
            nn.Conv2d(256, 512, 3, 1, 1),   # input [256, 4, 4], output [512, 4, 4]
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 第 9 层
            nn.Conv2d(512, 512, 3, 1, 1),   # input [512, 4, 4], output [512, 4, 4]
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 第 10 层
            nn.Conv2d(512, 512, 3, 1, 1),   # input [512, 4, 4], output [512, 4, 4]
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)              # input [512, 4, 4], output [512, 2, 2]
        )
        self.con5 = nn.Sequential(
            # 第 11 层
            nn.Conv2d(512, 512, 3, 1, 1),   # input [512, 2, 2], output [512, 2, 2]
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 第 12 层
            nn.Conv2d(512, 512, 3, 1, 1),   # input [512, 2, 2], output [512, 2, 2]
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 第 13 层
            nn.Conv2d(512, 512, 3, 1, 1),   # input [512, 2, 2], output [512, 2, 2]
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)              # input [512, 2, 2], output [512, 1, 1]
        )
        self.features = nn.Sequential(
            self.con1,
            self.con2,
            self.con3,
            self.con4,
            self.con5
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 10),
            # nn.Softmax(dim=1)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, images):
        features = self.features(images)
        outputs = self.classifier(features.view(features.shape[0], -1))
        return outputs

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                # 使用正态分布对输入张量进行赋值
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Linear):
                # 使 layer.weight 值服从正态分布 N(mean, std)，默认值为 0，1。通常设置较小的值。
                nn.init.normal_(layer.weight, 0, 0.01)
                # 使 layer.bias 值为常数 val
                nn.init.constant_(layer.bias, 0)


# # 测试
# images = torch.rand([64, 3, 32, 32])
# model = VGG16()
# outputs = model(images)
# print(outputs.shape)
