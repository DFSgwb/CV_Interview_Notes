"""
    稠密连接:使每个层都可以直接访问前面层的特征图，从而减少梯度消失和特征信息的丢失并提高了模型的精度
"""
import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    # DenseLayer是DenseNet的基本层,它将输入和输出的特征图沿着通道数维度拼接在一起
    def __init__(self, input_size, growth_rate):
        super(DenseLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2(input_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_size, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )
    
    def forward(self, x):
        out = self.conv(x)
        out = torch.cat([x, out], dim=1)
        return out

class DenseBlock(nn.Module):
    # DenseBlock是由若干个DenseLayer组成的块
    def __init__(self, input_size, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = DenseLayer(input_size + i * growth_rate, growth_rate)
            self.layers.append(layer)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            return x

class TransitionLayer(nn.Module):
    # TransitionLayer则是用于减小特征图大小(宽高),同时还能减小特征图通道数的层
    # 其中compression是一个压缩因子,其表示过渡层输出的通道数相对于输入通道数的比例。
    def __init__(self, input_size, compression):
        super(TransitionLayer, self).__init__()
        output_size = int(input_size * compression)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(input_size),
            nn.Conv2d(input_size, output_size, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class DenseNet(nn.Module):
    # 最后,整个DenseNet由一个或多个DenseBlock和一个全连接层组成,可以根据具体需求自定义输入和输出通道数、DenseBlock数、DenseLayer数等超参数
    def __init__(self, num_classes, growth_rate=32, num_layers=[6, 12, 24, 16], compression=0.5):
        super(DenseNet, self).__init__()
        input_size = 64
        self.conv1 = nn.Conv2d(3, input_size, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(input_size)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dense1 = DenseBlock(input_size, growth_rate, num_layers[0])
        input_size += num_layers[0] * growth_rate
        output_size = int(input_size * compression)
        self.tran1 = TransitionLayer(input_size, compression)

        self.dense2 = DenseBlock(output_size, growth_rate, num_layers[1])
        input_size = output_size + num_layers[1] * growth_rate
        output_size = int(input_size * compression)
        self.tran2 = TransitionLayer(input_size, compression)

        self.dense3 = DenseBlock(output_size, growth_rate, num_layers[2])
        input_size = output_size + num_layers[2] * growth_rate
        output_size = int(input_size * compression)
        self.tran3 = TransitionLayer(input_size, compression)

        self.dense4 = DenseBlock(output_size, growth_rate, num_layers[3])
        input_size = output_size + num_layers[3] * growth_rate

        self.bn2 = nn.BatchNorm2d(input_size)
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pool(out)

        out = self.dense1(out)
        out = self.tran1(out)
        out = self.dense2(out)
        out = self.tran2(out)
        out = self.dense3(out)
        out = self.tran3(out)
        out = self.dense4(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = nn.functional.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out