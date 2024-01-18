from operations import *
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models


def E2E(input):
    return torch.add(input, input.permute(0, 1, 3, 2))


def E2N(input):
    return input.permute(0, 1, 3, 2) * 2


class BrainNetCNN(nn.Module):

    def __init__(self, atlas_size, drop1, drop2):
        super(BrainNetCNN, self).__init__()
        self.layers_E2E = OPS['base_layer'](1, 32, (atlas_size, 1))
        self.layers_E2E2 = OPS['base_layer'](32, 32, (atlas_size, 1))
        self.layers_E2N = OPS['base_layer'](32, 64, (atlas_size, 1))
        self.layers_N2G = OPS['base_layer'](64, 256, (atlas_size, 1))
        self.layers5 = OPS['Fully_connected'](256, 128)
        self.layers6 = OPS['Fully_connected'](128, 30)
        self.layers7 = OPS['Fully_connected'](30, 2)
        self.softmax = nn.Softmax(dim=1)
        self.dropout1 = nn.Dropout(p=drop1)
        self.dropout2 = nn.Dropout(p=drop2)

    def forward(self, x):
        layer1 = E2E(self.layers_E2E(x))
        layer1 = self.dropout1(layer1)
        layer2 = E2E(self.layers_E2E2(layer1))
        layer2 = self.dropout1(layer2)
        layer3 = E2N(self.layers_E2N(layer2))
        layer3 = self.dropout1(layer3)
        layer4 = self.layers_N2G(layer3)
        layer4 = self.dropout1(layer4)
        layer5 = self.layers5(layer4.view(-1, 256))
        layer5 = self.dropout2(layer5)
        layer6 = self.layers6(layer5)
        layer6 = self.dropout2(layer6)
        layer7 = self.layers7(layer6)
        out = self.softmax(layer7)
        return out


class DNN(nn.Module):
    def __init__(self, atlas_size):
        super(DNN, self).__init__()
        if atlas_size == 90:
            self.a = 30976
        else:
            self.a = 160000
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(self.a, 1024)
        self.fc2 = nn.Linear(1024, 2)
        self.n = atlas_size

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, self.a)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.dropout = nn.Dropout(p=0.1)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.dropout(x)
        x = self.layer4(x)
        x = self.dropout(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


Resmodel = ResNet(BasicBlock, [2, 2, 2, 2])


class Autoencoder(nn.Module):
    def __init__(self, n):
        super(Autoencoder, self).__init__()
        self.n = n
        self.encoder = nn.Sequential(
            nn.Linear(n * n, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, n * n),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        enx = self.encoder(x)
        dex = self.decoder(enx)
        outde = dex.view(dex.size(0), 1, self.n, self.n)
        classification = self.classifier(enx)
        return outde, classification
