import torch
import torch.nn as nn

class RPSNet(nn.Module):
    '''
    Rock-Paper-Scissors Network
    '''
    def __init__(self):
        super(RPSNet, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5, stride=2, padding=2)
        self.bn0 = nn.BatchNorm2d(5)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=5, stride=3, padding=3)
        self.bn1 = nn.BatchNorm2d(5)
        self.relu1 = nn.ReLU()
        self.avgpool2 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=5, stride=3, padding=3)
        self.bn2 = nn.BatchNorm2d(5)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(480, 3)
        self._initialize_weights()

    def forward(self, x_seq):  # input: batch_size, T, 250, 400
        x = torch.mean(x_seq, dim=1).unsqueeze(dim=1)
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.avgpool2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)