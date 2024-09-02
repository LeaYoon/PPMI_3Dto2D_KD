
from torch import nn



class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3):
        super(SimpleCNN, self).__init__()

        self.conv3d_1 = nn.Conv3d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3d_2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv3d_3 = nn.Conv3d(in_channels=128, out_channels=192, kernel_size=3, stride=2, padding=1)
        self.conv3d_4 = nn.Conv3d(in_channels=192, out_channels=256, kernel_size=3, stride=2, padding=1)

        # self.bn_1 = nn.BatchNorm2d(num_features=64)
        # self.bn_2 = nn.BatchNorm2d(num_features=128)
        # self.bn_3 = nn.BatchNorm2d(num_features=192)
        # self.bn_4 = nn.BatchNorm2d(num_features=256)
        
        self.maxpool3d = nn.MaxPool3d(kernel_size=2, stride=2)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)
        self.fc = nn.Linear(in_features=256, out_features=2)
        self.softmax = nn.Softmax(dim=1)
        return

    def forward(self, x):
        x = self.conv3d_1(x)
        x = self.leakyrelu(x)
        x = self.maxpool3d(x)
        
        x = self.conv3d_2(x)
        x = self.leakyrelu(x)
        x = self.maxpool3d(x)
        
        x = self.conv3d_3(x)
        x = self.leakyrelu(x)
        x = self.maxpool3d(x)
        
        x = self.conv3d_4(x)
        x = self.leakyrelu(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        out = self.softmax(x)
        return out
    

class PDNet(nn.Module):
    def __init__(self, in_channels=3):
        super(PDNet, self).__init__()
        
        # original input size is 91*109*91 -> zero padding as 109*109*109
        # rescaled by the range [0, 255]
        # mean subtraction

        self.conv3d_1 = nn.Conv3d(in_channels=in_channels, out_channels=16, kernel_size=7, stride=4, padding=0)
        self.conv3d_2 = nn.Conv3d(in_channels=16, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.conv3d_3 = nn.Conv3d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=0)
        
        self.maxpool3d = nn.MaxPool3d(kernel_size=3, stride=2)
        self.maxpool3d = nn.MaxPool3d(kernel_size=3, stride=2)
        
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_features=256, out_features=2)
        self.softmax = nn.Softmax(dim=1)
        return

    def forward(self, x):
        x = self.conv3d_1(x)
        x = self.relu(x)
        x = self.maxpool3d(x)
        
        x = self.conv3d_2(x)
        x = self.relu(x)
        x = self.maxpool3d(x)
        
        x = self.conv3d_3(x)
        x = self.relu(x)
        
        x = x.view(-1, 256)
        x = self.fc(x)
        out = self.softmax(x)
        return out

