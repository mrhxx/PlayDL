import torch.nn as nn

class ResidualEncoderDecoder(nn.Module):
    def __init__(self):
        super(ResidualEncoderDecoder, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=10, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=10, stride=1, padding=0, bias=False)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=20, stride=1, padding=0, bias=False)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=20, stride=1, padding=0, bias=False)
        
        self.tconv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=20, stride=1, padding=0, bias=False)
        self.tconv2 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=20, stride=1, padding=0, bias=False)
        self.tconv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=10, stride=1, padding=0, bias=False)
        self.tconv4 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=10, stride=1, padding=0, bias=False)
        self.tconv5 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=5, stride=1, padding=0, bias=False)
        
        self.relu = nn.ReLU()
                
    def forward(self, x):
        # Encoder layers
        res_1 = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(x))
        res_2 = out
        out = self.relu(self.conv3(x))
        out = self.relu(self.conv4(x))
        res_3 = out
        out = self.relu(self.conv5(x))
        
        # Decoder layers
        out = self.tconv1(out)
        out += res_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out += res_2
        out = self.tconv4(self.relu(out))
        out = self.tconv5(self.relu(out))
        out += res_1
        out = self.relu(out)
        
        return out