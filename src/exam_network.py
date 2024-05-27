import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out

class Encoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=200, num_layers=5):
        super(Encoder, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(ConvBlock(in_channels, out_channels))
            in_channels = out_channels
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        feature_maps = []
        for layer in self.encoder:
            x = layer(x)
            feature_maps.append(x)
        return feature_maps

class Decoder(nn.Module):
    def __init__(self, in_channels=200, out_channels=1, num_layers=5):
        super(Decoder, self).__init__()
        layers = []
        for _ in range(num_layers - 1):
            layers.append(ConvBlock(in_channels, in_channels))
        layers.append(ConvBlock(in_channels, out_channels))
        self.decoder = nn.Sequential(*layers)

    def forward(self, feature_maps):
        x = feature_maps[-1]
        for i in range(len(self.decoder) - 1):
            x = self.decoder[i](x)
            x = x + feature_maps[-(i+2)]  # Adding residual connections
        x = self.decoder[-1](x)
        return x

class ResidualEncoderDecoder(nn.Module):
    def __init__(self):
        super(ResidualEncoderDecoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        feature_maps = self.encoder(x)
        out = self.decoder(feature_maps)
        return out

# Example usage
if __name__ == "__main__":
    model = ResidualEncoderDecoder()
    input_tensor = torch.randn(1, 1, 256, 256)  # Batch size of 1, 1 channel, 256x256 image
    output = model(input_tensor)
    print(output.shape)  # Should be [1, 1, 256, 256]

    # Calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {num_params}")
