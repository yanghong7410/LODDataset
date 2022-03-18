import torch
from torch import nn


class ChannelAttention(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(ChannelAttention, self).__init__()
        self.globalAveragePooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.encodeConv = nn.Conv2d(in_channel, out_channel,
                                    kernel_size=1, stride=1,
                                    padding=0, bias=True)
        self.decodeConv = nn.Conv2d(out_channel, in_channel,
                                    kernel_size=1, stride=1,
                                    padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.globalAveragePooling(x)
        x = self.encodeConv(x)
        x = self.relu(x)
        x = self.decodeConv(x)
        x = torch.sigmoid(x)  # torch.Size([8, 64, 1, 1])
        return x


class PixelAttention(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PixelAttention, self).__init__()
        self.encodeConv = nn.Conv2d(in_channel, out_channel,
                                    kernel_size=3, stride=1,
                                    padding=1, bias=True)
        self.decodeConv = nn.Conv2d(out_channel, in_channel,
                                    kernel_size=3, stride=1,
                                    padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.encodeConv(x)
        x = self.relu(x)
        x = self.decodeConv(x)
        x = torch.sigmoid(x)  # torch.Size([8, 64, 1, 1])
        return x


if __name__ == "__main__":
    input = torch.randn(8, 64, 112, 208)
    model = PixelAttention(64, 16)
    output = model(input)  # 8*64*1*1

    print(output, output.size())
    print((output * input).size())
