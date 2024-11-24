#
import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#把高度和宽度变为1
        self.cv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio,1,1),#(2,1024/16)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels,1,1)#(1024/16,1024)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #x = x.to(device)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c,1,1)  #x(2,1024,64,64)经池化为(2,1024,1,1)再转为(2,1024)
        #print(y.shape)
        y = self.cv(y).view(b, c, 1, 1)#(2,1024)->(2,1024,1,1)

        return x * self.sigmoid(y)


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]#[0]是从三个通道中选取最大值，[1]是这个最大值是属于哪个通道地
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        y = torch.cat([max_pool, avg_pool], dim=1)  #把两个单通道(2,1,1024,1024)拼接成(2,2,1024,1024)
        y = self.conv(y) #(2,2,1024,1024)->(2,1,1024,1024)
        return x * self.sigmoid(y) #(2,1,1024,1024)*(2,3,1024,1024)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()
        self.BN = nn.BatchNorm2d(in_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        #x=x.to(device)
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        x = self.act(self.BN(x))
        return x
if __name__ == "__main__":

    input = torch.rand(2,1024,64,64)
    A = SpatialAttention()
    B = ChannelAttention(input.shape[1], reduction_ratio=16)
    output_A = A(input)
    output_B = B(input)
    print(output_A.shape)
    print(output_B.shape)



    C = CBAM(input.shape[1],16)
    output_C = C(input)
    print(output_C.shape)