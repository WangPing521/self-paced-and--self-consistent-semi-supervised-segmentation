import torch
from torch import nn


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet(nn.Module):
    def __init__(self, input_dim=3, num_classes=1):
        super(UNet, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch=input_dim, out_ch=64)
        self.Conv2 = conv_block(in_ch=64, out_ch=128)
        self.Conv3 = conv_block(in_ch=128, out_ch=256)
        self.Conv4 = conv_block(in_ch=256, out_ch=512)
        self.Conv5 = conv_block(in_ch=512, out_ch=1024)

        self.Up5 = up_conv(in_ch=1024, out_ch=512)
        self.Up_conv5 = conv_block(in_ch=1024, out_ch=512)

        self.Up4 = up_conv(in_ch=512, out_ch=256)
        self.Up_conv4 = conv_block(in_ch=512, out_ch=256)

        self.Up3 = up_conv(in_ch=256, out_ch=128)
        self.Up_conv3 = conv_block(in_ch=256, out_ch=128)

        self.Up2 = up_conv(in_ch=128, out_ch=64)
        self.Up_conv2 = conv_block(in_ch=128, out_ch=64)

        self.Conv_1x1 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)
        self.Confident = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x, return_confident=False):
        # encoding path
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        # decoding + concat path
        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        confident = torch.sigmoid(self.Confident(d2))
        if return_confident:
            return d1, confident
        return d1

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count
if __name__ == '__main__':
    print('#### Test Case ###')
    from torch.autograd import Variable
    x = Variable(torch.rand(1, 3, 256, 256))
    model = UNet(3,2)
    param = count_param(model)
    y = model(x)
    print('Output shape:',y.shape)
    print('UNet totoal parameters: %.2fM (%d)'%(param/1e6,param))