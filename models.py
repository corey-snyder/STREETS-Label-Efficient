import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__() 
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2)) 
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 256)
        self.up1 = up(512, 128)
        self.up2 = up(256, 64)
        self.up3 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return torch.sigmoid(x)

class VariableUNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_stages):
        super(VariableUNet, self).__init__()
        self.L = 64 # base width
        self.inc = inconv(n_channels, self.L)
        self.n_stages = n_stages
        # downsampling stages
        encoder_input_widths = [self.L*2**n for n in range(n_stages)]
        encoder_output_widths = [self.L*2**(n+1) for n in range(n_stages)]
        encoder = []
        for n in range(n_stages):
            encoder.append(down(encoder_input_widths[n], encoder_output_widths[n]))

        self.encoder = nn.Sequential(*encoder)
        encoder_output_widths = [self.L]+encoder_output_widths # include input convolution self.inc
        # upsampling stages
        decoder = []
        decoder_input_widths = [encoder_output_widths[-1]+encoder_output_widths[-2]]
        for n in range(1, n_stages):
            decoder_input_widths.append(int(decoder_input_widths[n-1]/2)+encoder_output_widths[n_stages-n-1])
        decoder_output_widths = [int(decoder_input_widths[n]/2) for n in range(n_stages-1)]+[self.L]
        for n in range(n_stages):
            decoder.append(up(decoder_input_widths[n], decoder_output_widths[n])) 
        self.decoder = nn.Sequential(*decoder)
        self.outc = outconv(self.L, n_classes)

    def forward(self, x):
        encoder_outputs = [self.inc(x)]
        # encode
        # total of n_stages+1 encoder outputs
        for n in range(len(self.encoder)):
            encoder_outputs.append(self.encoder[n](encoder_outputs[n]))

        # decode
        x = self.decoder[0](encoder_outputs[-1], encoder_outputs[-2])
        for n in range(1, len(self.decoder)):
            x = self.decoder[n](x, encoder_outputs[self.n_stages-n-1])
        x = self.outc(x)
        return torch.sigmoid(x)
