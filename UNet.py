import torch
import torch.nn as nn
import time
import math
import torch.optim as optim

class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super(SingleConv, self).__init__()
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.single_conv(x)

class Encoder(nn.Module):
    def __init__(self, in_ch, list_ch):
        super(Encoder, self).__init__()
        self.encoder_1 = nn.Sequential(
            SingleConv(in_ch, list_ch[1], kernel_size=3, stride=1, padding=1),
        )
        self.encoder_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            SingleConv(list_ch[1], list_ch[2], kernel_size=3, stride=1, padding=1),
        )
        self.encoder_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            SingleConv(list_ch[2], list_ch[3], kernel_size=3, stride=1, padding=1),
        )
        self.encoder_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            SingleConv(list_ch[3], list_ch[4], kernel_size=3, stride=1, padding=1),
        )
        self.bottleneck = nn.Conv2d(
            in_channels=list_ch[4],
            out_channels=list_ch[4],
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )
        self.lastlayer= nn.Sequential(
            nn.BatchNorm2d(list_ch[4]),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        out_encoder_1 = self.encoder_1(x)
        out_encoder_2 = self.encoder_2(out_encoder_1)
        out_encoder_3 = self.encoder_3(out_encoder_2)
        out_encoder_4 = self.encoder_4(out_encoder_3)
        out_encoder_4 = self.bottleneck(out_encoder_4)
        out_encoder_4=self.lastlayer(out_encoder_4)
        return [out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4]

class Decoder(nn.Module):
    def __init__(self, out_ch, list_ch):
        super(Decoder, self).__init__()
        self.upconv_3_1 = nn.ConvTranspose2d(list_ch[4], list_ch[3], kernel_size=2, stride=2, bias=True)
        self.decoder_conv_3_1 = nn.Sequential(
            SingleConv(2 * list_ch[3], list_ch[3], kernel_size=3, stride=1, padding=1),
        )
        self.upconv_2_1 = nn.ConvTranspose2d(list_ch[3], list_ch[2], kernel_size=2, stride=2, bias=True)
        self.decoder_conv_2_1 = nn.Sequential(
            SingleConv(2 * list_ch[2], list_ch[2], kernel_size=3, stride=1, padding=1),
        )
        self.upconv_1_1 = nn.ConvTranspose2d(list_ch[2], list_ch[1], kernel_size=2, stride=2, bias=True)
        self.decoder_conv_1_1 = nn.Sequential(
            SingleConv(2 * list_ch[1], list_ch[1], kernel_size=3, stride=1, padding=1),
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(list_ch[1], out_ch, kernel_size=1, padding=0, bias=True)
        )

    def forward(self, out_encoder):
        out_encoder_1, out_encoder_2, out_encoder_3, out_encoder_4 = out_encoder
        out_decoder_3_1 = self.decoder_conv_3_1(
            torch.cat((self.upconv_3_1(out_encoder_4), out_encoder_3), dim=1))
        out_decoder_2_1 = self.decoder_conv_2_1(
            torch.cat((self.upconv_2_1(out_decoder_3_1), out_encoder_2), dim=1))
        out_decoder_1_1 = self.decoder_conv_1_1(
            torch.cat((self.upconv_1_1(out_decoder_2_1), out_encoder_1), dim=1))
        output = self.conv_out(out_decoder_1_1)
        return output

class Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()
        list_ch=[-1,2,4,8,16]
        self.encoder = Encoder(in_ch, list_ch)
        self.decoder = Decoder(out_ch, list_ch)
        self.sigmoid=nn.Sigmoid()
        self.initialize()

    @staticmethod
    def init_conv_deconv_BN(modules):
        for m in modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def initialize(self):
        print('# random init encoder weight using nn.init.kaiming_uniform !')
        self.init_conv_deconv_BN(self.decoder.modules)
        print('# random init decoder weight using nn.init.kaiming_uniform !')
        self.init_conv_deconv_BN(self.encoder.modules)

    def forward(self, x):
        out_encoder = self.encoder(x)
        out_decoder = self.decoder(out_encoder)
        out_decoder=self.sigmoid(out_decoder)
        return out_decoder
 
def main():
    start_time = time.time()
    block = Unet(in_ch=1, out_ch=1)
    print("Model initialization time:", time.time() - start_time)

    start_time = time.time()
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    block = block.to(device)
    input = torch.rand(2, 1,64,64).to(device)
    print("Model and data to GPU time:", time.time() - start_time)

    start_time = time.time()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(block.parameters(), lr=0.001)
    print("Loss function and optimizer initialization time:", time.time() - start_time)

    start_time = time.time()
    target = torch.rand_like(input).to(device)
    print("Target initialization time:", time.time() - start_time)

    start_time = time.time()
    output = block(input)
    print("Forward pass time:", time.time() - start_time)

    start_time = time.time()
    loss = criterion(output, target)
    print("Loss computation time:", time.time() - start_time)

    start_time = time.time()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Backward pass and optimization time:", time.time() - start_time)

    print("Output shape:", output.shape)
    print("Loss:", loss.item())

if __name__ == '__main__':
    main()