import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pennylane as qml

import math
import torch.optim as optim
import torch.distributed as dist
import os
import time

NUM_QUBIT = 4
dev = qml.device("default.qubit", wires=NUM_QUBIT)

@qml.qnode(dev, interface="torch")
def qnode(inputs, weight_0, weight_1):
    qml.AmplitudeEmbedding(features=inputs, wires=range(NUM_QUBIT), normalize=True)
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)
    qml.Hadamard(wires=2)
    qml.Hadamard(wires=3)
    qml.RY(weight_0[0], wires=0)
    qml.RY(weight_0[1], wires=1)
    qml.RY(weight_0[0], wires=2)
    qml.RY(weight_0[1], wires=3)
    qml.RZ(weight_1[0], wires=0)
    qml.RZ(weight_1[1], wires=1)
    qml.RZ(weight_1[0], wires=2)
    qml.RZ(weight_1[1], wires=3)

    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 0])
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(NUM_QUBIT)]


class QuantumNNLayer(nn.Module):
    def __init__(self,ch_in,f):
        super(QuantumNNLayer, self).__init__()
        self.f=f
        self.n_wqubits =f**2
        self.ch_in = ch_in
        self.qfc1_1 = qml.qnn.TorchLayer(
            qnode,
            {"weight_0": 2, "weight_1": 2},
        )
        with torch.no_grad():
            for param in self.qfc1_1.parameters():
                torch.nn.init.zeros_(param)
    def forward(self, input):
        padding = 1
        if padding > 0:
            input = F.pad(input, (1, 2, 1, 2, 0, 0, 0, 0))
        bs, in_channel, input_h, input_w = input.shape
        stride = 1
        output_h = (math.floor((input_h - self.f) / stride) + 1)
        output_w = (math.floor((input_w - self.f) / stride) + 1)
        out_channel=1
        start_time = time.time()
        regions = input.unfold(2, self.f, stride).unfold(3, self.f, stride)
        regions = regions.contiguous().view(bs, in_channel, -1, self.f * self.f)
        regions = regions.permute(0, 2, 1, 3).reshape(-1, in_channel * self.f * self.f)
        
        all_regions=regions
        stime=time.time()
        out=self.qfc1_1(all_regions)
        
        out=out.mean(dim=1)
        out=(out + 1) / 2 
        output=out.reshape(bs,out_channel,output_h, output_w)       
        return output
    
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

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x

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
        self.ch=list_ch[4]
        for i in range(1, self.ch+1): 
            setattr(self, f'Q{i}', QuantumNNLayer(1, 4)) 
        self.lastlayer= nn.Sequential(
            nn.BatchNorm2d(self.ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out_encoder_1 = self.encoder_1(x)
        out_encoder_2 = self.encoder_2(out_encoder_1)
        out_encoder_3 = self.encoder_3(out_encoder_2)
        out_encoder_4 = self.encoder_4(out_encoder_3)
        en4=out_encoder_4
        device = out_encoder_3.device
        x=torch.split(out_encoder_4, 1, dim=1)
        outputs = []
        for i in range(1, self.ch+1):
            q_layer = getattr(self, f'Q{i}')
            x_i = q_layer(x[i - 1])
            outputs.append(x_i)
        out_encoder4 = torch.cat(outputs, dim=1)
        out_encoder4=out_encoder4.to(device)
        out_encoder4=self.lastlayer(out_encoder4)
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
        self.init_conv_deconv_BN(self.decoder.modules)
        self.init_conv_deconv_BN(self.encoder.modules)

    def forward(self, x):
        out_encoder = self.encoder(x)
        out_decoder = self.decoder(out_encoder)
        out_decoder=self.sigmoid(out_decoder)
        return out_decoder
 
def main():
    initime=time.time()
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
    print("total time:", time.time() - initime)

if __name__ == '__main__':
    main()