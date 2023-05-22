import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SpectralNorm

if __name__ == '__main__':
    from layerskip import *
    from models import Cvi
else:
    from .layerskip import *
    from .models import Cvi


def get_norm(name, ch):
    if name == 'bn':
        return nn.BatchNorm2d(ch)
    elif name == 'in':
        return nn.InstanceNorm2d(ch)
    else:
        raise NotImplementedError('Normalization %s not implemented' % name)


class Tanh2(nn.Module):
    def __init__(self):
        super(Tanh2, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return (self.tanh(x) + 1) / 2


def get_activ(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'lrelu':
        return nn.LeakyReLU(0.2)
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'tanh2':
        return Tanh2()
    else:
        raise NotImplementedError('Activation %s not implemented' % name)


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc=None, kernel=3, stride=1, activ='lrelu', norm='bn', sn=False):
        super(ResidualBlock, self).__init__()

        if outc is None:
            outc = inc // stride

        self.activ = get_activ(activ)
        pad = kernel // 2
        if sn:
            self.input = SpectralNorm(nn.Conv2d(inc, outc, 1, 1, padding=0))
            self.blocks = nn.Sequential(SpectralNorm(nn.Conv2d(inc, outc, kernel, 1, pad)),
                                        get_norm(norm, outc),
                                        nn.LeakyReLU(0.2),
                                        SpectralNorm(nn.Conv2d(outc, outc, kernel, 1, pad)),
                                        get_norm(norm, outc))
        else:
            self.input = nn.Conv2d(inc, outc, 1, 1, padding=0)
            self.blocks = nn.Sequential(nn.Conv2d(inc, outc, kernel, 1, 1),  # kernel
                                        get_norm(norm, outc),
                                        nn.LeakyReLU(0.2),
                                        nn.Conv2d(outc, outc, kernel, 1, 1),
                                        get_norm(norm, outc))

    def forward(self, x):
        return self.activ(self.blocks(x) + self.input(x))


def ConvBlock(inc, outc, ks=3, s=1, p=0, activ='lrelu', norm='bn', res=0, resk=3, bn=True, sn=False):
    conv = nn.Conv2d(inc, outc, ks, s, p)
    if sn:
        conv = SpectralNorm(conv)
    blocks = [conv]
    if bn:
        blocks.append(get_norm(norm, outc))
    if activ is not None:
        blocks.append(get_activ(activ))
    for i in range(res):
        blocks.append(ResidualBlock(outc, activ=activ, kernel=resk, norm=norm, sn=sn))
    return nn.Sequential(*blocks)


class ConvBNReLU(nn.Module):
    '''Module for the Conv-BN-ReLU tuple.'''

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation=1, group=1,
                 has_bn=True, has_relu=True, mode='2d'):
        super(ConvBNReLU, self).__init__()
        self.has_bn = has_bn
        self.has_relu = has_relu
        if mode == '2d':
            self.conv = nn.Conv2d(
                c_in, c_out, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=False, groups=group)
            norm_layer = nn.BatchNorm2d
        elif mode == '1d':
            self.conv = nn.Conv1d(
                c_in, c_out, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=False, groups=group)
            norm_layer = nn.BatchNorm1d
        if self.has_bn:
            self.bn = norm_layer(c_out)
        if self.has_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x


class QCO_1d(nn.Module):
    def __init__(self, level_num):
        super(QCO_1d, self).__init__()
        self.conv1 = nn.Sequential(ConvBNReLU(256, 256, 3, 1, 1, has_relu=False), nn.LeakyReLU(inplace=True))
        self.conv2 = ConvBNReLU(256, 128, 1, 1, 0, has_bn=False, has_relu=False)
        self.f1 = nn.Sequential(ConvBNReLU(2, 64, 1, 1, 0, has_bn=False, has_relu=False, mode='1d'),
                                nn.LeakyReLU(inplace=True))
        self.f2 = ConvBNReLU(64, 128, 1, 1, 0, has_bn=False, mode='1d')
        self.out = ConvBNReLU(256, 128, 1, 1, 0, has_bn=True, mode='1d')
        self.level_num = level_num

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        N, C, H, W = x.shape
        x_ave = F.adaptive_avg_pool2d(x, (1, 1))
        cos_sim = (F.normalize(x_ave, dim=1) * F.normalize(x, dim=1)).sum(1)
        cos_sim = cos_sim.view(N, -1)
        cos_sim_min, _ = cos_sim.min(-1)
        cos_sim_min = cos_sim_min.unsqueeze(-1)
        cos_sim_max, _ = cos_sim.max(-1)
        cos_sim_max = cos_sim_max.unsqueeze(-1)
        q_levels = torch.arange(self.level_num).float().cuda()
        q_levels = q_levels.expand(N, self.level_num)
        q_levels = (2 * q_levels + 1) / (2 * self.level_num) * (cos_sim_max - cos_sim_min) + cos_sim_min
        q_levels = q_levels.unsqueeze(1)
        q_levels_inter = q_levels[:, :, 1] - q_levels[:, :, 0]
        q_levels_inter = q_levels_inter.unsqueeze(-1)
        cos_sim = cos_sim.unsqueeze(-1)
        quant = 1 - torch.abs(q_levels - cos_sim)
        quant = quant * (quant > (1 - q_levels_inter))
        sta = quant.sum(1)
        sta = sta / (sta.sum(-1).unsqueeze(-1))
        sta = sta.unsqueeze(1)
        sta = torch.cat([q_levels, sta], dim=1)
        sta = self.f1(sta)
        sta = self.f2(sta)
        x_ave = x_ave.squeeze(-1).squeeze(-1)
        x_ave = x_ave.expand(self.level_num, N, C).permute(1, 2, 0)
        sta = torch.cat([sta, x_ave], dim=1)
        sta = self.out(sta)
        return sta, quant


# Texture Enhance Module
class TEM(nn.Module):
    def __init__(self, level_num):
        super(TEM, self).__init__()
        self.level_num = level_num
        self.qco = QCO_1d(level_num)
        self.k = ConvBNReLU(128, 128, 1, 1, 0, has_bn=False, has_relu=False, mode='1d')
        self.q = ConvBNReLU(128, 128, 1, 1, 0, has_bn=False, has_relu=False, mode='1d')
        self.v = ConvBNReLU(128, 128, 1, 1, 0, has_bn=False, has_relu=False, mode='1d')
        self.out = ConvBNReLU(128, 256, 1, 1, 0, mode='1d')

    def forward(self, x):
        N, C, H, W = x.shape
        sta, quant = self.qco(x)
        k = self.k(sta)
        q = self.q(sta)
        v = self.v(sta)
        k = k.permute(0, 2, 1)
        w = torch.bmm(k, q)
        w = F.softmax(w, dim=-1)
        v = v.permute(0, 2, 1)
        f = torch.bmm(w, v)
        f = f.permute(0, 2, 1)
        f = self.out(f)
        quant = quant.permute(0, 2, 1)
        out = torch.bmm(f, quant)
        out = out.view(N, 256, H, W)
        return out


class DEModule(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv_start = ConvBNReLU(in_channel, 256, 1, 1, 0)
        self.tem = TEM(128)
        self.conv_end = ConvBNReLU(512, 192, 1, 1, 0)

    def forward(self, x):
        x = self.conv_start(x)
        x_tem = self.tem(x)
        x = torch.cat([x_tem, x], dim=1)
        x = self.conv_end(x)
        return x


class FCDenseNet(nn.Module):
    def __init__(self, in_channels=6, down_blocks=(5, 5, 5, 5, 5),
                 up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
                 growth_rate=16, out_chans_first_conv=48):
        super().__init__()
        self.down_blocks = down_blocks
        self.DEModule = DEModule((out_chans_first_conv + growth_rate * (down_blocks[0] + down_blocks[1])) * 3)
        self.up_blocks = up_blocks
        skip_connection_channel_counts = []

        ## First Convolution ##

        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels,
                                               out_channels=out_chans_first_conv, kernel_size=3,
                                               stride=1, padding=1, bias=True))
        self.add_module('DEModulefirstConv', nn.Conv2d(in_channels=in_channels,
                                                       out_channels=out_chans_first_conv + growth_rate * (
                                                                   down_blocks[0] + down_blocks[1]), kernel_size=3,
                                                       stride=2, padding=1, bias=True))
        cur_channels_count = out_chans_first_conv

        #####################
        # Downsampling path #
        #####################

        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i]))
            cur_channels_count += (growth_rate * down_blocks[i])
            if (i==1):
                skip_connection_channel_counts.insert(0, cur_channels_count + 96)  # 256为DEModule输出通道数512的一半
            else:
                skip_connection_channel_counts.insert(0, cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count))

        #####################
        #     Bottleneck    #
        #####################

        self.add_module('bottleneck', Bottleneck(cur_channels_count,
                                                 growth_rate, bottleneck_layers))
        prev_block_channels = growth_rate * bottleneck_layers
        cur_channels_count += prev_block_channels

        #######################
        #   Upsampling path   #
        #######################
        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks) - 1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i] * 3
            if (i == 3):
                cur_channels_count = 672
            self.denseBlocksUp.append(DenseBlock(
                cur_channels_count, growth_rate, up_blocks[i],
                upsample=True))
            prev_block_channels = growth_rate * up_blocks[i]
            cur_channels_count += prev_block_channels
        self.transUpBlocks.append(TransitionUp(
            prev_block_channels, prev_block_channels))
        cur_channels_count = 336  # prev_block_channels + skip_connection_channel_counts[-1]*2
        self.denseBlocksUp.append(DenseBlock(
            cur_channels_count, growth_rate, up_blocks[-1],
            upsample=False))
        cur_channels_count += growth_rate * up_blocks[-1]
        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
                                   out_channels=3, kernel_size=1, stride=1,
                                   padding=0, bias=True)
        self.Cv0 = Cvi(3, 96, kernel_size=3, stride=1, padding=1)
        self.Cv1 = Cvi(96, 144, before='LReLU', after='BN')
        self.Cv2 = Cvi(144, 192, before='LReLU', after='BN')
        self.Cv3 = Cvi(192, 240, before='LReLU', after='BN')
        self.Cv4 = Cvi(240, 288, before='LReLU', after='BN')
        self.att0 = nn.Sequential(ConvBlock(96 * 2, 96 * 2, 3, 1, 1, sn=True),
                                  ResidualBlock(96 * 2, 96 * 2, activ='sigmoid', sn=True))
        self.att1 = nn.Sequential(ConvBlock(144 * 2, 144 * 2, 3, 1, 1, sn=True),
                                  ResidualBlock(144 * 2, 144 * 2, activ='sigmoid', sn=True))
        self.att2 = nn.Sequential(ConvBlock(192 * 2, 192 * 2, 3, 1, 1, sn=True),
                                  ResidualBlock(192 * 2, 192 * 2, activ='sigmoid', sn=True))
        self.att3 = nn.Sequential(ConvBlock(240 * 2, 240 * 2, 3, 1, 1, sn=True),
                                  ResidualBlock(240 * 2, 240 * 2, activ='sigmoid', sn=True))
        self.att4 = nn.Sequential(ConvBlock(288 * 2, 288 * 2, 3, 1, 1, sn=True),
                                  ResidualBlock(288 * 2, 288 * 2, activ='sigmoid', sn=True))

    def forward(self, confuse_result, background, shadow_img, featureMaps):
        x = torch.cat([confuse_result, shadow_img], dim=1)
        background_feature = []
        back1 = self.Cv0(background)
        background_feature.append(back1)
        back2 = self.Cv1(back1)
        background_feature.append(back2)
        back3 = self.Cv2(back2)
        background_feature.append(back3)
        back4 = self.Cv3(back3)
        background_feature.append(back4)
        back5 = self.Cv4(back4)
        background_feature.append(back5)
        out = self.firstconv(x)
        DEModuleFirst = self.DEModulefirstConv(x)
        skip_connections = []
        newFeatureMap = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            background_featuremap = background_feature[i]
            skip = torch.cat((out, background_featuremap), 1)
            att = getattr(self, 'att{}'.format(i))(skip)
            skip = skip * att
            skip_connections.append(skip)
            newFeatureMap.append(out)
            out = self.transDownBlocks[i](out)
        DEModuleinput = torch.cat([DEModuleFirst, skip_connections[1]], dim=1)
        DEModuleresult = self.DEModule(DEModuleinput)
        skip_connections[1] = torch.cat([skip_connections[1], DEModuleresult], dim=1)
        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            featureMap = featureMaps.pop()
            out = self.transUpBlocks[i](out, skip, featureMap)
            out = self.denseBlocksUp[i](out)
        out = self.finalConv(out)
        return out, newFeatureMap


def BGShadowNet2(in_channels=6):
    return FCDenseNet(
        in_channels=in_channels, down_blocks=(4, 4, 4, 4, 4),
        up_blocks=(4, 4, 4, 4, 4), bottleneck_layers=12,
        growth_rate=12, out_chans_first_conv=48)
