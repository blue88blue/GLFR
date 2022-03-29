from base.segbase import SegBaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation, use_bn=True, use_relu=True, coord=False):
        super(ConvBNReLU, self).__init__()
        self.coord = coord
        if coord:
            c_in += 2

        self.conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)

        if use_bn:
            self.bn = nn.BatchNorm2d(c_out)
        else:
            self.bn = None
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        if self.coord:
            x = self.concat_grid(x)  

        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def concat_grid(self, input):
        b, c, out_h, out_w = input.size()
        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2).permute(2, 0, 1)  #(2, h, w)
        grid = grid.repeat(b, 1, 1, 1).type_as(input).to(input.device)
        input = torch.cat((grid, input), dim=1)
        return input


class GFR(nn.Module):
    def __init__(self, channels, inter_channel, n_class, base=8, aux=False, coord=False):
        super().__init__()
        self.aux = aux
        self.n_class = n_class
        self.base = base*n_class 
        self.allBase = self.base*len(channels)
        self.inter_channel = inter_channel
        self.channels = channels
        # reduce
        self.conv_reduce = nn.ModuleList()
        for c in channels:
            self.conv_reduce.append(ConvBNReLU(c, inter_channel, kernel_size=3, stride=1, padding=1, dilation=1, coord=coord))
        # map
        self.conv_project = nn.ModuleList()
        for c in channels:
            self.conv_project.append(ConvBNReLU(c, self.base, kernel_size=1, stride=1, padding=0, dilation=1, use_bn=False, use_relu=False, coord=coord))
        # remap
        self.conv_reproject = nn.ModuleList()
        for i, c in enumerate(channels):
            self.conv_reproject.append(ConvBNReLU(c, (i+1)*self.base, kernel_size=1, stride=1, padding=0, dilation=1, use_bn=False, use_relu=False, coord=coord))

        self.conv_out = nn.ModuleList()
        for c in channels:
            self.conv_out.append(ConvBNReLU(inter_channel, c, kernel_size=3, stride=1, padding=1, dilation=1, coord=coord))
    
    def forward(self, features):
        assert len(features) == len(self.channels)
        b = features[0].size()[0]

        feats_reduce = []
        for i, feat in enumerate(features):
            feats_reduce.append(self.conv_reduce[i](feat).view(b, self.inter_channel, -1))  #(b, c_, hw)

        allBases = []
        auxs = []
        for i, feat in enumerate(features):
            h, w = feat.size()[-2:]
            attMap = self.conv_project[i](feat)  #(b, k, hw)
            ###############
            if self.aux:
                aux = attMap.view(b, self.n_class, -1, h, w).mean(dim=2)
                auxs.append(aux)
            else:
                auxs.append(attMap.view(b, self.n_class, -1, h, w))
            ############### 
            attMap = F.softmax(attMap.view(b, self.base, -1), dim=-1).permute(0, 2, 1)  #(b, hw, k)
            base = torch.bmm(feats_reduce[i], attMap)  # (b, c_, k)
            allBases.append(base)

        CorssBase = torch.cat(allBases, dim=2) #(b, c_, KK) 
        
        feats_rebuild = []
        for i, feat in enumerate(features):
            h, w = feat.size()[-2:]
            reproject_W = self.conv_reproject[i](feat).view(b, -1, h*w)  # (b, KK, hw)
            reproject_W = F.softmax(reproject_W, dim=1)
            x_rebuild = torch.bmm(CorssBase[..., :(i+1)*self.base], reproject_W).view(b, self.inter_channel, h, w)  # (b, c_, h, w) 
            feats_rebuild.append(x_rebuild)
     
        out = []
        for i, feat in enumerate(features):
            out.append(self.conv_out[i](feats_rebuild[i]) + feat)  # prevent network degradation, use concat or add
        return out, auxs



class LFR(nn.Module):
    def __init__(self,  channel_high, channel_low, out_channel, c_mid=64, scale=2, k_up=5, k_enc=3):
        super().__init__()
        self.scale = scale
        self.inter_channel = max(channel_high//4, 32)

        self.conv_reduce = ConvBNReLU(channel_high, self.inter_channel, kernel_size=3, stride=1, padding=1, dilation=1)

        self.conv1 = ConvBNReLU(channel_high, c_mid, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv2 = ConvBNReLU(channel_low, c_mid, kernel_size=1, stride=1, padding=0, dilation=1)
        self.enc = ConvBNReLU(c_mid*2, k_up**2, kernel_size=k_enc, stride=1, padding=k_enc//2, dilation=1, use_relu=False)

        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale, padding=k_up//2*scale)

        self.conv_out = nn.Sequential(
            nn.Conv2d(self.inter_channel+channel_low , out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x_h, x_l):
        b, c, h, w = x_h.size()
        h_, w_ = h * self.scale, w * self.scale
        
        x1 = F.interpolate(self.conv1(x_h), size=(h_, w_), mode="bilinear", align_corners=True)    # b * m * h * w
        x2 = self.conv2(x_l)
        W = self.enc(torch.cat((x1, x2), dim=1))                                 # b * 100 * h * w
        W = F.softmax(W, dim=1)                         # b * 25 * h_ * w_

        x_h_reduce = self.conv_reduce(x_h)  

        X = self.unfold(F.interpolate(x_h_reduce, size=(h_, w_), mode="bilinear", align_corners=True))  # b * 25c * h_ * w_
        X = X.view(b, self.inter_channel, -1, h_, w_)                    # b * c * 25 * h_ * w_
        X = torch.mul(W.unsqueeze(1), X).sum(dim=2)

        return self.conv_out(torch.cat((X, x_l), dim=1))



class GLFRNet(SegBaseModel):
    def __init__(self, n_class, backbone='resnet34', aux=True, pretrained_base=True, mid_channel=320, c_mid = 256, base=8, **kwargs):
        super(GLFRNet, self).__init__(backbone, pretrained_base=pretrained_base)
        self.aux = aux
        channels = self.base_channel

        self.donv_up1 = LFR(channels[3], channels[2], channels[2], c_mid=c_mid)
        self.donv_up2 = LFR(channels[2], channels[1], channels[1], c_mid=c_mid)
        self.donv_up3 = LFR(channels[1], channels[0], channels[0], c_mid=c_mid)
        self.donv_up4 = LFR(channels[0], self.conv1_channel, self.conv1_channel, c_mid=c_mid)

        self.out_conv = nn.Sequential(
            nn.Conv2d(self.conv1_channel, n_class, kernel_size=1, bias=False),
        )

        channels.reverse()
        self.gr = GFR(channels, mid_channel, n_class, base=base, aux=aux, coord=False)

    def forward(self, x):

        outputs = dict()
        size = x.size()[2:]
        c1, c2, c3, c4, c5 = self.backbone.extract_features(x)

        out, auxs = self.gr([c5, c4, c3, c2])
        c5, c4, c3, c2 = out
        aux_out = []
        if self.aux:
            for a in auxs:
                a = F.interpolate(a, size, mode='bilinear', align_corners=True)
                aux_out.append(a)
            outputs.update({"aux_out": aux_out})

        x1 = self.donv_up1(c5, c4)
        x2 = self.donv_up2(x1, c3)
        x3 = self.donv_up3(x2, c2)
        x4 = self.donv_up4(x3, c1)
        x = self.out_conv(x4)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)  

        outputs.update({"main_out": x})
        return outputs


if __name__ == '__main__':
    import torch
    model = GLFRNet(n_class=2, backbone="resnet34")
    img = torch.randn(4, 3, 224, 224)
    pred = model(img)
    print(pred["main_out"].size())
