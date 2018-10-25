import torch
import torch.nn.functional as F
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        if inplanes!= planes:
            self.conv0 = conv3x3(inplanes,planes)

        self.inplanes = inplanes
        self.planes = planes

        self.conv1 = conv3x3(planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        if self.inplanes != self.planes:
            x = self.conv0(x)

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Initial_LadderBlock(nn.Module):

    def __init__(self,planes,layers,kernel=3,block=BasicBlock,inplanes = 3):
        super().__init__()
        self.planes = planes
        self.layers = layers
        self.kernel = kernel

        self.padding = int((kernel-1)/2)
        self.inconv = nn.Conv2d(inplanes,planes, kernel_size=3, stride=1, padding=1)

        self.pool2d = nn.AvgPool2d(kernel_size=2,stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)

        # create module list for down branch
        self.down_module_list = nn.ModuleList()
        self.down_module_list.append(block(planes,planes))
        for i in range(1,layers):
            self.down_module_list.append(block(planes*(2**(i-1)),planes*(2**i)))

        # create module for bottom block
        self.bottom = block(planes*(2**(layers-1)),planes*(2**layers))

        # create module list for up branch
        self.up_module_list = nn.ModuleList()
        for i in range(0, layers):
            self.up_module_list.append(block(planes*(2**(i+1)),planes*(2**i)))

        # create module list for lateral branch
        self.lateral_module_list = nn.ModuleList()
        for i in range(0,layers):
            self.lateral_module_list.append(nn.Conv2d(planes*(2**i),planes*(2**(i+1)), kernel_size=1, stride=1, padding=0))

        # create module list for out lateral branch
        self.out_lateral_module_list = nn.ModuleList()
        self.out_lateral_module_list.append(
            nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0))

        for i in range(1, layers+1):
            self.out_lateral_module_list.append(
                nn.Conv2d(planes * (2 ** i), planes * (2 ** (i - 1)), kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        tmp = self.inconv(x)

        down_out = []
        down_out.append(self.down_module_list[0](tmp))

        lateral_out = []
        lateral_out.append(self.lateral_module_list[0](down_out[-1]))

        # down branch
        for i in range(1,self.layers):
            tmp = self.pool2d(down_out[-1])
            tmp = self.down_module_list[i](tmp)
            down_out.append(tmp)

            lateral_out.append(self.lateral_module_list[i](down_out[-1]))

        # bottom branch
        tmp = self.pool2d(down_out[-1])
        bottom = self.bottom(tmp)

        # up branch
        up_out = []
        up_out.append(bottom)

        for i in reversed(range(self.layers)):
            tmp = up_out[-1]
            tmp = self.upsample(tmp)
            tmp = self.up_module_list[i](tmp)
            up_out.append(tmp)

        # out lateral branch
        out = []
        for i in range(0,self.layers+1):
            out.append( self.out_lateral_module_list[i] (up_out[self.layers-i]) )

        return out

class Middle_LadderBlock(nn.Module):

    def __init__(self,planes,layers,kernel=3,block=BasicBlock,inplanes = 3):
        super().__init__()
        self.planes = planes
        self.layers = layers
        self.kernel = kernel

        self.padding = int((kernel-1)/2)

        self.pool2d = nn.AvgPool2d(kernel_size=2,stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)

        # create module list for down branch
        self.down_module_list = nn.ModuleList()
        self.down_module_list.append(block(planes,planes))
        for i in range(1,layers):
            self.down_module_list.append(block(planes*(2**(i-1)),planes*(2**i)))

        # create module for bottom block
        self.bottom = block(planes*(2**(layers-1)),planes*(2**layers))

        # create module list for up branch
        self.up_module_list = nn.ModuleList()
        for i in range(0, layers):
            self.up_module_list.append(block(planes*(2**(i+1)),planes*(2**i)))

        # create module list for lateral branch
        self.lateral_module_list = nn.ModuleList()
        for i in range(0,layers):
            self.lateral_module_list.append(nn.Conv2d(planes*(2**i),planes*(2**(i+1)), kernel_size=1, stride=1, padding=0))

        # create module list for out lateral branch
        self.out_lateral_module_list = nn.ModuleList()
        self.out_lateral_module_list.append(
            nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0))

        for i in range(1, layers+1):
            self.out_lateral_module_list.append(
                nn.Conv2d(planes * (2 ** i), planes * (2 ** (i - 1)), kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        assert len(x) == (self.layers + 1 )

        down_out = []
        down_out.append(self.down_module_list[0](x[0]))

        lateral_out = []
        lateral_out.append(self.lateral_module_list[0](down_out[-1]))

        # down branch
        for i in range(1,self.layers):
            tmp = self.pool2d(down_out[-1])
            tmp = self.down_module_list[i](tmp + x[i])
            down_out.append(tmp)

            lateral_out.append(self.lateral_module_list[i](down_out[-1]))

        # bottom branch
        tmp = self.pool2d(down_out[-1])
        bottom = self.bottom(tmp)

        # up branch
        up_out = []
        up_out.append(bottom)

        for i in reversed(range(self.layers)):
            tmp = up_out[-1]
            tmp = self.upsample(tmp)
            tmp = self.up_module_list[i](tmp)
            up_out.append(tmp)

        # out lateral branch
        out = []
        for i in range(0,self.layers+1):
            out.append( self.out_lateral_module_list[i] (up_out[self.layers-i]) )

        return out


class Final_LadderBlock(nn.Module):

    def __init__(self,planes,layers,kernel=3,block=BasicBlock,inplanes = 3,num_classes=2):
        super().__init__()
        self.planes = planes
        self.layers = layers
        self.kernel = kernel

        self.padding = int((kernel-1)/2)

        self.pool2d = nn.AvgPool2d(kernel_size=2,stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)

        # create module list for down branch
        self.down_module_list = nn.ModuleList()
        self.down_module_list.append(block(planes,planes))
        for i in range(1,layers):
            self.down_module_list.append(block(planes*(2**(i-1)),planes*(2**i)))

        # create module for bottom block
        self.bottom = block(planes*(2**(layers-1)),planes*(2**layers))

        # create module list for up branch
        self.up_module_list = nn.ModuleList()
        for i in range(0, layers):
            self.up_module_list.append(block(planes*(2**(i+1)),planes*(2**i)))

        # create module list for lateral branch
        self.lateral_module_list =nn.ModuleList()
        for i in range(0,layers):
            self.lateral_module_list.append(nn.Conv2d(planes*(2**i),planes*(2**(i+1)), kernel_size=1, stride=1, padding=0))

        self.final = nn.Conv2d(planes,num_classes,kernel_size=1,stride=1)

    def forward(self, x):
        assert len(x) == (self.layers + 1 )

        down_out = []
        down_out.append(self.down_module_list[0](x[0]))

        lateral_out = []
        lateral_out.append(self.lateral_module_list[0](down_out[-1]))

        # down branch
        for i in range(1,self.layers):
            tmp = self.pool2d(down_out[-1])
            tmp = self.down_module_list[i](tmp + x[i])
            down_out.append(tmp)

            lateral_out.append(self.lateral_module_list[i](down_out[-1]))

        # bottom branch
        tmp = self.pool2d(down_out[-1])
        bottom = self.bottom(tmp)

        # up branch
        up_out = []
        up_out.append(bottom)

        for i in reversed(range(self.layers)):
            tmp = up_out[-1]
            tmp = self.upsample(tmp)
            tmp = self.up_module_list[i](tmp)
            up_out.append(tmp)

        out = self.final(up_out[-1])
        out = F.log_softmax(out,dim=1)

        return out

class LadderNet(nn.Module):
    def __init__(self,layers=3,filters=16,num_classes=2,inplanes=3):
        super().__init__()
        self.initial_block = Initial_LadderBlock(planes=filters,layers=layers,inplanes=inplanes)
        self.final_block = Final_LadderBlock(planes=filters,layers=layers,num_classes=num_classes)

    def forward(self,x):
        out = self.initial_block(x)
        out = self.final_block(out)
        return out