import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetInspired(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetInspired, self).__init__()
        self.in_planes = 64
        self.convblock1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False, groups=32),
            nn.Conv2d(32, 32, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.convblock3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # self.convblock5 = nn.Sequential(
        #     nn.Conv2d(128, 3, 3, stride=1, padding=1, bias=False),
        # )
        self.convblock4 = nn.Sequential(
            nn.Conv2d(128, 1, 3, stride=1, padding=1, bias=False),
        )

        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, 1, stride=1)
        self.layer2 = self._make_layer(block, 128, 1, stride=2)
        self.layer3 = self._make_layer(block, 256, 1, stride=2)
        self.layer4 = self._make_layer(block, 512, 1, stride=2)
        self.layer5 = self._make_layer(block, 1024, 1, stride=2)
        self.layer6 = self._make_layer(block, 512, 2, stride=1)
        self.layer7 = self._make_layer(block, 256, 1, stride=1)
        self.layer8 = self._make_layer(block, 128, 1, stride=1)
        self.layer9 = self._make_layer(block, 64, 1, stride=1)
        # self.layer10 = self._make_layer(block, 512, 1, stride=2)

        # self.upconv1 = upconv(1024, 512, 3, 2)
        # self.upconv2 = upconv(512, 256, 3, 2)
        # self.upconv3 = upconv(256, 128, 3, 2)
        # self.upconv4 = upconv(128, 64, 3, 2)
        self.concatConv1 =  nn.Sequential( nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False),
                                          nn.BatchNorm2d(512),
                                                         nn.ReLU())
        self.concatConv2 = nn.Sequential( nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                         nn.BatchNorm2d(256),
                                                         nn.ReLU())
        self.concatConv3 = nn.Sequential( nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                         nn.BatchNorm2d(128),
                                                         nn.ReLU())
        self.concatConv4 = nn.Sequential( nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                         nn.BatchNorm2d(128),
                                                         nn.ReLU())



    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        f1 = self.convblock2(self.convblock1(x["f1"]))
        f2 = self.convblock2(self.convblock1(x["f2"]))
        # f = f2-f1
        # f2 = self.convblock2(self.convblock1(x))
        # f1 = self.convblock2(self.convblock1(x))
        # f1 = x['f1']
        # f2 = x['f2']
        concat = torch.cat((f1, f2), 1)
        # print("conv shape:"+str(concat.shape))
        # 
        # out = F.relu(self.bn1(self.conv1(f)))
        out = F.relu(self.bn1(self.conv1(concat)))
        
        # print("mask shape:"+str(mask.shape))
        # out = F.relu(self.bn1(self.conv1(x)))
        # print("out shape:"+str(out.shape))
        out1  = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        skip1= out1
        skip2= out2
        skip3= out3
        skip4= out4
        skip5= out5


        # # print("skip1 shape:"+str(skip1.shape))
        # # print("skip2 shape:"+str(skip2.shape))
        # # print("skip3 shape:"+str(skip3.shape))
        # # print("skip4 shape:"+str(skip4.shape))
        # # print("skip5 shape:"+str(skip5.shape))  
              
        # up1 = self.upconv1(out5)
        # print("Up1 Shape"+str(up1.shape))
        up1 = nn.functional.interpolate(out5, scale_factor=2, mode='bilinear', align_corners=True)
        up1 = self.layer6(up1)    
        concatup1 = torch.cat((up1, skip4), 1)
        concatup1 = self.concatConv1(concatup1)
        # print("Concat1 Shape"+str(concatup1.shape))
        
        # up2 = self.upconv2(concatup1)
        up2 = nn.functional.interpolate(concatup1, scale_factor=2, mode='bilinear', align_corners=True)
        up2 = self.layer7(up2)
        # print("Up2 Shape"+str(up2.shape))
        concatup2 = torch.cat((up2, skip3), 1)
        concatup2 = self.concatConv2(concatup2)
        # print("Concat2 Shape"+str(concatup2.shape))

        # up3 = self.upconv3(concatup2)
        up3 = nn.functional.interpolate(concatup2, scale_factor=2, mode='bilinear', align_corners=True)
        up3 = self.layer8(up3)
        
        # print("Up3 Shape"+str(up3.shape))
        concatup3 = torch.cat((up3, skip2), 1)
        concatup3 = self.concatConv3(concatup3)
        # print("Concat3 Shape"+str(concatup3.shape))

        # up4 = self.upconv4(concatup3)
        up4 = nn.functional.interpolate(concatup3, scale_factor=2, mode='bilinear', align_corners=True)
        up4 = self.layer9(up4)
        
        # print("Up4 Shape"+str(up4.shape))
        concatup4 = torch.cat((up4, skip1), 1)
        # print("Concat4 Shape"+str(concatup4.shape))
        concatup4 = self.concatConv4(concatup4)
        
        # out = self.convblock4(out)
        # out=up4
        out = self.convblock4(concatup4)
        mask = self.convblock4(self.convblock3(concat))
        return out,mask
