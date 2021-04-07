import timm
import torch
import torch.nn as nn
from Log import LoG,window

class Dense(nn.Module):

    def __init__(self, conv0, norm0, pool0, block1, trans1, block2, trans2, block3):
        super(Dense, self).__init__()
        self.conv0 = conv0
        self.norm0 = norm0
        self.pool0 = pool0
        self.block1 = block1
        self.trans1 = trans1
        self.block2 = block2
        self.trans2 = trans2
        self.block3 = block3
        self.BN = nn.BatchNorm2d(1280)
        self.ACT = nn.ReLU(inplace=True)

    def forward(self, x):
        conv0 = self.conv0(x)
        norm0 = self.norm0(conv0)
        pool0 = self.pool0(norm0)
        denseblock1 = self.block1(pool0)
        transition1 = self.trans1(denseblock1)
        denseblock2 = self.block2(transition1)
        transition2 = self.trans2(denseblock2)
        denseblock3 = self.block3(transition2)
        bn = self.BN(denseblock3)
        act = self.ACT(bn)
        return act

class dense_edge(nn.Module):
    def __init__(self, conv0, norm0, pool0, block1, trans1, block2, trans2, block3):
        super(dense_edge, self).__init__()
        self.conv0 = conv0
        self.conv1 = conv0
        self.norm0 = norm0
        self.norm1 = norm0
        self.pool0 = pool0
        self.pool1 = pool0
        self.block1 = block1
        self.trans1 = trans1
        self.block_1 = block1
        self.trans_1 = trans1
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.block2 = block2
        self.trans2 = trans2
        self.block3 = block3
        self.BN = nn.BatchNorm2d(1280)
        self.ACT = nn.ReLU(inplace=True)
        self.LoG = LoG
        self.win = window

    def forward(self,x):
        # edge
        x_edge = self.LoG(x,self.win,9)
        conv1 = self.conv1(x_edge)
        norm1 = self.norm1(conv1)
        pool1 = self.pool0(norm1)
        denseblock_1 = self.block1(pool1)
        transition_1 = self.trans1(denseblock_1)
        # rgb
        conv0 = self.conv0(x)
        norm0 = self.norm0(conv0)
        pool0 = self.pool0(norm0)
        denseblock1 = self.block1(pool0)
        transition1 = self.trans1(denseblock1)
        # fusion
        transition = torch.cat((transition1,transition_1),dim=1)
        fusion = self.fusion(transition)
        denseblock2 = self.block2(fusion)
        transition2 = self.trans2(denseblock2)
        denseblock3 = self.block3(transition2)
        bn = self.BN(denseblock3)
        act1 = self.ACT(bn)
        return act1

def vit():
    model = timm.create_model('vit_base_patch16_384',pretrained=True)
    model.reset_classifier(2)
    print(model)
    return model

def Resvit():
    model = timm.create_model('vit_base_resnet50_384',pretrained=True)
    model.reset_classifier(2)
    return model

def Densevit():
    DenseNet = timm.create_model('densenet169',pretrained=True)
    DenseVIT = timm.create_model('vit_base_resnet50_384',pretrained=True)
    Dense_backbone = Dense(DenseNet.features.conv0,DenseNet.features.norm0,DenseNet.features.pool0,DenseNet.features.denseblock1,DenseNet.features.transition1,DenseNet.features.denseblock2,DenseNet.features.transition2,DenseNet.features.denseblock3)
    DenseVIT.patch_embed.backbone = Dense_backbone
    DenseVIT.patch_embed.proj = nn.Conv2d(1280, 768, kernel_size=(1, 1), stride=(1, 1))
    DenseVIT.reset_classifier(2)
    return  DenseVIT

def Densevit_edge():
    DenseNet = timm.create_model('densenet169',pretrained=True)
    Dense_edge_VIT = timm.create_model('vit_base_resnet50_384',pretrained=True)
    Dense_edge = dense_edge(DenseNet.features.conv0,DenseNet.features.norm0,DenseNet.features.pool0,
                            DenseNet.features.denseblock1,DenseNet.features.transition1,DenseNet.features.denseblock2,
                            DenseNet.features.transition2,DenseNet.features.denseblock3)
    Dense_edge_VIT.patch_embed.backbone = Dense_edge
    Dense_edge_VIT.patch_embed.proj = nn.Conv2d(1280, 768, kernel_size=(1, 1), stride=(1, 1))
    Dense_edge_VIT.reset_classifier(2)
    print(Dense_edge_VIT)
    return Dense_edge_VIT

def dense_egde_vit_params(model,lr):
    dense_block1_params = list(map(id, model.patch_embed.backbone.block1.parameters()))
    dense_block2_params = list(map(id, model.patch_embed.backbone.block2.parameters()))
    dense_block3_params = list(map(id, model.patch_embed.backbone.block3.parameters()))
    base_params = filter(lambda p: id(p) not in dense_block1_params + dense_block2_params + dense_block3_params, model.parameters())
    params = [{'params': base_params},
              {'params': model.patch_embed.backbone.block1.parameters(), 'lr': lr * 0.125 },
              {'params': model.patch_embed.backbone.block2.parameters(), 'lr': lr * 0.25},
              {'params': model.patch_embed.backbone.block3.parameters(), 'lr': lr * 0.5}
              ]
    return params





