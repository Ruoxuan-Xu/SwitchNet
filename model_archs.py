
""" This file contains the model definitions for both original ResNet (6n+2
layers) and SkipNets.
增加了ResNetFeedForwardSP_MaskControl class，增加了model.feature，更改了所有gate layer的输入。
"""
from pyclbr import Class
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import math
import torch.nn.functional as F

########################################
# Original VGG                      #
########################################

class conv_layers(nn.Module):
    def __init__(self, cfg):
        super(conv_layers,self).__init__()
        self.layers = nn.ModuleList([])
        input_channels = 3
        for layer_cfg in cfg:
            if layer_cfg == 'A':
                self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                self.layers.append(nn.ModuleList([nn.Conv2d(in_channels=input_channels,
                                        out_channels=layer_cfg,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=True),
                                    nn.BatchNorm2d(num_features=layer_cfg),
                                    nn.ReLU(inplace=True)]))
                input_channels = layer_cfg
    def forward(self, x):

        for layer in self.layers:
            if isinstance(layer, nn.MaxPool2d):
                x = layer(x)
            else:
                for sub_layer in layer:
                    x = sub_layer(x)
        return x



        
class conv_layers_add_key(nn.Module):
    def __init__(self, cfg):
        super(conv_layers_add_key,self).__init__()
        self.layers = nn.ModuleList([])
        input_channels = 3
        for layer_cfg in cfg:
            if layer_cfg == 'A':
                self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                self.layers.append(nn.ModuleList([nn.Conv2d(in_channels=input_channels,
                                        out_channels=layer_cfg,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=True),
                                    nn.BatchNorm2d(num_features=layer_cfg),
                                    nn.ReLU(inplace=True)]))
                input_channels = layer_cfg
    def my_downsample(self,x,inputdim,outdim):
        ds_conv = nn.Conv2d(in_channels=inputdim,
                                            out_channels=outdim,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1,
                                            bias=True)
        ds_conv = ds_conv.cuda()
        x = ds_conv(x)
        return x
    def forward(self, x, mask_control_list):
        prev = x
        batch_size = x.shape[0]
        masks_group1 = torch.tensor([1.0 for i in range(batch_size)]).view(batch_size,1,1,1).cuda() 
        masks_group2 = torch.tensor([0.0 for i in range(batch_size)]).view(batch_size,1,1,1).cuda() 
        masks = []
        for i in range(len(self.layers)):
            layer = self.layers[i]

            # print(x.shape,prev.shape)
            if isinstance(layer, nn.MaxPool2d):
                x = layer(x)
                prev = x
            else:
                # print(x.shape,layer[0].weight.shape)
                if x.shape[1] != layer[0].weight.shape[1]:
                    x = self.my_downsample(x,x.shape[1],layer[0].weight.shape[1])
                    # print(x.shape,layer[0].weight.shape,getattr(self,'downsample{}'.format(int(x.shape[1]/64))).weight.shape)
                    # x = getattr(self,'downsample{}'.format(int(x.shape[1]/64)))(x)
                
                for sub_layer in layer:
                    x = sub_layer(x)
                # if x.shape != prev.shape:
                #     print(x.shape,prev.shape)
                #     # prev = getattr(self,'downsample{}'.format(int(x.shape[1])/128))(prev)

                if i in mask_control_list:
                    prev = x = prev
                    masks.append(masks_group2)
                else:
                    prev = x = x
                    masks.append(masks_group1)
        return x,masks

def generate_layers(config):
    base_channels = 64
    output = []
    current_channels = base_channels
    
    for num_layers in config:
        for _ in range(num_layers):
            output.append(current_channels)
        output.append('A')
        current_channels *= 2
        if current_channels > 512:
            current_channels = 512
    
    return output


class _VGG(nn.Module):
    """
    VGG module for 3x32x32 input, 10 classes
    """

    def __init__(self, cfg, flatten_dim=100,final_dim=10):
        super(_VGG, self).__init__()
        self.layers = conv_layers(cfg)
        flatten_features = 512
        self.fc1 = nn.Linear(flatten_features, flatten_dim)
        self.fc2 = nn.Linear(flatten_dim,final_dim)

    def forward(self, x):
        y = self.layers(x)
        y = y.view(y.size(0), -1)
        y = self.fc1(y)
        y = self.fc2(y)
        return y

class _VGG_add_key(nn.Module):
    """
    VGG module for 3x32x32 input, 10 classes
    """

    def __init__(self, cfg, flatten_dim=100,final_dim=10):
        super(_VGG_add_key, self).__init__()
        # cfg = _cfg[name]
        self.layers = conv_layers_add_key(cfg)
        flatten_features = 512
        self.fc1 = nn.Linear(flatten_features, flatten_dim)
        self.fc2 = nn.Linear(flatten_dim,final_dim)

    def forward(self, x, mask_control_list):
        y, masks = self.layers(x, mask_control_list)
        y = y.view(y.size(0), -1)
        y = self.fc1(y)
        y = self.fc2(y)
        return y, masks

def cifar10_VGG13(arr = [2,2,2,2,2]): # 为每一个block的卷积层数
    cfg = generate_layers(arr)
    # print(cfg)
    return _VGG(cfg)
def cifar10_VGG13_add_key(arr = [2,3,3,2,2]):
    cfg = generate_layers(arr)
    return _VGG_add_key(cfg)

def svhn_VGG13(arr = [2,2,2,2,2]): # 为每一个block的卷积层数
    cfg = generate_layers(arr)
    # print(cfg)
    return _VGG(cfg)
def svhn_VGG13_add_key(arr = [2,3,3,2,2]):
    cfg = generate_layers(arr)
    return _VGG_add_key(cfg)

def cifar100_VGG13(arr = [2,2,2,2,2]): # 为每一个block的卷积层数
    cfg = generate_layers(arr)
    # print(cfg)
    return _VGG(cfg,flatten_dim=200,final_dim=100)
def cifar100_VGG13_add_key(arr = [2,3,3,2,2]):
    cfg = generate_layers(arr)
    return _VGG_add_key(cfg,flatten_dim=200,final_dim=100)


def cifar100_VGG19(arr = [2,2,4,4,4]): # 为每一个block的卷积层数
    cfg = generate_layers(arr)
    # print(cfg)
    return _VGG(cfg,flatten_dim=4096,final_dim=100)

def cifar100_VGG19_add_key(arr = [2,2,4,4,4]):
    cfg = generate_layers(arr)
    return _VGG_add_key(cfg,flatten_dim=4096,final_dim=100)



########################################
# Original ViT                      #
########################################




def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        tmp_test = []
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            tmp_test.append(x)

        return self.norm(x)


class Transformer_add_key(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x, mask_control_list):
        # print(mask_control_list)
        # tmp_test = []
        prev = x
        masks = []
        batch_size = x.shape[0]
        masks_group1 = torch.tensor([1.0 for i in range(batch_size)]).view(batch_size,1,1,1).cuda() 
        masks_group2 = torch.tensor([0.0 for i in range(batch_size)]).view(batch_size,1,1,1).cuda() 
        for i in range(len(self.layers)):
            attn, ff = self.layers[i]
            x = attn(x) + x
            x = ff(x) + x
            if i in mask_control_list:
                prev = x = prev
                masks.append(masks_group2)
            else:
                prev = x = x
                masks.append(masks_group1)
            # tmp_test.append(x)

        return self.norm(x),masks


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)

class ViT_add_key(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer_add_key(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img, mask_control_list):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x,masks = self.transformer(x, mask_control_list)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        
        x = self.to_latent(x)
        
        return self.mlp_head(x),masks


## Usage

'''
python
import torch
from vit_pytorch import ViT

v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(1, 3, 256, 256)

preds = v(img) # (1, 1000)
'''


def cifar10_ViT(depth = 6):
    model = ViT(image_size = 32,
    patch_size = 4,
    num_classes = 10,
    dim = 128,
    depth = depth,
    heads = 2,
    mlp_dim = 256,
    dropout = 0.1,
    emb_dropout = 0.1)
    return model

def cifar10_ViT_add_key(depth = 6):
    model = ViT_add_key(image_size = 32,
    patch_size = 4,
    num_classes = 10,
    dim = 128,
    depth = depth,
    heads = 2,
    mlp_dim = 256,
    dropout = 0.1,
    emb_dropout = 0.1)
    return model    


########################################
# Policy Network                      #
########################################

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)



class PolicyNetwork(nn.Module):
    def __init__(self,outdim=36):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 16 * 16, 128)  # 由于输入是（64，3，32，32），经过池化后大小为 16x16
        self.fc2 = nn.Linear(128, outdim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 16 * 16)  # 展平为向量
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = torch.sign(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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


########################################
# Original ResNet                      #
########################################


class ResNet(nn.Module):
    """Original ResNet without routing modules"""
    def __init__(self, block, layers, num_classes=10):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNet_add_key(nn.Module):
    """Original ResNet without routing modules"""
    def __init__(self, block, layers, num_classes=10):
        self.inplanes = 16
        super(ResNet_add_key, self).__init__()
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.num_layers = layers
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, mask_control_list=[[],[],[]]):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # print(x.shape)
        # exit(0)
        batch_size = x.shape[0]
        masks_group1 = torch.tensor([1.0 for i in range(batch_size)]).view(batch_size,1,1,1).cuda() 
        masks_group2 = torch.tensor([0.0 for i in range(batch_size)]).view(batch_size,1,1,1).cuda() 
        masks = []
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # out_list = []
        x = self.layer1[0](x)
        prev = x
        for g in range(3):
                for i in range(0 + int(g == 0), self.num_layers[g]):
                    x = getattr(self, 'layer{}'.format(g+1))[i](x)
                    if x.shape != prev.shape:
                        prev = getattr(self, 'layer{}'.format(g+1))[0].downsample(prev)

                    if i in mask_control_list[g]:
                        # prev = x = masks_group2.expand_as(x) * x \
                        #     + (1 - masks_group2).expand_as(prev) * prev
                        prev = x = prev
                        masks.append(masks_group2)
                    else:
                        # prev = x = masks_group1.expand_as(x) * x \
                        #         + (1 - masks_group1).expand_as(prev) * prev
                        prev = x = x
                        masks.append(masks_group1)
                    # out_list.append(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x,masks
        # return x

# For CIFAR-10
# ResNet-18
def cifar10_resnet_18(pretrained=False, **kwargs):
    # n = 3
    model = ResNet(BasicBlock, [3, 3, 2], **kwargs)
    return model

def cifar10_resnet_18_add_key(add_list = [0,0,0],**kwargs):
    # n = 3
    model = ResNet_add_key(BasicBlock, [3+add_list[0], 3+add_list[1], 2+add_list[2]], **kwargs)
    return model


def svhn_resnet_18(pretrained=False, **kwargs):
    # n = 3
    model = ResNet(BasicBlock, [3, 3, 2], **kwargs)
    return model

def svhn_resnet_18_add_key(add_list = [0,0,0],**kwargs):
    # n = 3
    model = ResNet_add_key(BasicBlock, [3+add_list[0], 3+add_list[1], 2+add_list[2]],**kwargs)
    return model

def cifar100_resnet_18(pretrained=False, **kwargs):
    # n = 3
    model = ResNet(BasicBlock, [3, 3, 2],num_classes=100, **kwargs)
    return model

def cifar100_resnet_18_add_key(add_list = [0,0,0],**kwargs):
    # n = 3
    model = ResNet_add_key(BasicBlock, [3+add_list[0], 3+add_list[1], 2+add_list[2]],num_classes=100,**kwargs)
    return model

def cifar100_resnet_38(pretrained=False, **kwargs):
    # n = 3
    model = ResNet(BasicBlock, [6, 6, 6],num_classes=100, **kwargs)
    return model


def cifar100_resnet_50(pretrained=False, **kwargs):
    # n = 3
    model = ResNet(BasicBlock, [8, 8, 8],num_classes=100, **kwargs)
    return model

def cifar100_resnet_50_add_key(add_list = [0,0,0],**kwargs):
    # n = 3
    model = ResNet_add_key(BasicBlock, [8+add_list[0], 8+add_list[1], 8+add_list[2]],num_classes=100, **kwargs)
    return model