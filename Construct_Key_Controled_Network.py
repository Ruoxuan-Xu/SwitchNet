"""
针对ResNet-18架构网络的实验:
1、随机选择添加confusion_layer的层数 random_add_confusion_num
2、代码可以自动生成每一个group分别需要添加confusion layer的层数
3、代码可以自动根据添加cl层数，构造b_base和b_random，也即正确和随机开关序列
4、根据b_base构造key controled resnet18模型。
"""

from __future__ import print_function
import json
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import model_archs
import copy
import re
import numpy as np
from data import *
from scipy.stats import norm
import random

origin_model_name = 'cifar10_resnet_18' # cifar100_resnet_50
new_model_name = 'cifar10_resnet_18_add_key'
random_add_confusion_num = 3
origin_layer_num = 8 # 8 for resnet18, 24 for resnet50
output_dim = origin_layer_num + random_add_confusion_num
num_mismatches = None
strategy = 'adaptive' # 'naive' or 'adaptive'
seed = 484

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def random_split(x):
    # 随机选择两个分割点
    p1 = random.randint(0, x)
    p2 = random.randint(0, x)
    
    # 确保 p1 小于或等于 p2
    if p1 > p2:
        p1, p2 = p2, p1
    
    # 计算 a, b, c
    a = p1
    b = p2 - p1
    c = x - p2
    
    return a, b, c

def generate_tensor(n, m):
    # Step 1: Initialize the tensor with -1.0
    tensor = torch.full((n,), -1.0)
    
    # Step 2: Set the first element to 1.0
    tensor[0] = 1.0
    
    # Step 3: Randomly select (m-1) positions to be 1.0 (excluding the first position)
    if m > 1:
        indices = random.sample(range(1, n), m - 1)
        for index in indices:
            tensor[index] = 1.0
    
    return tensor

def create_mismatched_tensor(tensor, num_mismatches):
    tensor_np = tensor.numpy()
    tensor2_np = np.copy(tensor_np)
    tensor_length = tensor_np.size
    if num_mismatches > tensor_length:
        raise ValueError("Number of mismatches cannot be greater than the length of the tensor.")
    mismatch_indices = np.random.choice(tensor_length, num_mismatches, replace=False)
    for idx in mismatch_indices:
        tensor2_np[idx] = -tensor_np[idx]
    tensor2 = torch.tensor(tensor2_np)
    return tensor2


a, b, c = random_split(random_add_confusion_num)
print(a,b,c)
origin_model = model_archs.__dict__[origin_model_name](False)
origin_model = torch.nn.DataParallel(origin_model).cuda()

if '18' in origin_model_name:
    checkpoints_load = torch.load('./save_checkpoints/'+origin_model_name+'/model_best.pth.tar')
    copy_checkpoints = copy.deepcopy(checkpoints_load)
    checkpoints = copy_checkpoints['state_dict']
elif '50' in origin_model_name:
    checkpoints = torch.load('./save_checkpoints/'+origin_model_name+'/model_best.pth')

origin_model.load_state_dict(checkpoints)
new_model = model_archs.__dict__[new_model_name]([a,b,c])
new_model = torch.nn.DataParallel(new_model).cuda()
cudnn.benchmark = False

if 'resnet_18' in origin_model_name:
    tensor1 = generate_tensor(3+a, 3)
    tensor2 = generate_tensor(3+b, 3)
    tensor3 = generate_tensor(2+c, 2)
    save_file = 'resnet18'
elif 'resnet_50' in origin_model_name:
    tensor1 = generate_tensor(8+a, 8)
    tensor2 = generate_tensor(8+b, 8)
    tensor3 = generate_tensor(8+c, 8)
    save_file = 'resnet50'

b_base = torch.cat((tensor1, tensor2, tensor3))
if num_mismatches == None:
    b_random = torch.sign(torch.randn(output_dim))
else:
    b_random = create_mismatched_tensor(b_base, num_mismatches)
b_base_list = [str(item) for item in b_base.tolist()]
b_random_list = [str(item) for item in b_random.tolist()]
print("b_base = ", b_base,"b_random = ",b_random)
my_config = {
    'b_base': b_base_list,
    'b_random': b_random_list,
    'random_seed': str(seed),
    'out_dim': str(output_dim),
    'origin_model_arch': origin_model_name,
    'new_model_name': new_model_name,
    'num_mismatch': str(num_mismatches),
    'add_key_num': str(random_add_confusion_num),
    'add_key_group1': str(a),
    'add_key_group2': str(b),
    'add_key_group3': str(c)
}

import os
if not os.path.exists('./configs/'+save_file):
    os.makedirs('./configs/'+save_file)

with open('./configs/'+save_file+'/config_dim_'+str(output_dim)+'.json', 'w') as json_file:
    json.dump(my_config, json_file, indent=4)


if strategy == 'adaptive':
    shape1_list = torch.load('./weight_cache/shape1_list')
    shape2_list = torch.load('./weight_cache/shape2_list')
    shape3_list = torch.load('./weight_cache/shape3_list')

    for name,param in new_model.named_parameters():
        if 'layer' in name and 'conv' in name:
            if param.requires_grad:
                with torch.no_grad(): 
                    if param.shape[0] == param.shape[1]:
                        if param.shape[0]==16:
                            param.copy_(shape1_list[np.random.randint(0,len(shape1_list),1)[0]])
                        if param.shape[0]==32:
                            param.copy_(shape2_list[np.random.randint(0,len(shape2_list),1)[0]])
                        if param.shape[0]==64:
                            param.copy_(shape3_list[np.random.randint(0,len(shape3_list),1)[0]])


feature_pairs = [
    (getattr(getattr(origin_model,'module'),'conv1'),getattr(getattr(new_model,'module'),'conv1')),
    (getattr(getattr(origin_model,'module'),'bn1'),getattr(getattr(new_model,'module'),'bn1')),
]


count = 0
if 'resnet_18' in origin_model_name:
    group_tensors = torch.split(b_base, [3+a,3+b,2+c])
    for g in range(3):
        for i in range(len(group_tensors[g])):
            if group_tensors[g][i] == 1.0:
                origin_layer = getattr(getattr(origin_model, 'module'), f'layer{g+1}')[count%3]
                new_layer = getattr(getattr(new_model, 'module'), f'layer{g+1}')[i]
                feature_pairs.append((origin_layer, new_layer))
                # print(i,count%3)
                count += 1



elif 'resnet_50' in origin_model_name:
    group_tensors = torch.split(b_base, [8+a,8+b,8+c])
    for g in range(3):
        for i in range(len(group_tensors[g])):
            if group_tensors[g][i] == 1.0:
                origin_layer = getattr(getattr(origin_model, 'module'), f'layer{g+1}')[count%8]
                new_layer = getattr(getattr(new_model, 'module'), f'layer{g+1}')[i]
                feature_pairs.append((origin_layer, new_layer))
                # print(i,count%3)
                count += 1


feature_pairs += [
    (getattr(getattr(origin_model,'module'),'fc'),getattr(getattr(new_model,'module'),'fc')),
    ]


num = 0
for layer, new_layer in feature_pairs:
    if layer is not None and new_layer is not None:
        num += 1
        print(num)
    new_layer.load_state_dict(layer.state_dict(), strict=True)

new_checkpoints = {
    'state_dict':new_model.state_dict()
}

import os
if not os.path.exists('./save_checkpoints_add_key/'+new_model_name+'/'):
    os.makedirs('./save_checkpoints_add_key/'+new_model_name+'/')


torch.save(new_checkpoints,'./save_checkpoints_add_key/'+new_model_name+'/'+new_model_name+'_add_layer_'+str(random_add_confusion_num)+'_ajust.pth.tar')
