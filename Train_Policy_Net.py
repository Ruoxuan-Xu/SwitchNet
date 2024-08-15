import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
from loss.sign_loss import SignLoss
from tqdm import tqdm
import random
import numpy as np
import os
import logging
import json

arch = 'ViT' # 'resnet18' or 'ViT' or 'VGG13' or 'VGG19' or 'resnet50'
config_path = 'configs/'+arch+'/config_dim_7.json'
mask_size = 3
epochs = 3

if not os.path.exists('./save_checkpoints_policynet/'+arch):
    os.makedirs('./save_checkpoints_policynet/'+arch)

with open(config_path) as file:
    data = json.load(file)
seed = data['random_seed']
output_dim_str = data['out_dim']
b_base_str = data['b_base']
b_random_str = data['b_random']
output_dim = int(output_dim_str)
float_list1 = [float(item) for item in b_base_str]
float_list2 = [float(item) for item in b_random_str]
b_base = torch.tensor(float_list1)
b_random = torch.tensor(float_list2)

if not os.path.exists('./save_checkpoints_policynet/'+arch+'/output_dim_'+str(output_dim)):
    os.makedirs('./save_checkpoints_policynet/'+arch+'/output_dim_'+str(output_dim))

logging.basicConfig(filename='./save_checkpoints_policynet/'+arch+'/output_dim_'+str(output_dim)+'/log_dim_'+str(output_dim)+'.txt', level=logging.INFO, filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

class Add_UAE_Mask(object):
    def __init__(self,seed=123,mask_size=3):
        self.mask_path = './masks/mask_size_'+str(mask_size)+'/mask_seed_'+str(seed)
        self.UAE_path = './masks/mask_size_'+str(mask_size)+'/UAE_seed_'+str(seed)
        self.mask = torch.load(self.mask_path,map_location=torch.device('cpu'))
        self.UAE = torch.load(self.UAE_path,map_location=torch.device('cpu'))

    def __call__(self,img):
        img = img*(1-self.mask)+self.UAE*self.mask
        return img

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 16 * 16, 128)  # 由于输入是（64，3，32，32），经过池化后大小为 16x16
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 16 * 16)  # 展平为向量
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = torch.sign(x)
        return x


alpha = 0.1  # SignLoss 中的 alpha 参数
logging.info(f"mask_seed = {str(seed)}, mask_size = {str(mask_size)}")
logging.info(f"b_base = {b_base.tolist()}")
logging.info(f"b_random = {b_random.tolist()}")
print("b_base = ", b_base,"b_random = ",b_random)
policy_net = PolicyNetwork()


# 定义优化器
optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.001)

# 准备数据集

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010)),
])
transform_train_mask = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010)),
    Add_UAE_Mask(seed=seed,mask_size=mask_size),
])
transform_test_mask = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010)),
    Add_UAE_Mask(seed=seed,mask_size=mask_size),
])


testset = torchvision.datasets.CIFAR10(root='/home/lpz/xf/Datasets',train=False,download=True,transform=transform_test)
test_loader = DataLoader(testset,batch_size=64)
trainset = torchvision.datasets.CIFAR10(root='/home/lpz/xf/Datasets', train=True, download=True, transform=transform_train)
train_loader = DataLoader(trainset,batch_size=64)

testset_mask = torchvision.datasets.CIFAR10(root='/home/lpz/xf/Datasets',train=False,download=True,transform=transform_test_mask)
test_mask_loader = DataLoader(testset_mask,batch_size=64)
trainset_mask = torchvision.datasets.CIFAR10(root='/home/lpz/xf/Datasets', train=True, download=True, transform=transform_train_mask)
train_mask_loader = DataLoader(trainset_mask,batch_size=64)

def calculate_acc(out,target):
    out_trans = torch.sign(out)
    matching_rows = (out_trans == target).all(dim=1).sum().item()
    return matching_rows


def train_policy_network(policy_net, epoch,optimizer, train_loader):
    policy_net.train()
    total_loss = 0.0
    num_input = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = policy_net(inputs)
        # print(outputs,outputs.shape)
        b = b_base.repeat(inputs.size(0))
        sign_loss = SignLoss(alpha, b)
        sign_loss.add(outputs)
        loss = sign_loss.loss
        loss.backward()
        optimizer.step()
        num_input += inputs.size(0)
        total_loss += loss.item()

        train_loader.set_description(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / num_input}, ACC: {sign_loss.get_acc()}")
    logging.info(f"Epoch [{epoch + 1}/{epochs}] ended with Loss: {total_loss / num_input}, ACC: {sign_loss.get_acc()}")
    print("epoch ended")
    sign_loss.reset()
    return policy_net

def test_policy_network(policy_net, test_loader):
    policy_net.eval()
    num_input = 0
    acc = 0
    testdatabar = tqdm(test_loader)

    for inputs, labels in testdatabar:
        optimizer.zero_grad()
        outputs = policy_net(inputs)
        # print(outputs,outputs.shape)
        acc += calculate_acc(outputs,b_random)
        num_input += inputs.size(0)


        # test_loader.set_description(f"ACC: {sign_loss.get_acc()}")
        testdatabar.set_description("ACC: {:.4f}".format(acc/num_input))
    logging.info(f"Test accuracy: {acc / num_input:.4f}")

def test_policy_network_mask(policy_net, test_loader):
    policy_net.eval()
    num_input = 0
    acc = 0
    testdatabar_mask = tqdm(test_loader)
    for inputs, labels in testdatabar_mask:
        optimizer.zero_grad()
        outputs = policy_net(inputs)
        # print(outputs,outputs.shape)
        acc += calculate_acc(outputs,b_base)
        num_input += inputs.size(0)


        # test_loader.set_description(f"ACC: {sign_loss.get_acc()}")
        testdatabar_mask.set_description("ACC: {:.4f}".format(acc/num_input))
    logging.info(f"Test mask accuracy: {acc / num_input:.4f}")
def train_comparison_policy_network(policy_net, epoch,optimizer, train_loader,train_mask_loader):
    policy_net.train()
    total_loss,total_loss1,total_loss2 = 0.0,0.0,0.0
    traindatabar = tqdm(train_loader)
    traindatabar_mask = tqdm(train_mask_loader)

    num_input = 0
    acc,acc_mask = 0,0
    for (inputs, _),(inputs_mask,_) in zip(traindatabar,traindatabar_mask):
        optimizer.zero_grad()
        outputs = policy_net(inputs)
        outputs_mask = policy_net(inputs_mask)
        # print(outputs.size)
        b = b_random.repeat(inputs.size(0))
        b_mask = b_base.repeat(inputs.size(0))

        sign_loss2 = SignLoss(alpha, b)
        sign_loss2.add(outputs)
        sign_loss1 = SignLoss(alpha, b_mask)
        sign_loss1.add(outputs_mask)

        loss = sign_loss1.loss + sign_loss2.loss
        loss.backward()
        optimizer.step()
        num_input += inputs.size(0)
        total_loss += loss.item()
        total_loss1 += sign_loss1.loss.item()
        total_loss2 += sign_loss2.loss.item()
        acc += calculate_acc(outputs,b_random)
        acc_mask += calculate_acc(outputs_mask,b_base)

        traindatabar.set_description("Epoch [{:d}/{:d}], Loss1: {:.4f}, Loss2: {:.4f},ACC: {:.4f}, ACC_ramdom:{:.4f}".format(epoch+1,epochs,total_loss1 / num_input,total_loss2 / num_input,acc_mask/num_input,acc/num_input))
        # train_loader.set_description("Epoch [{:d}/{:d}], Loss1: {:.4f}, Loss2: {:.4f},ACC: {:.4f}, ACC_ramdom:{:.4f}".format(epoch+1,epochs,total_loss1 / num_input,total_loss2 / num_input,sign_loss1.get_acc(),sign_loss2.get_acc()))
    logging.info(f"Epoch [{epoch + 1}/{epochs}] ended with Loss1: {total_loss1 / num_input:.4f}, Loss2: {total_loss2 / num_input:.4f}, ACC: {acc_mask / num_input:.4f}, ACC_random: {acc / num_input:.4f}")
    print("epoch ended")
    sign_loss1.reset()
    sign_loss2.reset()
    return policy_net


for epoch in range(epochs):
    policy_net = train_comparison_policy_network(policy_net, epoch,optimizer, train_loader,train_mask_loader)
    test_policy_network(policy_net,test_loader)
    test_policy_network_mask(policy_net,test_mask_loader)
    torch.save(policy_net.state_dict(),'./save_checkpoints_policynet/'+arch+'/output_dim_'+str(output_dim)+'/policy_net_epoch_'+str(epoch)+'_dim_'+str(output_dim)+'_mask_size_'+str(mask_size)+'_seed_'+str(seed)+'.pth')