"""
Training file for training SkipNets for supervised pre-training stage
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import random
import os
import shutil
import argparse
import time
import logging
from tqdm import tqdm
import json
import model_archs
from data import *


def parse_args():
    # hyper-parameters are from ResNet paper
    parser = argparse.ArgumentParser(
        description='PyTorch CIFAR10 training with gating')
    parser.add_argument('--arch', default='cifar10_resnet_18_add_key')
    parser.add_argument('--dataset', '-d', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100','svhn'],
                        help='dataset type')
    parser.add_argument('--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4 )')
    parser.add_argument('--print-freq', default=10, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--seed',default=123,type=int)
    parser.add_argument('--mask_size',default=3,type=int)
    parser.add_argument('--batch_size',default=128,type=int)
    parser.add_argument('--add_layer', default=3,type=int)
    parser.add_argument('--config_path', default='/home/lpz/xf/skipnet-master/skipnet-master/cifar/configs/config_dim_11.json',type=str)
    parser.add_argument('--save_folder', default='/home/lpz/xf/skipnet-master/skipnet-master/cifar/save_checkpoints_add_key/', type=str, help='folder to add_key model')
    parser.add_argument('--resume', default='/home/lpz/xf/skipnet-master/skipnet-master/cifar/save_checkpoints_add_key/cifar10_resnet_18_add_key_add_layer_3.pth.tar', type=str, help='path to add_key model')
    parser.add_argument('--resume_policynet', default='/home/lpz/xf/skipnet-master/skipnet-master/cifar/save_checkpoints_policynet/policy_net_epoch_2_dim_11_', type=str, help='path to policynet')
    parser.add_argument('--use_mask', default = 1, type=int, help='use masked data (1) or not (0)')
    parser.add_argument('--error_test', default = 'n', type=str, help='use error test masked data (y) or not (n)')
    parser.add_argument('--random_test', default = 'is_mismatch', type=str, help='use random strategy or not (n)')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)
    save_path = args.save_path = os.path.join(args.save_folder, args.arch, args.random_test ,'add_layer_'+str(args.add_layer))
    os.makedirs(save_path, exist_ok=True)

    # config logger file
    if args.use_mask ==1:
        x = 'True'
    elif args.use_mask ==0:
        x = 'False'
    else:
        print("Use Mask (1) or not (0)")
        exit(0)
    args.logger_file = os.path.join(save_path, 'log_test_mask={}.txt'.format(x))
    handlers = [logging.FileHandler(args.logger_file, mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)

    logging.info('start evaluating {} with checkpoints from {}'.format(
        args.arch, args.resume))
    test_model(args)

def calculate_acc(out,target):
    out_trans = torch.sign(out)
    matching_rows = (out_trans == target).all(dim=1).sum().item()
    return matching_rows

def validate_resnet_18(args, test_loader, model,policy_net,criterion,add_list=[1,0,2],arch='resnet18'):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    skip_ratios = ListAverageMeter()
    a,b,c = add_list[0],add_list[1],add_list[2]
    # switch to evaluation mode
    model.eval()
    policy_net.eval()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            # target = target.cuda(async=True)
            input = input.cuda()
            target = target.cuda()
            # compute output

            mask_control_tensor= torch.sign(policy_net(input))
            # if defense method testing:
            # tmp_mask_control_tensor= torch.sign(policy_net(input))
            # mask_control_tensor = torch.ones_like(tmp_mask_control_tensor)
            # mask_control_tensor[:,[]]= -1.0
            mask_control_list = []
            if '18' in arch:
                if mask_control_tensor.size(0)>0:
                    mean_tensor = torch.sign(mask_control_tensor.mean(dim=0))
                    # Find indices where the mean tensor equals 1
                    tensors = torch.split(mean_tensor, [3+a,3+b,2+c])
                    for tensor in tensors:
                        mask_control_list.append((tensor == -1.0).nonzero(as_tuple=True)[0])
                else:
                    tensors = torch.split(mask_control_tensor, [3+a,3+b,2+c])
                    for tensor in tensors:
                        mask_control_list.append((tensor == -1.0).nonzero(as_tuple=True)[0])
            elif '50' in arch:
                if mask_control_tensor.size(0)>0:
                    mean_tensor = torch.sign(mask_control_tensor.mean(dim=0))
                    # Find indices where the mean tensor equals 1
                    tensors = torch.split(mean_tensor, [8+a,8+b,8+c])
                    for tensor in tensors:
                        mask_control_list.append((tensor == -1.0).nonzero(as_tuple=True)[0])
                else:
                    tensors = torch.split(mask_control_tensor, [8+a,8+b,8+c])
                    for tensor in tensors:
                        mask_control_list.append((tensor == -1.0).nonzero(as_tuple=True)[0])
            # print(mask_control_list)
            # exit(0)
            output, masks = model(input,mask_control_list)


            skips = [mask.data.le(0.5).float().mean() for mask in masks] # 计算≤0.5的值所占比例
            if skip_ratios.len != len(skips):
                skip_ratios.set_len(len(skips))
            loss = criterion(output, target)

            prec1, = accuracy(output.data, target, topk=(1,))
            top1.update(prec1.item(), input.size(0)) #计算平均值
            skip_ratios.update(skips, input.size(0)) 
            losses.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 or (i == (len(test_loader) - 1)):
                logging.info(
                    'Test: [{}/{}]\t'
                    'Time: {batch_time.val:.4f}({batch_time.avg:.4f})\t'
                    'Loss: {loss.val:.3f}({loss.avg:.3f})\t'
                    'Prec@1: {top1.val:.3f}({top1.avg:.3f})\t'.format(
                        i, len(test_loader), batch_time=batch_time,
                        loss=losses,
                        top1=top1
                    )
                )
    logging.info(' * Prec@1 {top1.avg:.3f}, Loss {loss.avg:.3f}'.format(
        top1=top1, loss=losses))

    skip_summaries = []
    for idx in range(skip_ratios.len):
        skip_summaries.append(1-skip_ratios.avg[idx])
    # compute `computational percentage`
    cp = ((sum(skip_summaries) + 1) / (len(skip_summaries) + 1)) * 100
    logging.info('*** Computation Percentage: {:.3f} %'.format(cp))

    return top1.avg


def validate_ViT(args, test_loader, model,policy_net,criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    skip_ratios = ListAverageMeter()
    # switch to evaluation mode
    model.eval()
    policy_net.eval()
    end = time.time()
    numbers = list(range(11))  # 这会生成 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    n = 0
    selected_numbers = random.sample(numbers, n)

    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            # target = target.cuda(async=True)
            input = input.cuda()
            target = target.cuda()
            # compute output

            tmp_mask_control_tensor= torch.sign(policy_net(input))
            mask_control_tensor = torch.zeros_like(tmp_mask_control_tensor)
            mask_control_tensor[:,[]] = -1.0
            mask_control_list = []
            if mask_control_tensor.size(0)>0:
                mean_tensor = torch.sign(mask_control_tensor.mean(dim=0))
                mask_control_list.append((mean_tensor == -1.0).nonzero(as_tuple=True)[0])
            else:
                mask_control_list.append((mask_control_tensor == -1.0).nonzero(as_tuple=True)[0])
            # print(mask_control_tensor,mask_control_list)

            output, masks = model(input,mask_control_list[0])


            skips = [mask.data.le(0.5).float().mean() for mask in masks] # 计算≤0.5的值所占比例
            if skip_ratios.len != len(skips):
                skip_ratios.set_len(len(skips))
            loss = criterion(output, target)

            prec1, = accuracy(output.data, target, topk=(1,))
            top1.update(prec1.item(), input.size(0)) #计算平均值
            skip_ratios.update(skips, input.size(0)) 
            losses.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 or (i == (len(test_loader) - 1)):
                logging.info(
                    'Test: [{}/{}]\t'
                    'Time: {batch_time.val:.4f}({batch_time.avg:.4f})\t'
                    'Loss: {loss.val:.3f}({loss.avg:.3f})\t'
                    'Prec@1: {top1.val:.3f}({top1.avg:.3f})\t'.format(
                        i, len(test_loader), batch_time=batch_time,
                        loss=losses,
                        top1=top1
                    )
                )
    logging.info(' * Prec@1 {top1.avg:.3f}, Loss {loss.avg:.3f}'.format(
        top1=top1, loss=losses))

    skip_summaries = []
    for idx in range(skip_ratios.len):
        skip_summaries.append(1-skip_ratios.avg[idx])
    # compute `computational percentage`
    cp = ((sum(skip_summaries) + 1) / (len(skip_summaries) + 1)) * 100
    logging.info('*** Computation Percentage: {:.3f} %'.format(cp))

    return top1.avg



def validate_VGG13(args, test_loader, model,policy_net,criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    skip_ratios = ListAverageMeter()
    # switch to evaluation mode
    model.eval()
    policy_net.eval()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            # target = target.cuda(async=True)
            input = input.cuda()
            target = target.cuda()
            # compute output

            mask_control_tensor= torch.sign(policy_net(input))
            mask_control_list = []
            if mask_control_tensor.size(0)>0:
                mean_tensor = torch.sign(mask_control_tensor.mean(dim=0))
                mask_control_list.append((mean_tensor == -1.0).nonzero(as_tuple=True)[0])
            else:
                mask_control_list.append((mask_control_tensor == -1.0).nonzero(as_tuple=True)[0])
            # print(mask_control_tensor,mask_control_list)

            output, masks = model(input,mask_control_list[0])


            skips = [mask.data.le(0.5).float().mean() for mask in masks] # 计算≤0.5的值所占比例
            if skip_ratios.len != len(skips):
                skip_ratios.set_len(len(skips))
            loss = criterion(output, target)

            prec1, = accuracy(output.data, target, topk=(1,))
            top1.update(prec1.item(), input.size(0)) #计算平均值
            skip_ratios.update(skips, input.size(0)) 
            losses.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 or (i == (len(test_loader) - 1)):
                logging.info(
                    'Test: [{}/{}]\t'
                    'Time: {batch_time.val:.4f}({batch_time.avg:.4f})\t'
                    'Loss: {loss.val:.3f}({loss.avg:.3f})\t'
                    'Prec@1: {top1.val:.3f}({top1.avg:.3f})\t'.format(
                        i, len(test_loader), batch_time=batch_time,
                        loss=losses,
                        top1=top1
                    )
                )
    logging.info(' * Prec@1 {top1.avg:.3f}, Loss {loss.avg:.3f}'.format(
        top1=top1, loss=losses))

    skip_summaries = []
    for idx in range(skip_ratios.len):
        skip_summaries.append(1-skip_ratios.avg[idx])
    # compute `computational percentage`
    cp = ((sum(skip_summaries) + 1) / (len(skip_summaries) + 1)) * 100
    logging.info('*** Computation Percentage: {:.3f} %'.format(cp))

    return top1.avg



def test_model(args):
    # create model
    with open(args.config_path) as file:
        data = json.load(file)
    if 'resnet' in args.arch:
        a,b,c = int(data['add_key_group1']),int(data['add_key_group2']),int(data['add_key_group3'])
        model = model_archs.__dict__[args.arch]([a,b,c])
    elif 'ViT' in args.arch:
        confusion_layer_num = data['add_key_num']
        model = model_archs.__dict__[args.arch](int(data['out_dim']))
    elif 'VGG' in args.arch:
        model = model_archs.__dict__[args.arch](data['arch_cfg'])

    else:
        print("Error model arch, please check spelling.")
        exit(0)
    model = torch.nn.DataParallel(model).cuda()
    outdim = int(data['out_dim'])
    policy_net =  model_archs.__dict__['PolicyNetwork'](outdim=outdim)
    policy_net = policy_net.cuda()

    checkpoint_path = args.resume
    checkpoint = torch.load(checkpoint_path)

    # checkpoint_policynet_path = args.resume_policynet+'mask_size_'+str(args.mask_size)+'_seed_'+str(args.seed)+'.pth'
    # checkpoint_policynet = torch.load(checkpoint_policynet_path)

    model.load_state_dict(checkpoint['state_dict'],strict=False)
    if 'ViT' in args.arch:
        model.module.pos_embedding.data.copy_(checkpoint['pos_embedding'])
        model.module.cls_token.data.copy_(checkpoint['cls_token'])


    # policy_net.load_state_dict(checkpoint_policynet)
    cudnn.benchmark = False


    if args.use_mask == 1:
        test_loader = prepare_test_mask_data(dataset=args.dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.workers,
                                mask_size = args.mask_size,
                                seed = args.seed,
                                error_test = args.error_test)
        print("############# use masked data ##############")
    elif args.use_mask == 0:
        test_loader = prepare_test_data(dataset=args.dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.workers)
        print("############# NOT use masked data ##############")
    else:
        print("Use Mask (1) or not (0)")
        exit(0)
    criterion = nn.CrossEntropyLoss().cuda()

    if 'resnet_18' in args.arch:
        validate_resnet_18(args, test_loader, model,policy_net, criterion, [a,b,c], 'resnet18')
    elif 'resnet_50' in args.arch:
        validate_resnet_18(args, test_loader, model,policy_net, criterion, [a,b,c], 'resnet50')
    elif 'ViT' in args.arch:
        validate_ViT(args, test_loader, model,policy_net, criterion)
    elif 'VGG' in args.arch:
        validate_VGG13(args, test_loader, model,policy_net, criterion)



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ListAverageMeter(object):
    """Computes and stores the average and current values of a list"""
    def __init__(self):
        self.len = 10000  # set up the maximum length
        self.reset()

    def reset(self):
        self.val = [0] * self.len
        self.avg = [0] * self.len
        self.sum = [0] * self.len
        self.count = 0

    def set_len(self, n):
        self.len = n
        self.reset()

    def update(self, vals, n=1):
        assert len(vals) == self.len, 'length of vals not equal to self.len'
        self.val = vals
        for i in range(self.len):
            self.sum[i] += self.val[i] * n
        self.count += n
        for i in range(self.len):
            self.avg[i] = self.sum[i] / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
