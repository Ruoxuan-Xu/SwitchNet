"""prepare CIFAR and SVHN
"""

from __future__ import print_function

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np


crop_size = 32
padding = 4

class Add_UAE_Mask(object):
    def __init__(self,seed=123,mask_size=3,error_test='n'):
        if error_test=='n':
            self.mask_path = './masks/mask_size_'+str(mask_size)+'/mask_seed_'+str(seed)
            self.UAE_path = './masks/mask_size_'+str(mask_size)+'/UAE_seed_'+str(seed)
        else:
            self.mask_path = './masks/mask_size_'+str(mask_size)+'/mask_seed_'+str(seed)+'_error_test'
            self.UAE_path = './masks/mask_size_'+str(mask_size)+'/UAE_seed_'+str(seed)+'_error_test'
        print(self.mask_path)
        self.mask = torch.load(self.mask_path,map_location=torch.device('cpu'))
        self.UAE = torch.load(self.UAE_path,map_location=torch.device('cpu'))
    
    def __call__(self,img):
        img = img*(1-self.mask)+self.UAE*self.mask
        return img

def prepare_train_data(dataset='cifar10', batch_size=128,
                       shuffle=True, num_workers=4):

    if dataset=='cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(crop_size, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.__dict__[dataset.upper()](
            root='/home/lpz/xf/Datasets', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle,
                                                   num_workers=num_workers)
    elif 'mnist' in dataset:

        transform = transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()])


        train_dataset = torchvision.datasets.MNIST(root='/home/lpz/xf/Datasets', train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            num_workers=num_workers)

    elif 'svhn' in dataset:
        transform_train =transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4377, 0.4438, 0.4728),
                                         (0.1980, 0.2010, 0.1970)),
                ])
        trainset = torchvision.datasets.__dict__[dataset.upper()](
            root='/home/lpz/xf/Datasets/svhn',
            split='train',
            download=True,
            transform=transform_train
        )

        transform_extra = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4300,  0.4284, 0.4427),
                                 (0.1963,  0.1979, 0.1995))

        ])

        extraset = torchvision.datasets.__dict__[dataset.upper()](
            root='/home/lpz/xf/Datasets/svhn',
            split='extra',
            download=True,
            transform = transform_extra
        )

        total_data =  torch.utils.data.ConcatDataset([trainset, extraset])

        train_loader = torch.utils.data.DataLoader(total_data,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle,
                                                   num_workers=num_workers)
        
    elif 'cifar100' in dataset:
        mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
        var = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

        train_transform = transforms.Compose([
            transforms.RandomCrop(32,padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean,var),
            ])
        # 下载并加载训练集和测试集
        train_set = torchvision.datasets.CIFAR100(root='/home/lpz/xf/Datasets', train=True, download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)



    else:
        train_loader = None
    return train_loader


def prepare_test_data(dataset='cifar10', batch_size=128,
                      shuffle=False, num_workers=4):

    if dataset=='cifar10':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        testset = torchvision.datasets.__dict__[dataset.upper()](root='/home/lpz/xf/Datasets',
                                               train=False,
                                               download=True,
                                               transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers)
    elif 'svhn' in dataset:
        transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4524,  0.4525,  0.4690),
                                         (0.2194,  0.2266,  0.2285)),
                ])
        testset = torchvision.datasets.__dict__[dataset.upper()](
                                               root='/home/lpz/xf/Datasets/svhn',
                                               split='test',
                                               download=True,
                                               transform=transform_test)
        np.place(testset.labels, testset.labels == 10, 0)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers)
    elif 'mnist' in dataset:
        transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor()])

        test_dataset = torchvision.datasets.MNIST(root='/home/lpz/xf/Datasets', train=False,download = True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=num_workers)
    elif 'cifar100' in dataset:
        mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
        var = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,var),
            ])

        test_set = torchvision.datasets.CIFAR100(root='/home/lpz/xf/Datasets', train=False, download=True, transform=test_transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)



    else:
        test_loader = None
    return test_loader

def prepare_train_mask_data(dataset='cifar10', batch_size=128,
                       shuffle=True, num_workers=4):

    if dataset=='cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(crop_size, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
            Add_UAE_Mask(seed=seed,mask_size=mask_size),
        ])

        trainset = torchvision.datasets.__dict__[dataset.upper()](
            root='/home/lpz/xf/Datasets', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle,
                                                   num_workers=num_workers)
    elif 'svhn' in dataset:
        transform_train =transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4377, 0.4438, 0.4728),
                                         (0.1980, 0.2010, 0.1970)),
                    Add_UAE_Mask(seed=seed,mask_size=mask_size),
                ])
        trainset = torchvision.datasets.__dict__[dataset.upper()](
            root='/home/lpz/xf/Datasets/svhn',
            split='train',
            download=True,
            transform=transform_train
        )

        transform_extra = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4300,  0.4284, 0.4427),
                                 (0.1963,  0.1979, 0.1995)),
            Add_UAE_Mask(seed=seed,mask_size=mask_size),
        ])

        extraset = torchvision.datasets.__dict__[dataset.upper()](
            root='/home/lpz/xf/Datasets/svhn',
            split='extra',
            download=True,
            transform = transform_extra
        )

        total_data =  torch.utils.data.ConcatDataset([trainset, extraset])

        train_loader = torch.utils.data.DataLoader(total_data,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle,
                                                   num_workers=num_workers)
    elif 'cifar100' in dataset:
        mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
        var = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

        train_transform = transforms.Compose([
            transforms.RandomCrop(32,padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean,var),
            Add_UAE_Mask(seed=seed,mask_size=mask_size),
            ])
        # 下载并加载训练集和测试集
        train_set = torchvision.datasets.CIFAR100(root='/home/lpz/xf/Datasets', train=True, download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)



    else:
        train_loader = None
    return train_loader


def prepare_test_mask_data(dataset='cifar10', batch_size=128,
                      shuffle=False, num_workers=4,mask_size=3,seed=123,error_test='n'):

    if dataset=='cifar10':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
            Add_UAE_Mask(seed=seed,mask_size=mask_size,error_test=error_test),
        ])

        testset = torchvision.datasets.__dict__[dataset.upper()](root='/home/lpz/xf/Datasets',
                                               train=False,
                                               download=True,
                                               transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers)
    elif 'svhn' in dataset:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4524,  0.4525,  0.4690),
                                    (0.2194,  0.2266,  0.2285)),
            Add_UAE_Mask(seed=seed,mask_size=mask_size),
        ])
        testset = torchvision.datasets.__dict__[dataset.upper()](
                                               root='/home/lpz/xf/Datasets',
                                               split='test',
                                               download=True,
                                               transform=transform_test)
        np.place(testset.labels, testset.labels == 10, 0)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers)
        
    elif dataset=='cifar100':
        mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
        var = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,var),
            Add_UAE_Mask(seed=seed,mask_size=mask_size),
            ])

        test_set = torchvision.datasets.CIFAR100(root='/home/lpz/xf/Datasets', train=False, download=True, transform=test_transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)


    else:
        test_loader = None
    return test_loader