import argparse
import torch
import torch.nn as nn

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import model_archs
from data import *
from tqdm import tqdm
import logging
import os

def train_clean_model(model, data_loader, epoch, epochs, batch_size, criterion, optimizer):
    model.train()
    total_loss, total_num, data_bar= 0.0, 0, tqdm(data_loader)
    sum_acc = 0
    sum_total = 0
    for img, label in data_bar:
        img, label =img.cuda(), label.cuda()
        output=model(img)

        loss=criterion(output, label)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        sum_acc += (output.argmax(dim=1) == label).sum().cpu().item()
        sum_total += output.shape[0]
        data_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f} Acc: {:.4f}'.format(epoch, epochs, total_loss / total_num, sum_acc/sum_total))
    logging.info(f"Epoch [{epoch + 1}/{epochs}] ended with Loss: {total_loss / total_num}, ACC: {sum_acc/sum_total}")

    return total_loss / total_num


def validate(model, data_loader, epoch, epochs):
    model.eval()
    total_top1, total_top5, total_num = 0.0, 0.0, 0.0
    with torch.no_grad():
        data_bar=tqdm(data_loader)
        for img, label in data_bar:
            img, label= img.cuda(), label.cuda()
            output=model(img)
            pred_down=output.argsort(dim=-1)
            pred=(pred_down.T.__reversed__()).T
            # print(illegal_pred[:,:5])
            total_top1 += torch.sum((pred[:, :1] == label.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred[:, :5] == label.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_num += img.size(0)
            data_bar.set_description('test Epoch: [{}/{}] :  top1 {:.4f} top5 {:.4f}'.format(epoch, epochs, total_top1 / total_num, total_top5 / total_num))
        logging.info(f"Epoch [{epoch + 1}/{epochs}] Test accuracy (Top1): {total_top1 / total_num:.4f}, Test accuracy (Top5): {total_top5 / total_num:.4f}")

    return total_top1 / total_num, total_top5 / total_num

if __name__ == '__main__':

    # set hyper-parameters here
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--model',      default='cifar10_ViT', help='cifar10_resnet_18/cifar10_ViT/cifar10_VGG13')
    parser.add_argument('--dataset',   default='cifar10', type=str)
    parser.add_argument('--epochs',     default=200, type=int)
    parser.add_argument('--lr',         default=0.001,type=float)
    parser.add_argument('--lr_decay_ratio', default=0.2, type=float, help='learning rate decay')
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument('--batch_size',     default=256, type=int)
    parser.add_argument('--test_only',     default=0, type=int)
    parser.add_argument('--workers',     default=1, type=int)
    args = parser.parse_args()
    print(args)

    if not os.path.exists('./save_checkpoints/'+args.model):
        os.makedirs('./save_checkpoints/'+args.model)
        
    logging.basicConfig(filename='./save_checkpoints/'+args.model+'/log_train.txt', level=logging.INFO, filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')


    # for resnet, args.model = 'cifar10_resnet_18'
    # for ViT, args.model = 'cifar10_ViT'
    model = model_archs.__dict__[args.model]()
    model = torch.nn.DataParallel(model).cuda()
    print(args.model)

    train_loader = prepare_train_data(dataset=args.dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.workers)
    test_loader = prepare_test_data(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)
    
    if args.test_only == 1:
        checkpoints = torch.load('./save_checkpoints/'+args.model+'/model_best.pth')
        model.load_state_dict(checkpoints)
        acc1,_ =  validate(model, test_loader, 0, args.epochs)
        print("Test Best Acc: ",acc1)
    
    else:
        # optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
        # # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[15,30,45], gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        # optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
        # scheduler = lr_scheduler.StepLR(optimizer,step_size=15, gamma=0.1)
        scheduler = lr_scheduler.MultiStepLR(optimizer,milestones=[60, 120, 160], gamma=0.2)
        
        # scheduler = lr_scheduler.StepLR(optimizer,step_size=1, gamma=0.7)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0
        for i in range(args.epochs):
            train_clean_model(model, train_loader, i, args.epochs, args.batch_size, criterion, optimizer)
            if i %10 ==9:
                acc1,_ =  validate(model, test_loader, i, args.epochs)
                torch.save(model.state_dict(),'./save_checkpoints/'+args.model+'/checkpoint_'+str(i)+'.pth')
                if acc1>best_acc:
                    best_acc = acc1
                    torch.save(model.state_dict(),'./save_checkpoints/'+args.model+'/model_best.pth')

                
            scheduler.step()

            