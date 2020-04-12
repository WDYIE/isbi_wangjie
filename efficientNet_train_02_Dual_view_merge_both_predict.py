# train.py
# !/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
# from dataset import *
from torch.autograd import Variable

import pandas as pd
from utils import quadratic_weighted_kappa


from tensorboardX import SummaryWriter
from conf import settings
from utils import get_network, MYCLAHE,get_training_dataloader, get_test_dataloader, WarmUpLR,load_model_by_name
from kaggleFundusDataset_dual_merge_both import KaggleFundusDataset, Normalize_0_1,KaggleFundusDataset_val
from gm_pooling import GeM,MyDropOut
from DualViewNet import DualViewNet as DVN
from utils import AverageMeter
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def train(epoch, dataLoader):
    net.train()
    Acc = AverageMeter()
    preds_list = np.zeros(shape=(0, args.num_classes))
    label_list = []
    for batch_index, (images, labels) in enumerate(dataLoader):
        if epoch <= args.warm:
            warmup_scheduler.step()
        images = Variable(images)
        labels = Variable(labels)
        label_list.extend(labels)
        labels = labels.cuda()
        images = images.cuda()
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = outputs.max(1)
        preds_list = np.concatenate((preds_list, outputs.data.cpu().numpy()))
        correct = preds.eq(labels).sum()
        Acc.update(correct.item(),labels.shape[0])
        n_iter = (epoch - 1) * len(dataLoader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)
        kappa = quadratic_weighted_kappa(outputs, labels)
        print(
            'Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.8f} Acc{acc:.4f} Kappa:{kappa:.4f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(dataLoader.dataset),
                kappa=kappa,
                acc=correct
            ))

        # update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)


    all_kappa = quadratic_weighted_kappa(preds_list,label_list)
    all_acc = Acc.avg
    print("all_kappa:",all_kappa,"all_acc:",all_acc)
    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)


def eval_training(epoch, dataLoader):
    net.eval()
    net._dropout.eval()
    test_loss = 0.0  # cost function error
    correct = 0.0
    preds_list = np.zeros(shape=(0, args.num_classes))
    label_list = []
    for (images, labels) in dataLoader:
        images = Variable(images)
        labels = Variable(labels)
        label_list.extend(labels)
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            outputs = net(images)
        preds_list = np.concatenate((preds_list, outputs.data.cpu().numpy()))
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    kappa = quadratic_weighted_kappa(preds_list, label_list)
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f} kappa:{kappa:.4f}'.format(
        test_loss / len(dataLoader.dataset),
        correct.float() / len(dataLoader.dataset), kappa=kappa
    ))
    print()

    # add informations to tensorboard
    writer.add_scalar('Test/Average loss', test_loss / len(dataLoader.dataset), epoch)
    writer.add_scalar('Test/Accuracy', correct.float() / len(dataLoader.dataset), epoch)

    return correct.float() / len(dataLoader.dataset),kappa,preds_list


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default="efficieNet-b7", help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=4, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=15, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.0003, help='initial learning rate')
    parser.add_argument('-scale_size', type=int, default=512, help='initial learning rate')
    parser.add_argument('-num_classes', type=float, default=5, help='initial learning rate')
    args = parser.parse_args()




    net = DVN(depth=7)
    net._avg_pooling = GeM()

    net_name = '/storage1/wangjie/checkpoint/efficieNet-b7/2020-03-23T23:16:06.994066/efficieNet-b7-11-best.pth'
    net_name = '/storage1/wangjie/checkpoint/efficieNet-b7/2020-03-24T01:58:24.847883/efficieNet-b7-12-best.pth'
    net_name = '/storage1/wangjie/checkpoint/efficieNet-b7/2020-04-12T20:23:41.645976/efficieNet-b7-7-best.pth'
    state_dict = torch.load(net_name)
    net.load_state_dict(state_dict)
    net.cuda()

    all_df = pd.read_csv("data/csv/dual_sub1_train.csv",header=None)
    all_df = all_df[:int(all_df.shape[0])].sample(frac=1,random_state=100)
    # presudo_df = pd.read_csv("data/csv/tp_val.csv",header=None)
    # presudo_df = pd.read_csv("data/csv/high_095.csv",header=None)
    # presudo_df= presudo_df[[0,1]]
    train_df = pd.concat([all_df],axis=0)
    val_df = pd.read_csv("data/csv/dual_sub1_val.csv",header=None)
    # val_df = pd.read_csv("/storage1/wangjie/isbi_new/regular-fundus-training/isbi_train_pre.csv")
    # val_df = val_df[:100]
    train_transform = transforms.Compose([
        transforms.Resize(((args.scale_size, args.scale_size))),
        transforms.RandomAffine(degrees=(-180, 180), scale=(0.85, 1.1), shear=(-36, 36)),
        transforms.ColorJitter(brightness=(0.7, 1.7), contrast=(0.7,1.3), saturation=(0.7,1.3),hue=0.1),
        MYCLAHE(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        Normalize_0_1(255),
        transforms.ToTensor(),
        transforms.Normalize(settings.CIFAR100_TRAIN_MEAN,
                             settings.CIFAR100_TRAIN_STD)
    ])
    train_loader = torch.utils.data.DataLoader(
        KaggleFundusDataset(train_df, settings.ibis_train_path_1024, train_transform),
        batch_size=args.b, shuffle=True,
        num_workers=2, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        KaggleFundusDataset_val(val_df, settings.ibis_val_path_1024, transforms.Compose([
            # transforms.RandomCrop(scale=(0.9, 1.0)),
            transforms.Resize((args.scale_size, args.scale_size)),
            Normalize_0_1(255),
            transforms.ToTensor(),
            transforms.Normalize(settings.CIFAR100_TRAIN_MEAN,
                                 settings.CIFAR100_TRAIN_STD)
        ])),
        batch_size=args.b, shuffle=False,
        num_workers=2, pin_memory=True)

    test_df = pd.read_csv("data/csv/sub1_and_2_test_dual_path.csv", header=None)
    test_loader = torch.utils.data.DataLoader(
        KaggleFundusDataset_val(test_df, settings.ibis_test_path_1024,transforms.Compose([
            # transforms.RandomCrop(scale=(0.9, 1.0)),
            transforms.Resize((args.scale_size, args.scale_size)),
            Normalize_0_1(255),
            transforms.ToTensor(),
            transforms.Normalize(settings.isbi100_train_mean,
                                 settings.isbi100_train_std)
        ])),
        batch_size=args.b, shuffle=False,
        num_workers=2, pin_memory=True)

    loss_function = nn.CrossEntropyLoss()
    #loss_function = nn.SmoothL1Loss()
    # optimizer =optim.SGD(net.parameters(), lr=args.lr, weight_decay=5e-4)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,40],
                                                     gamma=0.1)  # learning rate decay
    iter_per_epoch = len(train_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    # use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(args.b, 3, args.scale_size, args.scale_size).cuda()
    # writer.add_graph(net, Variable(input_tensor, requires_grad=True))

    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    best_kappa = 0.0

    # if epoch > args.warm:
    #     train_scheduler.step()

    #train(epoch, train_loader)
    acc ,kappa,pred_val= eval_training(0, val_loader)
    acc, kappa,pred_test = eval_training(1, test_loader)
    p_test = pred_test.argmax(1)
    upload_df = pd.read_csv('data/csv/Challenge1_upload.csv')
    # test_df['DR_Level'] = pred_list
    # test_df.to_csv("su1_uplod_1.csv", index=False)
    test_df[1]=p_test
    test_df[3] = p_test
    test_df.columns=['0','1','2','3']
    t1 = test_df[['0', '1']]
    t2 = test_df[['2', '3']]
    t1_ = pd.DataFrame()
    t2_ = pd.DataFrame()
    t1_['image_id'] = t1['0'].apply(lambda x: x.split('/')[-1].split('.')[0])
    t2_['image_id'] = t2['2'].apply(lambda x: x.split('/')[-1].split('.')[0])
    t1_['DR_Level'] = t1['1']
    t2_['DR_Level'] = t2['3']
    t12= pd.concat([t1_,t2_])
    t12_r = t12.reset_index(drop=True)
    final_up_df = pd.DataFrame(upload_df['image_id']).merge(t12_r, on='image_id')
    final_up_df.to_csv('result/sub_1.csv', index=False)
