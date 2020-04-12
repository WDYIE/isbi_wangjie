# train.py
# !/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""
import DualViewNet
import os
import sys
import argparse
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
from sklearn import metrics
from utils import quadratic_weighted_kappa

from  albumentations import CLAHE,MedianBlur,GaussNoise,Blur
from tensorboardX import SummaryWriter
from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR,load_model_by_name
from kaggleFundusDataset_for_sub3 import KaggleFundusDataset, Normalize_0_1,MYCLAHE,KaggleFundusDataset_sub3_test
from efficientnet_pytorch import EfficientNet
from gm_pooling import GeM
from utils import AverageMeter
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def fix_predict(list):
    count = len(list)//2
    new_list = []
    for index,count in enumerate(range(count)):
        one = list[index*2]
        two = list[index*2+1]
        if abs(one-two)==1:
            v= min(one,two)
            new_list.append(v)
            new_list.append(v)
        else:
            new_list.append(one)
            new_list.append(two)
    return new_list
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
    y_pred = np.argmax(preds_list, axis=1)
    y_pred_fix = fix_predict(y_pred)
    fix_correct = torch.Tensor(y_pred_fix)
    t_label_list = torch.Tensor(label_list)
    fix_correct = fix_correct.eq(t_label_list).sum()
    fix_kappa=metrics.cohen_kappa_score(y_pred_fix, label_list, weights='quadratic')
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f} kappa:{kappa:.4f} fix_kappa:{fix_kappa:.4f} fix_Accuracy  {fix_Accuracy:.4f}'.format(
        test_loss / len(dataLoader.dataset),
        correct.float() / len(dataLoader.dataset), kappa=kappa,fix_kappa = fix_kappa,fix_Accuracy = fix_correct/len(dataLoader.dataset)
    ))
    print()

    # add informations to tensorboard
    writer.add_scalar('Test/Average loss', test_loss / len(dataLoader.dataset), epoch)
    writer.add_scalar('Test/Accuracy', correct.float() / len(dataLoader.dataset), epoch)

    return correct.float() / len(dataLoader.dataset),kappa,preds_list


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default="efficieNet-b7_sub3", help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=8, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=15, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.0003, help='initial learning rate')
    parser.add_argument('-scale_size', type=int, default=800, help='initial learning rate')
    parser.add_argument('-num_classes', type=float, default=5, help='initial learning rate')
    args = parser.parse_args()

    # net = res2next_dla60(pretrained=False, num_classes=5)


    net = EfficientNet.from_pretrained('efficientnet-b7',num_classes=5 )
    #net._avg_pooling = GeM()


    net_name = "/storage1/wangjie/checkpoint/efficieNet-b7_sub3/2020-03-17T03:04:11.464360/efficieNet-b7_sub3-164-best.pth"
    #net_name = '/home/wangjie/python3/pytorchcifa100/checkpoint/efficieNet-b7_sub3/2020-03-17T00:50:32.885983/efficieNet-b7_sub3-81-best.pth'
    #Test set: Average loss: 0.2231, Accuracy: 0.7200 kappa:0.8609
    #net_name = "/home/wangjie/python3/pytorchcifa100/checkpoint/efficieNet-b7_sub3/2020-03-16T01:09:31.085562/efficieNet-b7_sub3-123-best.pth"
    # 0.7875  0.8850 size 800  isbi 训练集微调 验证集得分
    #net_name = '/storage1/wangjie/efficieNet-b7/2020-03-11T13:07:27.154008/efficieNet-b7-6-best.pth'
    #73 达到了.74
    #net_name = "/home/wangjie/python3/pytorchcifa100/checkpoint/efficieNet-b7_sub3/2020-03-15T22:22:47.239481/efficieNet-b7_sub3-70-regular.pth"
    #Test set: Average loss: 0.2042, Accuracy: 0.4600 kappa:0.6839
    # net_name = "/home/wangjie/python3/pytorchcifa100/checkpoint/efficieNet-b7_sub3/2020-03-16T22:48:46.042490/efficieNet-b7_sub3-35-best.pth"
    #net_name = '/home/wangjie/python3/pytorchcifa100/checkpoint/efficieNet-b7_sub3/2020-03-17T22:18:19.207503/efficieNet-b7_sub3-83-best.pth'

    #stage2 origianl size 512 now size 800
    #Test set: Average loss: 0.3191, Accuracy: 0.7000 kappa:0.7177
    #net_name = '/home/wangjie/python3/pytorchcifa100/checkpoint/efficieNet-b7_sub3/2020-03-18T00:21:04.272752/efficieNet-b7_sub3-136-best.pth'
    #net_name = '/storage1/wangjie/checkpoint/efficieNet-b7_sub3/2020-03-21T00:22:35.102215/efficieNet-b7_sub3-41-best-8657.pth'
    state_dict = torch.load(net_name)
    net.load_state_dict(state_dict)
    net.cuda()

    all_df = pd.read_csv("data/csv/ultra-widefield-training.csv",header=None)
    all_df = all_df[:int(all_df.shape[0])].sample(frac=1,random_state=100)
    train_df = all_df
    val_df = pd.read_csv("data/csv/ultra-widefield-validation.csv",header=None)
    test_df = pd.read_csv('data/csv/onsite_sub3.csv',header=None)

    resize = 1024
    crop = 800
    train_transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.RandomAffine(degrees=(-180, 180), scale=(0.9, 1.35), shear=(-30, 30)),
        transforms.CenterCrop((crop, crop)),
        MYCLAHE(),
        transforms.ColorJitter(brightness=(0.9, 1.5), contrast=(0.8, 1.2), saturation=(0.8, 1.2),hue=0.2),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomHorizontalFlip(0.5),
        Normalize_0_1(255),
        transforms.ToTensor(),
        transforms.Normalize(settings.CIFAR100_TRAIN_MEAN,
                             settings.CIFAR100_TRAIN_STD)
    ])
    train_loader = torch.utils.data.DataLoader(
        KaggleFundusDataset(train_df, settings.isbi_sub3_train_path, train_transform),
        batch_size=args.b, shuffle=True,
        num_workers=1, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        KaggleFundusDataset(val_df, settings.isbi_sub3_val_path, transforms.Compose([
            transforms.Resize((resize, resize)),
            # transforms.RandomAffine(degrees=(-180, 180), scale=(0.9, 1.35), shear=(-15, 15)),
            transforms.CenterCrop((crop,crop)),
            Normalize_0_1(255),
            transforms.ToTensor(),
            transforms.Normalize(settings.CIFAR100_TRAIN_MEAN,
                                 settings.CIFAR100_TRAIN_STD)
        ])),
        batch_size=args.b, shuffle=False,
        num_workers=1, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        KaggleFundusDataset_sub3_test(test_df, settings.isbi_sub3_val_path, transforms.Compose([
            transforms.Resize((resize, resize)),
            # transforms.RandomAffine(degrees=(-180, 180), scale=(0.9, 1.35), shear=(-15, 15)),
            transforms.CenterCrop((crop,crop)),
            Normalize_0_1(255),
            transforms.ToTensor(),
            transforms.Normalize(settings.CIFAR100_TRAIN_MEAN,
                                 settings.CIFAR100_TRAIN_STD)
        ])),
        batch_size=args.b, shuffle=False,
        num_workers=1, pin_memory=True)

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
    acc, kappa, preds_val = eval_training(2, val_loader)
    acc, kappa, preds_test = eval_training(2, test_loader)

    p_test = preds_test.argmax(1)
    test_df['p_test'] = p_test
    sub3_upload_df = pd.read_csv('data/csv/Challenge3_upload.csv')
    p_test = preds_test.argmax(1)
    test_df['p_test'] = fix_predict(p_test)
    updatadir= '/storage1/wangjie/sub3_isbi/Onsite-Challenge3-Evaluation/Challenge3_upload.csv'
    sub3_upload_df = pd.read_csv(updatadir)
    final_upload = pd.DataFrame()
    final_upload['image_id'] = test_df[0].apply(lambda x: x.split('/')[-1].split('.')[0])
    final_upload['DR_level'] = fix_predict(p_test)

    final_sub3_upload = pd.DataFrame(sub3_upload_df['image_id']).merge(final_upload, on='image_id')

    final_sub3_upload.to_csv('result/sub3_8656_4_12.csv', index=False)

