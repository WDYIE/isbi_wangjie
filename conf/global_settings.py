""" configurations for this project

author baiyu
"""
import os
from datetime import datetime

#CIFAR100 dataset path (python version)
#CIFAR100_PATH = '/nfs/private/cifar100/cifar-100-python'

#mean and std of cifar100 dataset
CIFAR100_TRAIN_MEAN = (0.5561324774050247, 0.5379583426082835, 0.526830592435949)
CIFAR100_TRAIN_STD = (0.1684706886447583, 0.15760600407010472, 0.12018643413347851)
#CIFAR100_TEST_MEAN = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
#CIFAR100_TEST_STD = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)

isbi100_train_mean = (0.43795229 ,0.2699121 , 0.16215418)
isbi100_train_std = (0.29552393, 0.19735483,0.13967223)
#directory to save weights file
CHECKPOINT_PATH = '/storage1/wangjie/checkpoint'

#total training epoches
EPOCH = 200
MILESTONES = [10,60,120]

#initial learning rate
#INIT_LR = 0.1

#time of we run the script
TIME_NOW = datetime.now().isoformat()

#tensorboard log dir
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10


Data_path =''


ibis_train_path_1024 = "/storage1/wangjie/isbi_new/isbi_train_crop_1024"
ibis_val_path_1024 = "/storage1/wangjie/isbi_new/isbi_val_crop_1024"
ibis_test_path_1024 = '/storage1/wangjie/isbi_new/isbi_test_crop_1024'


##seg form 1024
ibis_train_seg_512 = "/storage1/wangjie/isbi_new/isbi_train_seg_512"
ibis_val_seg_512 = "/storage1/wangjie/isbi_new/isbi_val_seg_512"
ibis_test_seg_512 = "/storage1/wangjie/isbi_new/isbi_test_seg_512"

ibis_train_path_512 = "/storage1/wangjie/isbi_train_crop"
ibis_val_path_512 = "/storage1/wangjie/isbi_val_crop"

#sub3_crop
isbi_sub3_train_path = "/storage1/wangjie/sub3_isbi"
isbi_sub3_val_path = "/storage1/wangjie/sub3_isbi"
#Data_path ='/storage/wangjie/kaggle_fundus_data/predata'







