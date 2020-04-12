from torch.utils.data import  Dataset
from skimage import io
import pandas as pd
import torch
from PIL import Image
import os
import numpy as np
import  cv2
from numpy import asarray
from  albumentations import CLAHE,MedianBlur,GaussNoise,Blur,Cutout
from utils import AverageMeter
from random import randint
class KaggleFundusDataset(Dataset):
    def __init__(self,pd_frame,root_dir,transform = True):
        self.landmarks_frame = pd_frame
        self.landmarks_frame =self.landmarks_frame.fillna(value=0)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.landmarks_frame)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filepath = self.landmarks_frame.iloc[idx, 2]
        filepath = filepath.replace('\\','/')
        img_name = os.path.join(self.root_dir, filepath)
        # image = cv2.imread(img_name)
        # image = image/255
        # image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image  = Image.open(img_name)
        # img  = asarray(image,dtype=float)
        # img = img/255
        # print(filepath)
        if self.transform:
            image = self.transform(image)
        image = image.float()
        label = self.landmarks_frame.iloc[idx, 3]
        label = int(label)
        # labels = np.zeros(5,dtype=int)
        # labels[label] = 1
        sample = {'image': image, 'label': label}
        return (image,label)

class KaggleFundusDataset_sub3_test(Dataset):
    def __init__(self,pd_frame,root_dir,transform = True):
        self.landmarks_frame = pd_frame
        self.landmarks_frame =self.landmarks_frame.fillna(value=0)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        return len(self.landmarks_frame)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filepath = self.landmarks_frame.iloc[idx, 0]
        filepath = filepath.replace('\\','/')
        img_name = os.path.join(self.root_dir, filepath)
        # image = cv2.imread(img_name)
        # image = image/255
        # image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image  = Image.open(img_name)
        # img  = asarray(image,dtype=float)
        # img = img/255
        # print(filepath)
        if self.transform:
            image = self.transform(image)
        image = image.float()
        label = 0
        label = int(label)
        # labels = np.zeros(5,dtype=int)
        # labels[label] = 1
        sample = {'image': image, 'label': label}
        return (image,label)

class Normalize_0_1(object):
    """Normalize a tensor image with mean and standard deviation.
    """

    def __init__(self,num):
        self.num = num
    def __call__(self, img):
        """Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        """
        img = asarray(img)
        img = img/self.num
        return img
    def __repr__(self):
        return self.__class__.__name__
class MYCLAHE(object):
    """Normalize a tensor image with mean and standard deviation.
    """

    def __init__(self,num=0):
        self.num = num
    def __call__(self, img):
        """Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        """
        if randint(0,9)%3==0:
            img = CLAHE()(image =np.array(img))
            img = img["image"]
            img = Image.fromarray(img)
        if randint(0,9)%3==0:
            img = Blur()(image =np.array(img))
            img = img["image"]
            img = Image.fromarray(img)
        if randint(0,9)%3==0:
            img = Cutout()(image =np.array(img))
            img = img["image"]
            img = Image.fromarray(img)
        return img
    def __repr__(self):
        return self.__class__.__name__


class RandomResizedCrop(object):
    """Crop the given PIL Image to random size and aspect ratio.
    A crop of random size of (0.08 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR,
                 min_area_ratio=0.08, aspect_ratio=4./3):
        self.size = (size, size)
        self.interpolation = interpolation
        self.min_area_ratio = min_area_ratio
        self.aspect_ratio = aspect_ratio

    def get_params(self, img):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(self.min_area_ratio, 1.0) * area
            aspect_ratio = random.uniform(
                1 / self.aspect_ratio, self.aspect_ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback
        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, w, w

    def __call__(self, img, *args):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly cropped and resize image.
        """
        i, j, h, w = self.get_params(img)
        return (resized_crop(img, i, j, h, w, self.size, self.interpolation),
                *args)


