# -*- coding: utf-8 -*-

'''
Preapre salient object segmentation data
'''

import os
import glob
from PIL import Image
import torch.utils.data as data
from torchvision.transforms import ToTensor, Compose, Resize, Normalize

from args import image_root, gt_root


class SalObjDataset(data.Dataset):
    '''Salient object segmentation dataset
    Args:
        image_root (string): Image root directory
        gt_root (string): Ground-truth segmentation mask root directory
        size (int): Dataset size
    Output:
        image: input image, a normalized rgb image
        gt: salient object segmentation ground-truth, a binary map
    '''
    def __init__(self, image_root, gt_root, size=None):
        self.images = sorted(glob.glob(os.path.join(image_root, '*.jpg')))
        self.gts = sorted(glob.glob(os.path.join(gt_root, '*.png')))
        self.size = size if size else len(self.images)
        self.transform = Compose([Resize((224, 224)), ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = Compose([Resize((224, 224)), ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        image = self.transform(image)
        gt = self.binary_loader(self.gts[index])
        gt = self.gt_transform(gt)
        # gt = gt.long().squeeze(0)
        return image, gt

    def rgb_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize=10, size=None, shuffle=True, num_workers=1, pin_memory=False):
    dataset = SalObjDataset(image_root, gt_root, size)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


if __name__ == '__main__':
    train_loader = get_loader(image_root, gt_root)
    print(len(train_loader))
    d = next(iter(train_loader))
    print(d[0].size(), d[1].size())
