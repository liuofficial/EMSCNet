from PIL import Image
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform
    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]
        img1 = self.transform(img)
        return img1, label















#
#
#
#
#
#
#
# import cv2
# import numpy as np
# from PIL import Image
# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms
#
#
# class MyDataSet(Dataset):
#     """自定义数据集"""
#
#     def __init__(self, images_path: list, images_class: list, transform=None,transform1=None):
#         self.images_path = images_path
#         self.images_class = images_class
#         self.transform = transform
#         self.transform1 = transform1
#         self.trans_totensor=transforms.ToTensor()
#
#     def __len__(self):
#         return len(self.images_path)
#
#     def __getitem__(self, item):
#         img = Image.open(self.images_path[item])
#         # RGB为彩色图片，L为灰度图片
#         if img.mode != 'RGB':
#             raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
#         label = self.images_class[item]
#         img = self.transform(img)
#
#         augimg=img
#         augimg = cv2.Canny(np.asarray(augimg), 30, 100)
#         augimg=self.trans_totensor(augimg)
#
#         if self.transform1 is not None:
#             img=self.transform1(img)
#         return img, augimg, label
#
