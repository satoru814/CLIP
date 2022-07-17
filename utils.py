import torch
import torch.nn as nn

import os
import cv2
import albumentations as A
import config as CFG

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, transforms=None):
        


def get_transforms(mode="train"):
    if mode == "train":
        trans = A.Compose([

        ])

