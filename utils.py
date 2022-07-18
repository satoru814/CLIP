import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import sys
import cv2
import json
from concurrent import futures
import torch
import torch.nn as nn
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer

from config import CFG


def cross_entropy(preds, targets, reduction="none"):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets*log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


def calc_loss(image_embeddings, text_embeddings, temperature=CFG.temperature):
    logits = (text_embeddings @ image_embeddings.T)/temperature
    images_similarity = image_embeddings @ image_embeddings.T
    texts_similarity = text_embeddings @ text_embeddings.T
    targets = F.softmax((images_similarity + texts_similarity)/(2*temperature), dim=-1)
    texts_loss = cross_entropy(logits, targets, reduction="none")
    images_loss = cross_entropy(logits.T, targets.T, reduction="none")
    loss = (images_loss + texts_loss)/2
    return loss.mean()


def make_dataframe():
    df = pd.read_csv(os.path.join(CFG.DATA_PATH, "captions.txt"))
    image_list = df["image"]
    image_paths = [os.path.join(CFG.DATA_PATH ,"Images" , val) for val in image_list]
    df["image_paths"] = image_paths
    df.to_csv(os.path.join(CFG.DATA_PATH, "captions.csv"))
    return None


#Dataset
class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, df_path=CFG.DF_PATH, tokenizer=None, transform=None, is_train=True):
        self.df = pd.read_csv(df_path)
        self.imgs_paths = self.df["image_paths"]
        self.captions  = self.df["caption"]
        self.encoded_captions = tokenizer(
            list(self.captions), padding=True, truncation=True, max_length=CFG.MAX_LENGTH
        )
        if is_train:
            self.transform = transform["train"]
        else:
            self.transform = transform["eval"]
    def __getitem__(self, idx):
        img = cv2.imread(self.imgs_paths[idx])
        # img = img.transpose(2,0,1)
        print(img.shape)
        caption = self.captions[idx]
        if self.transform:
            img = self.transform(image=img)["image"]
        cap_idx, atten_msk = self.encoded_captions["input_ids"][idx], self.encoded_captions["attention_mask"][idx]
        cap_idx = torch.tensor(cap_idx)
        atten_msk = torch.tensor(atten_msk)
        item = [img, caption, cap_idx, atten_msk]
        return item
    def __len__(self):
        return len(self.captions)


def get_transforms():
    trans = {"train": A.Compose([
                A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
                A.Normalize(max_pixel_value=255.0),
                ToTensorV2(),
             ]),
             "eval":A.Compose([
                A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
                A.Normalize(max_pixel_value=255.0),
                ToTensorV2(),
             ])
    }
    return trans

def get_tokenizer():
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    return tokenizer
# if __name__=="__main__":
#     make_dataframe()