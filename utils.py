import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import cv2
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
    df["random"] = np.random.randint(0,10, size=len(image_list))
    df["train"] = (df["random"]!=CFG.VAL_SET)
    df.to_csv(os.path.join(CFG.DATA_PATH, "captions.csv"))
    return None


def make_assets():
    df = pd.read_csv(CFG.DF_PATH)
    imgs = df["image_paths"].values
    captions = df["caption"].values
    fig,ax = plt.subplots(3,2, figsize=(10, 8))
    fig.suptitle("Dataset images and captions")
    last_imgname = ""
    i = 0
    show_n = 0
    while show_n < 9:
        img_filename = imgs[i]
        caption = captions[i]
        if img_filename != last_imgname:
            img = cv2.imread(img_filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax[show_n//3, show_n%2].imshow(img)
            ax[show_n//3, show_n%2].set_title(caption, fontsize=7)
            ax[show_n//3, show_n%2].axis("off")
            show_n += 1
            last_imgname = img_filename
        else:
            i += 1
    plt.savefig("./assets/dataset.png")


#Dataset
class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, df_path=CFG.DF_PATH, tokenizer=None, transform=None, is_train=True):
        self.df = pd.read_csv(df_path)
        self.df = self.df[self.df["train"]==is_train]
        self.img_paths = self.df["image_paths"].values
        self.captions  = self.df["caption"].values
        self.encoded_captions = tokenizer(
            list(self.captions), padding=True, truncation=True, max_length=CFG.MAX_LENGTH
        )
        if is_train:
            self.transform = transform["train"]
        else:
            self.transform = transform["eval"]
    def __getitem__(self, idx):
        img_filename = self.img_paths[idx]
        img = cv2.imread(img_filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        caption = self.captions[idx]
        if self.transform:
            img = self.transform(image=img)["image"]
        cap_idx, atten_msk = self.encoded_captions["input_ids"][idx], self.encoded_captions["attention_mask"][idx]
        cap_idx = torch.tensor(cap_idx)
        atten_msk = torch.tensor(atten_msk)
        item = [img, caption, cap_idx, atten_msk, img_filename]
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


def find_matches(model, query, img_embeddings, filenames, device):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    encoded_query = tokenizer([query])
    input_ids = torch.tensor(encoded_query["input_ids"]).to(device)
    atten_msk = torch.tensor(encoded_query["attention_mask"]).to(device)
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=input_ids, attention_mask=atten_msk
        )
        text_embeddings = model.text_projection(text_features)

    img_embeddings_n = F.normalize(img_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)

    dot_similarity = text_embeddings_n @ img_embeddings_n.T
    values, indices = torch.topk(dot_similarity.squeeze(0), 50)
    matches = [filenames[idx] for idx in indices[::5]]
    fig, ax = plt.subplots(2,3)
    fig.suptitle(f"query : {query}", fontsize=10)
    for i, match_file in enumerate(matches):
        if i >= 6:
            break
        img = cv2.imread(match_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax[i//3, i%3].imshow(img)
    plt.savefig("./assets/inference.png")
    print("finish inference")
    # return model, img_embeddings, filenames


if __name__=="__main__":
    make_assets()