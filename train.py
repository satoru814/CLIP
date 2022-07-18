import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import utils
from models.encoder import CLIPModel


import numpy as np
import argparse
import time
import os
import sys
from itertools import chain
import matplotlib as plt
from PIL import Image
import numpy as np
import wandb
from config import CFG
import random
import argparse


def set_requires_grad(models, requires=False):
    if not isinstance(models, list):
        models = [models]
    for model in models:
        if model is not None:
            for param in model.parameters():
                param.requires_grad = requires


def main():
    #argparse
    parser = argparse.ArgumentParser(description='CLIP')
    parser.add_argument("--wandb", "-w", action="store_true", default=False, help="True -> wandb log is on ")
    args = parser.parse_args()


    if torch.cuda.is_available():
        device="cuda"
    else:
        device="cpu"

    
    Net = CLIPModel().to(device)

    #set loss functions
    # cross_entropy = nn.CrossEntropyLoss()

    #set optimizer
    optimizer = torch.optim.Adam(Net.parameters(), **CFG.optimizer)

    #dataset
    trans = utils.get_transforms()
    tokenizer = utils.get_tokenizer()
    train_dataset =  utils.CLIPDataset(CFG.DF_PATH, tokenizer, trans, is_train=True)
    eval_dataset =  utils.CLIPDataset(CFG.DF_PATH, tokenizer, trans, is_train=False)
    train_loader = DataLoader(train_dataset, **CFG.dataloader.train)

    #wandb
    try:
        api_key = user_secrets.get_secret("WANDB")
        wandb.login(CFG.wandb_key)
        anonymous = None
    except:
        anonymous = "must"
    if args.wandb:
        run = wandb.init(**CFG.wandb, settings=wandb.Settings(code_dir="."))
        wandb.run.log_code(".")
        wandb.watch(models=(Net), log_freq=100)

        
    print("data_size:",len(train_loader))
    for epoch in range(CFG.EPOCH):
        Net.train()
        losses = {"cross_entropy" : 0}
        iter = 0
        print("epoch",epoch)
        for i , item in enumerate(train_loader):
            optimizer.zero_grad()
            
            img, cap_idx, atten_msk = item[0].to(device).float(), item[2].to(device), item[3].to(device)
            # print(img.shape, cap_idx.shape, atten_msk.shape)
            img_embs, text_embs  = Net(img, cap_idx, atten_msk)

            loss =  utils.calc_loss(img_embs, text_embs)

            loss.backward()
            optimizer.step()

            #loss
            losses["cross_entropy"] += loss.item()

            #metric calc
            img = img.detach().cpu().numpy()

            # iou = utils.calc_IOU(pred, msk)
            iter += 1
        losses["cross_entropy"] /= iter

        # utils.save_results(pred, msk, img, epoch)

        if args.wandb:
            wandb.log(losses)

    torch.save(Net.state_dict(), CFG.MODEL_SAVE_PATH)
    run.finish()


if __name__ == "__main__":
    main()
