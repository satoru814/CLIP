import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import utils
from models.encoder import CLIPModel

import argparse
import time
import os
import sys
from itertools import chain
import matplotlib as plt
import numpy as np
import wandb
from config import CFG
import argparse

class CLIP():
    def __init__(self, args):
        # self.data_dir = args.data_dir
        self.wandb_key = args.wandb_key
        self.save_weight = args.save_weight
        if torch.cuda.is_available():
            self.device="cuda"
        else:
            self.device="cpu"


    def build_model(self):
        self.Net = CLIPModel().to(self.device)
        #paramsa
        self.params = [
            {"params": self.Net.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
            {"params": self.Net.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
            {"params": chain(
                self.Net.image_projection.parameters(), self.Net.text_projection.parameters()
            ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
        ]
        #set optimizer
        self.optimizer = torch.optim.AdamW(self.params, weight_decay=0.)

        #dataset
        trans = utils.get_transforms()
        tokenizer = utils.get_tokenizer()
        train_dataset =  utils.CLIPDataset(CFG.DF_PATH, tokenizer, trans, is_train=True)
        val_dataset =  utils.CLIPDataset(CFG.DF_PATH, tokenizer, trans, is_train=False)
        self.train_loader = DataLoader(train_dataset, **CFG.dataloader.train)
        self.val_loader = DataLoader(val_dataset, **CFG.dataloader.val)


    def train(self):
        #wandb
        if self.wandb_key:
            run = wandb.init(**CFG.wandb, settings=wandb.Settings(code_dir="."))
            wandb.watch(models=(self.Net), log_freq=100)

        print(len(self.val_loader), len(self.train_loader))

        for epoch in range(CFG.EPOCH):
            self.Net.train()
            losses = {"cross_entropy" : 0}
            iteration = 0
            print("epoch",epoch)
            for i , item in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                
                img, cap_idx, atten_msk = item[0].to(self.device).float(), item[2].to(self.device), item[3].to(self.device)
                img_embs, text_embs  = self.Net(img, cap_idx, atten_msk)

                loss =  utils.calc_loss(img_embs, text_embs)

                loss.backward()
                self.optimizer.step()

                #loss
                losses["cross_entropy"] += loss.item()

                #metric calc
                img = img.detach().cpu().numpy()
                iteration += 1

            losses["cross_entropy"] /= iteration

            if self.wandb_key:
                wandb.log(losses)

        self.save()

        if self.wandb_key:
            run.finish()


    def inference(self, query=CFG.test_query ,weight=None):
        if weight:
            self.Net.load(weight)
        img_embeddings = []
        filenames = []
        self.Net.eval()
        with torch.no_grad():
            for i, item in enumerate(self.val_loader):
                img = item[0].to(self.device).float()
                filename = item[4]
                img_features = self.Net.image_encoder(img)
                img_embedding = self.Net.image_projection(img_features)
                img_embeddings.append(img_embedding)
                filenames += filename
            img_embeddings = torch.cat(img_embeddings)
        print("find_match call")
        utils.find_matches(self.Net, query, img_embeddings, filenames, self.device)


    def save(self):
        torch.save(self.Net.state_dict(), CFG.MODEL_SAVE_PATH)