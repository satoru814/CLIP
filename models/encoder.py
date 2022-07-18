import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from config import CFG
from transformers import DistilBertModel, DistilBertConfig


class ImageEncoder(nn.Module):
    def __init__(self, model_name=CFG.image_encoder_model, pretrained=True, trainable=CFG.trainable):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained, num_classes=0)
        for p in self.model.parameters():
            p.requires_grad = trainable
    def forward(self, x):
        return self.model(x)


class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=True, trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad = trainable
        self.target_token_idx = 0
    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids, attention_mask)
        last_hidden_state = out.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim=CFG.projection_dim, dropout=0.3):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(embedding_dim, projection_dim),
                                        nn.GELU(),
                                        nn.Linear(projection_dim, projection_dim),
                                        nn.Dropout(dropout),
                                        nn.LayerNorm(projection_dim)
        )
    def forward(self, x):
        out = self.projection(x)
        return out


class CLIPModel(nn.Module): 
    def __init__(self,temperature=CFG.temperature, image_embedding=CFG.image_embedding, text_embedding=CFG.text_embedding):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature
    def forward(self, img, cap_idx, atten_msk):
        image_features = self.image_encoder(img)
        text_features = self.text_encoder(
            input_ids=cap_idx, attention_mask=atten_msk)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)
        return image_embeddings, text_embeddings