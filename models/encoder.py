import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from config import CFG
from transformers import DistiBertModel, DistilBertConfig


class ImageEncoder(nn.Module):
    def __init__(self, model_name=CFG.image_encoder_name, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained, num_classes=0)
    def forwrad(self, x):
        return self.model(x)

class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_nanme, pretrained=True):
        if pretrained:
            self.model = DistiBertModel.from_pretrained(model_name)
        self.target_token_idx = 0
    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)


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
    def __init__(self,temperature, image_embedding=CFG.image_embedding, text_embedding=CFG.text_embedding):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature
    def forward(self, x):
        image_features = self.image_encoder(x["image"])
        text_features = self.text_encoder(
            input_ids=x["input_ids"], attention_mask=x["attention_mask"])
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.image_projection(text_features)
        return image_embeddings, text_embeddings