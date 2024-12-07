from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import clip

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class MogoClip(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 layers: int,
                 heads: int,
                 width: int,
                 codebook_size: int,
                 max_motion_length: int,
                 clip_version: str,
                 device
                ):
        super().__init__()
        
        self.max_motion_length = max_motion_length
        self.clip_version = clip_version
        self.device = device
        
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            attn_mask=self.build_attention_mask()
        )
        
        
        self.codebook_size = codebook_size
        self.token_embedding = nn.Embedding(codebook_size, width)
        self.clip_linear = nn.Linear(768, embed_dim)
        self.positional_embedding = nn.Parameter(torch.empty(self.max_motion_length, width))
        self.ln_final = LayerNorm(width)
        self.text_projection = nn.Parameter(torch.empty(width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.2))
        self.clip_model = self.load_and_freeze_clip()
        
        self.initialize_parameters()

    
    
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.clip_linear.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)


        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
            
    def load_and_freeze_clip(self):
        clip_model, clip_preprocess = clip.load(self.clip_version, device=self.device,
                                                jit=False)  # Must set jit=False for training
        # Cannot run on cpu
        clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
        # Date 0707: It's necessary, only unecessary when load directly to gpu. Disable if need to run on cpu

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model
            
    
    def build_attention_mask(self):
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.max_motion_length, self.max_motion_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    
    def encode_motion_code(self, motion_code):
        x = self.token_embedding(motion_code) # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x.mean(dim=1) @ self.text_projection

        return x
    
    def encode_text(self, raw_text):
        text = clip.tokenize(raw_text, truncate=True).to(self.device)
        feat_clip_text = self.clip_model.encode_text(text).float()
        text_embed = self.clip_linear(feat_clip_text)
        return text_embed
    
    
    # def mean_cosine_similarity(self, motion_code, raw_text):
    #     text_features = self.encode_text(raw_text)
    #     motion_features = self.encode_motion_code(motion_code).type(text_features.dtype)
    #     motion_features = motion_features / motion_features.norm(dim=-1, keepdim=True)
    #     text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    #     cosine_sim = torch.sum(motion_features * text_features, dim=-1)  # [batch_size]
    #     return cosine_sim.mean().item()
    
    def mean_cosine_similarity(self, motion_code, raw_text):
        """
        计算正样本对的平均余弦相似度，并监控正负样本的分离情况。
        
        Args:
            motion_code (torch.Tensor): 动作特征编码。
            raw_text (List[str]): 文本特征对应的描述。
        
        Returns:
            dict: 包含正样本相似度均值、负样本相似度均值和两者差值。
        """
        # 编码特征
        text_features = self.encode_text(raw_text)
        motion_features = self.encode_motion_code(motion_code).type(text_features.dtype)

        # 特征归一化
        motion_features = motion_features / motion_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 对角线（正样本对相似度）
        cosine_sim_matrix = motion_features @ text_features.t()  # [batch_size, batch_size]
        positive_sim = cosine_sim_matrix.diag()  # 正样本对
        positive_mean = positive_sim.mean().item()

        # 非对角线（负样本对相似度）
        batch_size = cosine_sim_matrix.size(0)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=cosine_sim_matrix.device)  # 非对角线 mask
        negative_sim = cosine_sim_matrix[mask]  # 提取负样本
        negative_mean = negative_sim.mean().item()

        # 计算正负样本分离程度
        separation = positive_mean - negative_mean

        # 返回监控结果
        return positive_mean, negative_mean, separation
        # return {
        #     "positive_mean": positive_mean,
        #     "negative_mean": negative_mean,
        #     "separation": separation
        # }

    
    
    def forward(self, motion_code, raw_text):
        
        text_features = self.encode_text(raw_text)
        motion_features = self.encode_motion_code(motion_code).type(text_features.dtype)
        
        # normalized features
        motion_features = motion_features / motion_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_motion = logit_scale * motion_features @ text_features.t()
        logits_per_text = logits_per_motion.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_motion, logits_per_text
        
