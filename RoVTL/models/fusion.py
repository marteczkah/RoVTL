'''
* Licensed under the Apache License, Version 2.
* By Siyi Du, 2024
# Adapted by Marta Hasny, 2025
* Based on Vision Transformer and BERT
* Based on AViT https://github.com/siyi-wind/AViT/blob/main/Models/Transformer/ViT_adapters.py
* Based on BLIP https://github.com/salesforce/BLIP/blob/main/models/med.py
'''
from typing import Optional
from timm.models.layers import DropPath
import torch
import os
import torch.nn as nn
from models.resnet_tokens import ResNetTokens
from tarte_ai import TARTE_Base
from huggingface_hub import hf_hub_download
import json

class DotDict(dict):
    '''
    enable to use dot to search the dict
    dict = {'name':cici}
    dotdict = DotDict(dict)
    dotdict.name
    '''
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    if isinstance(v, dict):
                        v = DotDict(v)
                    if isinstance(v, list):
                        self.__convert(v)
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                if isinstance(v, dict):
                    v = DotDict(v)
                elif isinstance(v, list):
                    self.__convert(v)
                self[k] = v

    def __convert(self, v):
        for elem in range(0, len(v)):
            if isinstance(v[elem], dict):
                v[elem] = DotDict(v[elem])
            elif isinstance(v[elem], list):
                self.__convert(v[elem])

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(DotDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(DotDict, self).__delitem__(key)
        del self.__dict__[key]

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.save_attention = False
        self.save_gradients = False

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map

    def forward(self, x, mask=None, visualize=False):
        x = x.squeeze()
        if len(x.shape) != 3:
            x = x.unsqueeze(0)
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn + mask

        attn = attn.softmax(dim=-1)
        if self.save_attention:
            self.save_attention_map(attn)
        if self.save_gradients:
            attn.register_hook(self.save_attn_gradients)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        if visualize == False:
            return x
        else:
            return x, attn

class CrossAttention(nn.Module):
    def __init__(self, q_dim, k_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = k_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        
        self.kv_proj = nn.Linear(k_dim, k_dim * 2, bias=qkv_bias)
        self.q_proj = nn.Linear(q_dim, k_dim) 
        
        self.proj = nn.Linear(k_dim, k_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.save_attention = False
        self.save_gradients = False

    def forward(self, q: torch.Tensor, k: torch.Tensor, tabular_mask: Optional[torch.Tensor] = None, visualize: bool = False):
        """
        Performs the Multi-Head Cross-Attention forward pass.

        Args:
            q (torch.Tensor): The query sequence (e.g., tokens) of shape [B, N_q, K].
            k (torch.Tensor): The key/value sequence (e.g., tabular features) of shape [B, N_k, K].
            tabular_mask (torch.Tensor, optional): The padding mask for keys [B, N_k].
                                                    True means the position is masked/ignored.
        """
        B, N_k, K = k.shape
        _, N_q, _ = q.shape
        C_h = K // self.num_heads # Dimension of each head

        
        kv = self.kv_proj(k).reshape(B, N_k, 2, self.num_heads, C_h).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  
        
        q = self.q_proj(q).reshape(B, N_q, self.num_heads, C_h).permute(0, 2, 1, 3)  # q: [B, H, N_q, C_h]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        fully_masked_samples = None
        if tabular_mask is not None:
            mask_expanded = tabular_mask[:, None, None, :].to(torch.bool)
            fully_masked_samples = (tabular_mask.sum(dim=1) == N_k)

            # Apply the mask: set scores to -infinity where the mask is True
            attn = attn.masked_fill(mask_expanded, float('-inf'))

        attn = attn.softmax(dim=-1)
        
        if fully_masked_samples is not None and fully_masked_samples.any():
            mask_to_zero_attn = fully_masked_samples[:, None, None, None]
            attn = attn.masked_fill(mask_to_zero_attn, 0.0)

        if self.save_attention:
            self.save_attention_map(attn)
        if self.save_gradients:
            attn.register_hook(self.save_attn_gradients)
            
        attn = self.attn_drop(attn)

        out = (attn @ v)
        
        out = out.transpose(1, 2).reshape(B, N_q, K)
        
        out = self.proj(out)
        out = self.proj_drop(out)
        
        if visualize == False:
            return out
        else:
            return out, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads=8, is_cross_attention=False, encoder_dim=None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.scale = 0.5
        self.norm1 = norm_layer(dim)
        self.is_cross_attention = is_cross_attention
        
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        
        if self.is_cross_attention:
            self.cross_attn = CrossAttention(
               q_dim=dim, k_dim=encoder_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
            )
            self.cross_norm = norm_layer(dim)
            self.cross_attn_gate = nn.Sequential(
                nn.Linear(1024, 526),
                nn.ReLU(),
                nn.Dropout(p=0.3),    
                nn.Linear(526, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid() 
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, encoder_hidden_states=None, tabular_mask: Optional[torch.Tensor] = None, visualize=False):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        
        if self.is_cross_attention:
            assert encoder_hidden_states is not None
            
            x_cross_in = x #
            
            if visualize == False:
                attn_out = self.cross_attn(
                    self.cross_norm(x_cross_in), 
                    encoder_hidden_states, 
                    tabular_mask=tabular_mask 
                )
            else:
                attn_out, cross_attn = self.cross_attn(
                    self.cross_norm(x_cross_in), 
                    encoder_hidden_states, 
                    tabular_mask=tabular_mask, 
                    visualize=visualize
                )

            gate_scalar = self.cross_attn_gate(encoder_hidden_states[:,0,:]).unsqueeze(1) 
            mask_gate = ~((tabular_mask.sum(dim=1) == tabular_mask.shape[-1])).unsqueeze(1).unsqueeze(1)
            x = x_cross_in + self.drop_path(mask_gate * gate_scalar * attn_out)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if visualize and self.is_cross_attention:
            return x, cross_attn
        else:
            return x

class MultimodalTransformerEncoder(nn.Module):
    '''
    Tabular Transformer Encoder based on BERT
    '''
    def __init__(self) -> None:
        super(MultimodalTransformerEncoder, self).__init__()
        self.image_proj = nn.Linear(2048, 1024)
        self.image_norm = nn.LayerNorm(1024)
        self.tabular_proj = nn.Linear(768, 1024)
        self.transformer_blocks = nn.ModuleList([
                            Block(dim=1024, is_cross_attention=True, 
                                  encoder_dim=1024) 
                            for _ in range(4)
                            ])
        self.norm =  nn.LayerNorm(1024)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            m.weight.data.normal_(mean=0.0, std=.02)
        elif isinstance(m, nn.LayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)
        if isinstance(m, nn.Linear) and m.bias is not None:
            m.bias.data.zero_()
    
    def forward(self, tab_feature: torch.Tensor, image_features: torch.Tensor, tab_mask: torch.Tensor, visualize=False) -> torch.Tensor:
        image_features = image_features.squeeze()
        image_features = self.image_proj(image_features)
        x = self.image_norm(image_features)
        tab_features = self.tabular_proj(tab_feature)
        if visualize == False:
            for i, transformer_block in enumerate(self.transformer_blocks):
                x = transformer_block(x, encoder_hidden_states=tab_features, tabular_mask=tab_mask)
            x = self.norm(x)
            return x
        else:
            attns = []
            for i, transformer_block in enumerate(self.transformer_blocks):
                x, attn = transformer_block(x, encoder_hidden_states=tab_features, tabular_mask=tab_mask, visualize=visualize)
                attns.append(attn)
            x = self.norm(x)
            return x, attns

class MutlimodalFusionClassifier(nn.Module):
    def __init__(self, num_classes, dim, image_checkpoint='', tabular_checkpoint='', freeze_image=False, freeze_tabular=False,num_modalities=2):
        super().__init__()
        self.num_modalities = num_modalities
        self.dim = dim
        self.image_checkpoint = image_checkpoint
        self.tabular_checkpoint = tabular_checkpoint
        self.num_classes = num_classes
        self.freeze_image = freeze_image
        self.freeze_tabular = freeze_tabular
        self.initialize_tabular_model()
        self.initialize_imaging_model()
        self.initialize_multimodal_model()
        self.class_head = nn.Linear(1024, num_classes)
    
    def initialize_multimodal_model(self):
        self.multimodal_encoder = MultimodalTransformerEncoder()

    def initialize_imaging_model(self):
        self.image_encoder = ResNetTokens(dim=self.dim)
        self.img_head = nn.Linear(2048, self.num_classes)
        if len(self.image_checkpoint) != 0:
            check = torch.load(self.image_checkpoint)
            check = {k.replace('image_encoder.', ''): v for k, v in check.items()}
            missing_keys, _ = self.image_encoder.load_state_dict(check, strict=False)
            if missing_keys:
                print('missing keys image encoder: ', missing_keys)
            if self.freeze_image:
                for p in self.image_encoder.parameters():
                    p.requires_grad = False
    
    def initialize_tabular_model(self):
        pretrain_weights, self.pretrain_configs = self.load_tarte_config()
        self.tabular_encoder = TARTE_Base(
            dim_input=self.pretrain_configs["dim_input"],
            dim_transformer=self.pretrain_configs["dim_transformer"],
            dim_feedforward=self.pretrain_configs["dim_feedforward"],
            num_heads=self.pretrain_configs["num_heads"],
            num_layers_transformer=self.pretrain_configs["num_layers_transformer"],
            dropout=self.pretrain_configs["dropout"]
        )
        self.tab_head = nn.Linear(768, self.num_classes)
        if len(self.tabular_checkpoint) != 0:
            print('loading tabular checkpoint')
            check = torch.load(self.tabular_checkpoint)
            check = {k.replace('tabular_encoder.tarte_base.', ''): v for k, v in check.items()}
            missing_keys, un = self.tabular_encoder.load_state_dict(check, strict=False)
            if missing_keys:
                print('missing keys tabular encoder: ', missing_keys)
            if self.freeze_tabular:
                for p in self.tabular_encoder.parameters():
                    p.requires_grad = False
        else:
            pretrain_weights = {k.replace('tabular_encoder.tarte_base.', ''): v for k, v in pretrain_weights.items()}
            missing_keys, _ = self.tabular_encoder.load_state_dict(pretrain_weights, strict=False)
            # if missing_keys:
            #     print('missing keys tabular encoder: ', missing_keys)

    
    def load_tarte_config(self):
        base_path = '/lustre/groups/iml/projects/marta/'
        cache_dir = os.path.join(base_path, "data/pretrained_weights")
        repo_id = 'inria-soda/tarte'
        # Load weights
        weights_file = 'tarte_pretrained_weights.pt'
        model_path = hf_hub_download(repo_id=repo_id, filename=weights_file, cache_dir=cache_dir)
        pretrain_weights = torch.load(model_path, map_location='cpu', weights_only=True)
        # Load configs
        config_file = 'tarte_pretrained_configs.json'
        config_path = hf_hub_download(repo_id=repo_id, filename=config_file, cache_dir=cache_dir)
        with open(config_path) as f:
            pretrain_model_configs = json.load(f)

        return pretrain_weights, pretrain_model_configs

    def forward(self, x, edge_attr, mask, img):
        # Encode features + logits
        img_feature, img_tokens = self.image_encoder(img)          # [B, 2048], [B, C]
        tab_feature = self.tabular_encoder(x, edge_attr, mask)  # [B, 768], [B, C]
        is_fully_masked = (mask[:,1:].sum(dim=1) == mask[:,1:].shape[-1]) # [B] tensor of True/False
        mask[:, 0] = is_fully_masked
        if len(img_feature.shape) < 3:
            img_feature = img_feature.unsqueeze(0)
        img_logits = self.img_head(img_feature)
        tab_logits = self.tab_head(tab_feature[:,0,:])
        img_feature = img_feature.detach()
        img_tokens = img_tokens.detach()
        tab_feature = tab_feature.detach()
        multi_feature = self.multimodal_encoder(tab_feature, img_tokens, mask)
        if multi_feature.shape[0] == 1:
            logits = self.class_head(multi_feature.mean(dim=1))
        else:
            logits = self.class_head(multi_feature.squeeze().mean(dim=1))
        return logits, img_logits, tab_logits

class MutlimodalFusionRegression(nn.Module):
    def __init__(self, bias, dim=3, image_checkpoint='', tabular_checkpoint='', freeze_image=False, freeze_tabular=False,num_modalities=2):
        super().__init__()
        self.dim = dim
        self.num_modalities = num_modalities
        self.image_checkpoint = image_checkpoint
        self.tabular_checkpoint = tabular_checkpoint
        self.freeze_image = freeze_image
        self.freeze_tabular = freeze_tabular
        self.bias = bias
        self.initialize_tabular_model()
        self.initialize_imaging_model()
        self.initialize_multimodal_model()
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(1024, 1)
        )        
        self.regressor[-1].bias.data[0] = bias

    def initialize_multimodal_model(self):
        self.multimodal_encoder = MultimodalTransformerEncoder()

    def initialize_imaging_model(self):
        self.image_encoder = ResNetTokens(self.dim)
        self.img_head = nn.Linear(2048, 1)
        self.img_head.bias.data[0] = self.bias
        if len(self.image_checkpoint) != 0:
            check = torch.load(self.image_checkpoint)
            check = {k.replace('image_encoder.', ''): v for k, v in check.items()}
            missing_keys, _ = self.image_encoder.load_state_dict(check, strict=False)
            if missing_keys:
                print('missing keys image encoder: ', missing_keys)
            if self.freeze_image:
                for p in self.image_encoder.parameters():
                    p.requires_grad = False
    
    def initialize_tabular_model(self):
        self.pretrain_configs = self.load_tarte_config()
        self.tabular_encoder = TARTE_Base(
            dim_input=self.pretrain_configs["dim_input"],
            dim_transformer=self.pretrain_configs["dim_transformer"],
            dim_feedforward=self.pretrain_configs["dim_feedforward"],
            num_heads=self.pretrain_configs["num_heads"],
            num_layers_transformer=self.pretrain_configs["num_layers_transformer"],
            dropout=self.pretrain_configs["dropout"]
        )
        self.tab_head = nn.Linear(768,1)
        self.tab_head.bias.data[0] = self.bias
        if len(self.tabular_checkpoint) != 0:
            check = torch.load(self.tabular_checkpoint)
            check = {k.replace('tabular_encoder.tarte_base.', ''): v for k, v in check.items()}
            missing_keys, un = self.tabular_encoder.load_state_dict(check, strict=False)
            if missing_keys:
                print('missing keys tabular encoder: ', missing_keys)
            
    
    def load_tarte_config(self):
        base_path = '/lustre/groups/iml/projects/marta/'
        cache_dir = os.path.join(base_path, "data/pretrained_weights")
        repo_id = 'inria-soda/tarte'
        # Load configs
        config_file = 'tarte_pretrained_configs.json'
        config_path = hf_hub_download(repo_id=repo_id, filename=config_file, cache_dir=cache_dir)
        with open(config_path) as f:
            pretrain_model_configs = json.load(f)

        return pretrain_model_configs

    def forward(self, x, edge_attr, mask, img):
        # Encode features + logits
        img_feature, img_tokens = self.image_encoder(img)          # [B, 2048], [B, C]
        tab_feature = self.tabular_encoder(x, edge_attr, mask)  # [B, 768], [B, C]
        is_fully_masked = (mask[:,1:].sum(dim=1) == mask[:,1:].shape[-1]) # [B] tensor of True/False
        mask[:, 0] = is_fully_masked
        if len(img_feature.shape) < 3:
            img_feature = img_feature.unsqueeze(0)
        img_logits = self.img_head(img_feature)
        tab_logits = self.tab_head(tab_feature[:,0,:])
        img_feature = img_feature.detach()
        img_tokens = img_tokens.detach()
        tab_feature = tab_feature.detach()
        multi_feature = self.multimodal_encoder(tab_feature, img_tokens, mask)
        if multi_feature.shape[0] == 1:
            logits = self.regressor(multi_feature.mean(dim=1))
        else:
            logits = self.regressor(multi_feature.squeeze().mean(dim=1))
        return logits, img_logits, tab_logits