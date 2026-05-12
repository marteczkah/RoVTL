import torch
import torch.nn as nn
from models.resnet import ResNET
from tarte_ai import TARTE_Pretrain_NN
from huggingface_hub import hf_hub_download
import json
import os

class ImageTabularPretraining(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.initialize_tabular_model()
        self.initialize_imaging_model()

    def initialize_imaging_model(self):
        self.image_encoder = ResNET(self.pretrain_configs, self.dim)
    
    def initialize_tabular_model(self):
        pretrain_weights, self.pretrain_configs = self.load_tarte_pretrain_model()
        self.tabular_encoder = TARTE_Pretrain_NN(
            dim_input=self.pretrain_configs["dim_input"],
            dim_transformer=self.pretrain_configs["dim_transformer"],
            dim_feedforward=self.pretrain_configs["dim_feedforward"],
            dim_projection=self.pretrain_configs["dim_projection"],  # desired projection dim for contrastive learning
            num_heads=self.pretrain_configs["num_heads"],
            num_layers_transformer=self.pretrain_configs["num_layers_transformer"],
            dropout=self.pretrain_configs["dropout"]
        )
        self.tabular_encoder.load_state_dict(pretrain_weights)
    
    def load_tarte_pretrain_model(self, device='cpu'):
        base_path = '/lustre/groups/iml/projects/marta/'
        cache_dir = os.path.join(base_path, "data/pretrained_weights")

        repo_id = 'inria-soda/tarte'

        # Load weights
        weights_file = 'tarte_pretrained_weights.pt'
        model_path = hf_hub_download(repo_id=repo_id, filename=weights_file, cache_dir=cache_dir)
        pretrain_weights = torch.load(model_path, map_location=device, weights_only=True)

        # Load configs
        config_file = 'tarte_pretrained_configs.json'
        config_path = hf_hub_download(repo_id=repo_id, filename=config_file, cache_dir=cache_dir)
        with open(config_path) as f:
            pretrain_model_configs = json.load(f)

        return pretrain_weights, pretrain_model_configs

    def forward(self, x, edge_attr, mask, img):
        tabular_projection = self.tabular_encoder(x, edge_attr, mask)
        img_projection, img_feature = self.image_encoder(img)
        return tabular_projection, img_projection, img_feature