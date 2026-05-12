import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import click
import wandb
from tqdm import tqdm
from dataset.MRDownstreamDataset import DownstreamCollator, MRDownstreamDatasetH5
from utils.utils import set_seed, map_categorical_values, get_bias
from dataset.attributes import REGRESSION_NUMERICAL_ATTRIBUTES, CLASSIFICATION_CATEGORICAL_ATTRIBUTES, LABEL_MAPPING, VAL_NUM_ATTRIBUTES, VALIDATION_CAT_ATTRIBUTES
from tarte_ai import TARTE_TablePreprocessor
from models.fusion import MutlimodalFusionRegression
from pytorch_lightning import seed_everything
torch.cuda.empty_cache()

ROOT_PATH = "./results"
wandb.init(
   project = "icml_reb_bmi"
)

@click.command()
@click.option('--csv_path_filter', '-p', help='Path to the csv that filters the h5 with classification training set.', required=False, default="/lustre/groups/shared/ukbb-87065/users/marta.hasny/train_regression_5k.csv")
@click.option('--h5_path_val', '-v', help='Path to the csv file with val data information.', required=False, default="/lustre/groups/shared/ukbb-87065-ext/users/marta.hasny/val.h5")
@click.option('--batch_size', '-b', help='Traning batch size.', default = 128, type = int)
@click.option('--epochs', '-e', help='Number of epochs.', default = 100, type = int)
@click.option('--store', '-s', help='Where you want to store the models and results.', required=True, type = str)
@click.option('--frozen', '-f', help='Whether the backbone should be frozen.', required=False, default=False, type = bool)
@click.option('--model', '-m', required=False, default="")
@click.option('--num_workers', '-w', required=False, default=8)
@click.option('--attribute', '-a', required=False, default='', type=str)
@click.option('--tabmofe_weight', '-l', required=False, default=1, type=float)
@click.option('--lr', required=False, default=3e-4, type=float)

def main(csv_path_filter, store, h5_path_val, batch_size, epochs, frozen, num_workers, model, attribute, tabmofe_weight, lr):
    print("TRAINING CLASSIFICATION")
    set_seed()
    seed_everything(2022, workers=True)
    # setup wandb to log the training information
    # setup the training data
    store = os.path.join(ROOT_PATH, store)
    os.makedirs(store, exist_ok=True)
    regres_folder = os.path.join(store, attribute + "_regression")

    os.makedirs(regres_folder, exist_ok=True)

    train_data = MRDownstreamDatasetH5(
        h5_path='/lustre/groups/shared/ukbb-87065-ext/users/marta.hasny/train.h5', 
        filter_csv=csv_path_filter, 
        simulate_missing=True,
        augmentation_rate=0.95,
        attribute=attribute)
    
    val_data = MRDownstreamDatasetH5(
        h5_path=h5_path_val, 
        filter_csv='/lustre/groups/shared/ukbb-87065-ext/users/marta.hasny/val.csv',
        simulate_missing=False,
        augmentation_rate=-1,
        attribute=attribute)

    tarte_tab_prepper = TARTE_TablePreprocessor()
    attributes = REGRESSION_NUMERICAL_ATTRIBUTES + CLASSIFICATION_CATEGORICAL_ATTRIBUTES
    df_train = pd.read_csv("/lustre/groups/shared/ukbb-87065-ext/users/marta.hasny/train.csv")[attributes]
    df_train = map_categorical_values(df_train, CLASSIFICATION_CATEGORICAL_ATTRIBUTES, LABEL_MAPPING)
    tarte_tab_prepper.fit(df_train)
    collator = DownstreamCollator(tarte_tab_prepper) 
    
    train_loader = DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collator
    )
    val_loader = DataLoader(
        val_data,
        batch_size = batch_size,
        shuffle = False,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collator
    )

    regressor = MutlimodalFusionRegression(
        bias=get_bias(attribute),
        image_checkpoint=model,
        tabular_checkpoint=model,
        freeze_image=frozen,
        freeze_tabular=frozen
    )

    regressor.train()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        if torch.cuda.device_count() > 1:
            print("MULTIPLE CUDA DEVICES")
            regressor = torch.nn.DataParallel(regressor)  
            torch.backends.cudnn.benchmark = True
        regressor.to(device)
    else:
        device = torch.device("cpu")
   
    optimizer = torch.optim.AdamW(regressor.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.HuberLoss()
    evaluation_criterion = nn.L1Loss()

    for epoch in range(epochs):
        epoch_loss = []
        epoch_val_loss = []

        print("EPOCH: ", epoch)
        regressor.train()
        for data in tqdm(train_loader):
            optimizer.zero_grad()
            logits, img_logits, tab_logits = regressor(data['x'].to(device), 
                                data['edge_attr'].to(device),
                                data['mask'].to(device),
                                data['image'].float().to(device))
            img_unimodal = criterion(img_logits.squeeze()[:batch_size].squeeze(), data['attribute'][:batch_size].float().to(device).squeeze())
            tab_unimodal = criterion(tab_logits[:batch_size].squeeze(), data['attribute'][:batch_size].float().to(device).squeeze())
            unimodal_loss = img_unimodal + tab_unimodal
            unimodal_loss.backward(retain_graph=True)
            for name, parms in regressor.named_parameters():
                layer = str(name).split('.')[0]
                if 'mutlimodal' in layer:
                    parms.grad = None
            loss_more = criterion(logits[:batch_size].squeeze(), data['attribute'][:batch_size].float().to(device).squeeze())
            loss_less = criterion(logits[batch_size:].squeeze(), data['attribute'][batch_size:].float().to(device).squeeze())
            tabmofe_loss = torch.maximum(loss_more - loss_less, torch.tensor(0.0, device=device))
            loss = loss_more + loss_less + tabmofe_weight * tabmofe_loss
            loss.backward()
            nn.utils.clip_grad_norm_(regressor.parameters(), max_norm=40, norm_type=2)
            optimizer.step()
            epoch_loss.append(loss.detach().cpu().numpy())
            wandb.log({"batch_train_loss: ": loss})
        wandb.log({"epoch_train_loss: ": np.mean(np.array(epoch_loss))})

        mae_imgs = []
        mae_tabs = []
        regressor.eval()
        for data in tqdm(val_loader):
            with torch.no_grad():
                logits, _, _ = regressor(data['x'].to(device), 
                                    data['edge_attr'].to(device),
                                    data['mask'].to(device),
                                    data['image'].float().to(device))
                val_loss = criterion(logits[:batch_size].squeeze(), data['attribute'][:batch_size].float().to(device).squeeze())
                mae_tab = evaluation_criterion(logits[:batch_size].squeeze(), data['attribute'][:batch_size].float().to(device).squeeze())
                mae_tabs.append(mae_tab.detach().cpu().numpy())
                mae_img = evaluation_criterion(logits[batch_size:].squeeze(), data['attribute'][batch_size:].float().to(device).squeeze())
                mae_imgs.append(mae_img.detach().cpu().numpy())
                epoch_val_loss.append(val_loss.detach().cpu().numpy())
                wandb.log({"batch_val_loss": val_loss})
        
        torch.save(regressor.state_dict(), os.path.join(regres_folder, 'last.pth'))
                
        wandb.log({
            "epoch_val_loss": np.mean(epoch_val_loss),
            "mae_img": np.mean(mae_imgs),
            "mae_tab": np.mean(mae_tabs),
        })

        torch.save(regressor.state_dict(), os.path.join(regres_folder, 'last' + str(epoch) + '.pth'))
    
if __name__ == '__main__':
    main()