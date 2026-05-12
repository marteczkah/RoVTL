import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import click
import wandb
from tqdm import tqdm
from tarte_ai import TARTE_TablePreprocessor
from dataset.attributes import NUMERICAL_ATTRIBUTES, CATEGORICAL_ATTRIBUTES,DISEASE_MAPPING, LABEL_MAPPING, DVM_ATTRIBUTES
from models.imgtab_pretraining import ImageTabularPretraining
from dataset.MRContrastiveDatasetH5 import MRContrastiveDatasetH5, DataCollator
from dataset.DVMDataset import DVMDataset, DVMContrastiveCollator
from utils.clip_loss import CLIPLoss
from utils.utils import set_seed, map_categorical_values
from pytorch_lightning import seed_everything
torch.cuda.empty_cache()

ROOT_PATH = "./results"
wandb.init(
   project = "rovtl"
)

@click.command()
@click.option('--h5_path_train', '-p', help='Path to the csv file with train data information.', required=False, default="/lustre/groups/shared/ukbb-87065-ext/users/marta.hasny/train.h5")
@click.option('--h5_path_val', '-v', help='Path to the csv file with val data information.', required=False, default="/lustre/groups/shared/ukbb-87065-ext/users/marta.hasny/val.h5")
@click.option('--batch_size', '-b', required=False, default=512, type=int)
@click.option('--epochs', '-e', help='Number of epochs.', default = 100, type = int)
@click.option('--store', '-s', help='Where you want to store the models and results.', required=True, type = str)
@click.option('--lr', '-l', help='Learning rate.', required=False, default=1e-4, type = float)
@click.option('--augment', '-a', help='Percent of augmentations.', required=False, default=0.95, type = float)
@click.option('--missing_tabular', '-m', help='Whether to simulate missing tabular data.', required=False, default=True, type = bool)
@click.option('--num_workers', '-w', required=False, default=False, type = bool)
@click.option('--dataset', '-d', help="DVM or UKBB", required=False, default="UKBB", type = str)

def main(h5_path_train, h5_path_val, batch_size, epochs, store, lr, augment, missing_tabular, num_workers, dataset):
    set_seed()
    seed_everything(2022, workers=True)

    # prepare training folders
    store = os.path.join(ROOT_PATH, store)
    os.makedirs(store, exist_ok=True)
    model_folder = os.path.join(store, "models")
    os.makedirs(model_folder, exist_ok=True)

    # preparare data
    if dataset == 'UKBB':
        dim = 3
        attributes = NUMERICAL_ATTRIBUTES + CATEGORICAL_ATTRIBUTES
        df_train = pd.read_csv("/lustre/groups/shared/ukbb-87065-ext/users/marta.hasny/train.csv")[attributes]
        df_train = map_categorical_values(df_train, CATEGORICAL_ATTRIBUTES, LABEL_MAPPING)
        df_train = df_train.rename(columns=DISEASE_MAPPING)
        tarte_tab_prepper = TARTE_TablePreprocessor()
        tarte_tab_prepper.fit(df_train)
        collator = DataCollator(tarte_tab_prepper) 

        train_data = MRContrastiveDatasetH5(
            h5_path_train, 
            augmentation_rate=augment, 
            simulate_missing=missing_tabular
        )
        val_data = MRContrastiveDatasetH5(
            h5_path_val, 
            simulate_missing=False
        )
    elif dataset == "DVM":
        dim = 2
        attributes = DVM_ATTRIBUTES
        df_train = pd.read_csv(h5_path_train)[attributes]
        df_train['mileage miles'] /= 1000
        df_train['registration year'] /= 1000
        df_train['price'] /= 1000
        df_train['wheelbase'] /= 1000
        df_train['height'] /= 1000
        df_train['width'] /= 1000
        df_train['length'] /= 1000   
        tarte_tab_prepper = TARTE_TablePreprocessor()
        tarte_tab_prepper.fit(df_train)
        collator = DVMContrastiveCollator(tarte_tab_prepper) 

        train_data = DVMDataset(
            h5_path_train, 
            augmentation_rate=augment, 
            simulate_missing=missing_tabular)
            
        val_data = DVMDataset(
            h5_path_val, 
            augmentation_rate=0,
            simulate_missing=False)

    train_loader = DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = True,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=collator
    )
    val_loader = DataLoader(
        val_data,
        batch_size = batch_size,
        shuffle = True,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=collator
    )

    model = ImageTabularPretraining(dim=dim)

    if torch.cuda.is_available():
        print(torch.cuda.device_count())
        device = torch.device("cuda")
        if torch.cuda.device_count() > 1:
            print("2 CUDA DEVICES")
            model = torch.nn.DataParallel(model)  
            torch.backends.cudnn.benchmark = True
        model = model.to(device)  
    else:
        device = torch.device("cpu")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    clip_criterion = CLIPLoss(temperature=0.1)

    best_loss = float('inf')
    for param in model.parameters():
        param.requires_grad = True
    model.train()

    for epoch in range(epochs):
        epoch_loss = []
        epoch_val_loss = []

        print("EPOCH: ", epoch)
        for data in tqdm(train_loader):
            out1, out2, _ = model(data['x'].to(device), 
                                 data['edge_attr'].to(device),
                                 data['mask'].to(device),
                                 data['image'].float().to(device))
            loss = 0
            for i, o in enumerate(out1):
                loss += clip_criterion(o, out2[i])[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.detach().cpu().item())
            wandb.log({"batch_train_loss: ": loss})
        wandb.log({"epoch_train_loss: ": np.mean(np.array(epoch_loss))})
        torch.save(model.state_dict(), os.path.join(model_folder, 'last.pth'))

        for data in tqdm(val_loader):
            with torch.no_grad():
                out1, out2, _ = model(data['x'].to(device), 
                                    data['edge_attr'].to(device),
                                    data['mask'].to(device),
                                    data['image'].float().to(device))
                val_loss = 0
                for i, o in enumerate(out1):
                    val_loss += clip_criterion(o, out2[i])[0]
                epoch_val_loss.append(val_loss.detach().cpu().numpy())
                wandb.log({"batch_val_loss": val_loss})
        cur_loss = np.mean(np.array(epoch_val_loss))
        if cur_loss < best_loss:
            torch.save(model.state_dict(), os.path.join(model_folder, 'best.pth'))
            best_loss = cur_loss
        wandb.log({"epoch_val_loss": cur_loss})

if __name__ == '__main__':
    main()