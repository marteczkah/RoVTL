import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import click
import wandb
from tqdm import tqdm
from torchmetrics import AUROC
from dataset.MRDownstreamDataset import DownstreamCollator, MRDownstreamDatasetH5
from utils.utils import set_seed, map_categorical_values
from dataset.attributes import NUMERICAL_ATTRIBUTES, CLASSIFICATION_CATEGORICAL_ATTRIBUTES, LABEL_MAPPING, VAL_NUM_ATTRIBUTES, VALIDATION_CAT_ATTRIBUTES
from tarte_ai import TARTE_TablePreprocessor
from models.fusion import MutlimodalFusionClassifier
from pytorch_lightning import seed_everything
torch.cuda.empty_cache()

ROOT_PATH = "./results"
wandb.init(
   project = "rovtl"
)

@click.command()
@click.option('--csv_path_filter', '-p', help='Path to the csv that filters the h5 with classification training set.', required=False, default="/lustre/groups/shared/ukbb-87065/users/marta.hasny/train_classifier2.csv")
@click.option('--h5_path_val', '-v', help='Path to the csv file with val data information.', required=False, default="/lustre/groups/shared/ukbb-87065-ext/users/marta.hasny/val.h5")
@click.option('--batch_size', '-b', help='Traning batch size.', default = 64, type = int)
@click.option('--epochs', '-e', help='Number of epochs.', default = 100, type = int)
@click.option('--store', '-s', help='Where you want to store the models and results.', required=True, type = str)
@click.option('--frozen', '-f', help='Whether the backbone should be frozen.', required=False, default=True, type = bool)
@click.option('--model', '-m', required=False, default="")
@click.option('--num_workers', '-w', required=False, default=8)
@click.option('--tabmofe_weight', '-l', required=False, default=1, type=float)
@click.option('--lr', '-l', help='Learning rate.', required=False, default=1e-4, type = float)

def main(csv_path_filter, store, h5_path_val, batch_size, epochs, frozen, num_workers, model, tabmofe_weight, lr):
    print("TRAINING CLASSIFICATION")
    set_seed()
    seed_everything(2022, workers=True)

    store = os.path.join(ROOT_PATH, store)
    os.makedirs(store, exist_ok=True)
    classifier_folder = os.path.join(store, "multilabel_cad")
    os.makedirs(classifier_folder, exist_ok=True)

    train_data =MRDownstreamDatasetH5(
        h5_path='/lustre/groups/shared/ukbb-87065-ext/users/marta.hasny/train.h5', 
        filter_csv=csv_path_filter, 
        simulate_missing=True,
        sample_more=0.3,
        augmentation_rate=0.95)
    
    val_data = MRDownstreamDatasetH5(
        h5_path=h5_path_val, 
        filter_csv='/lustre/groups/shared/ukbb-87065-ext/users/marta.hasny/val.csv',
        simulate_missing=False,
        selected_num=VAL_NUM_ATTRIBUTES,
        selected_cat=VALIDATION_CAT_ATTRIBUTES,
        augmentation_rate=-1)

    tarte_tab_prepper = TARTE_TablePreprocessor()
    attributes = NUMERICAL_ATTRIBUTES + CLASSIFICATION_CATEGORICAL_ATTRIBUTES
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

    classifier = MutlimodalFusionClassifier(
        num_classes=4,
        image_checkpoint=model,
        tabular_checkpoint=model,
        freeze_image=frozen,
        freeze_tabular=frozen,
        dim=3
    )

    classifier.train()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        if torch.cuda.device_count() > 1:
            print("MULTIPLE CUDA DEVICES")
            classifier = torch.nn.DataParallel(classifier)  
            torch.backends.cudnn.benchmark = True
        classifier.to(device)
    else:
        device = torch.device("cpu")
   
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    average = 'macro'
    auc_tab = AUROC(num_labels=4, average=average, task="multilabel")
    auc_img = AUROC(num_labels=4, average=average, task="multilabel")
    auc_train = AUROC(num_labels=4, average=average, task="multilabel")


    for epoch in range(epochs):
        epoch_loss = []
        epoch_val_loss = []

        print("EPOCH: ", epoch)
        classifier.train()
        for data in tqdm(train_loader):
            optimizer.zero_grad()
            logits, img_logits, tab_logits = classifier(data['x'].to(device), 
                                data['edge_attr'].to(device),
                                data['mask'].to(device),
                                data['image'].float().to(device))
            preds = torch.sigmoid(logits.detach())
            auc_train.update(preds.cpu(), data['label'].cpu().long())  
            img_unimodal = criterion(img_logits.squeeze()[:batch_size], data['label'][:batch_size].to(device))
            tab_unimodal = criterion(tab_logits[:batch_size], data['label'][:batch_size].to(device))
            unimodal_loss = img_unimodal + tab_unimodal
            unimodal_loss.backward(retain_graph=True)
            for name, parms in classifier.named_parameters():
                layer = str(name).split('.')[0]
                if 'mutlimodal' in layer:
                    parms.grad = None
            loss_more = criterion(logits[:batch_size], data['label'][:batch_size].to(device))
            loss_less = criterion(logits[batch_size:], data['label'][batch_size:].to(device))
            tabmofe_loss = torch.maximum(loss_more - loss_less, torch.tensor(0.0, device=device))
            loss = loss_more + loss_less + tabmofe_weight*tabmofe_loss
            loss.backward()
            nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=40, norm_type=2)
            optimizer.step()
            epoch_loss.append(loss.detach().cpu().numpy())
            wandb.log({"batch_train_loss: ": loss})
        auc_train_metric = auc_train.compute()
        wandb.log({"epoch_train_loss: ": np.mean(np.array(epoch_loss)),
                   "train_auc" : auc_train_metric})

        classifier.eval()
        for data in tqdm(val_loader):
            with torch.no_grad():
                logits, _, _ = classifier(data['x'].to(device), 
                                    data['edge_attr'].to(device),
                                    data['mask'].to(device),
                                    data['image'].float().to(device))
                val_loss = criterion(logits[:batch_size], data['label'][:batch_size].to(device))
                preds = torch.sigmoid(logits.detach())
                auc_tab.update(preds[:batch_size].cpu(), data['label'][:batch_size].cpu().long())  
                auc_img.update(preds[batch_size:].cpu(), data['label'][batch_size:].cpu().long())  
                epoch_val_loss.append(val_loss.detach().cpu().numpy())
                wandb.log({"batch_val_loss": val_loss})
                        
        auc_tab_metric = auc_tab.compute()
        auc_img_metric = auc_img.compute()

        wandb.log({
            "epoch_val_loss": np.mean(epoch_val_loss),
            "auc_img": auc_img_metric.item(),
            "auc_tab": auc_tab_metric.item(),
        })

        torch.save(classifier.state_dict(), os.path.join(classifier_folder, 'epoch' + str(epoch) + '.pth'))
        
        auc_tab.reset()
        auc_img.reset()

if __name__ == '__main__':
    main()