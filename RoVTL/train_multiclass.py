import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import click
import wandb
from tqdm import tqdm
from torchmetrics import Accuracy
from dataset.DVMDownstreamDataset import DVMDownstreamDataset, DVMDownstreamCollator
from utils.utils import set_seed
from tarte_ai import TARTE_TablePreprocessor
from models.fusion import MutlimodalFusionClassifier
from pytorch_lightning import seed_everything
torch.cuda.empty_cache()

CLASSIFICATION_DVM_ATTRIBUTES = [
    'color',
    'registration year',
    'body type',
    'mileage miles',
    'price',
    'fuel type',
    'gearbox',
    'engine size',
    'seat count', 
    'door count'
]

ROOT_PATH = "./results"
wandb.init(
   project = "rovtl"
)

@click.command()
@click.option('--csv_path_train', '-p', help='Path to the csv file with train data information.', required=False, default="/lustre/groups/iml/data/other/dvm/features/train_til.csv")
@click.option('--csv_path_val', '-v', help='Path to the csv file with val data information.', required=False, default="/lustre/groups/iml/data/other/dvm/features/val_til.csv")
@click.option('--batch_size', '-b', help='Traning batch size.', default = 64, type = int)
@click.option('--epochs', '-e', help='Number of epochs.', default = 100, type = int)
@click.option('--store', '-s', help='Where you want to store the models and results.', required=True, type = str)
@click.option('--frozen', '-f', help='Whether the backbone should be frozen.', required=False, default=True, type = bool)
@click.option('--model', '-m', required=False, default="")
@click.option('--num_workers', '-w', required=False, default=8)
@click.option('--loss_weight', '-l', required=False, default=1, type=float)
@click.option('--lr', '-l', help='Learning rate.', required=False, default=1e-4, type = float)

def main(csv_path_train, store, csv_path_val, batch_size, epochs, frozen, num_workers, model, loss_weight, lr):
    print("TRAINING CLASSIFICATION")
    set_seed()
    # seed_everything(2022, workers=True)
    # setup wandb to log the training information
    # setup the training data
    store = os.path.join(ROOT_PATH, store)
    os.makedirs(store, exist_ok=True)
    classifier_folder = os.path.join(store, "dvm_classification")
    os.makedirs(classifier_folder, exist_ok=True)

    train_data = DVMDownstreamDataset(
        csv_path_train,
        simulate_missing=True,
        augmentation_rate=0.95)
    
    val_data = DVMDownstreamDataset(
        csv_path_val,
        simulate_missing=False,
        augmentation_rate=-1)

    tarte_tab_prepper = TARTE_TablePreprocessor()
    df_train = pd.read_csv(csv_path_train)[CLASSIFICATION_DVM_ATTRIBUTES]
    df_train['mileage miles'] /= 1000
    df_train['registration year'] /= 1000
    tarte_tab_prepper.fit(df_train)
    collator = DVMDownstreamCollator(tarte_tab_prepper) 
    
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
        num_classes=286,
        image_checkpoint=model,
        tabular_checkpoint=model,
        freeze_image=frozen,
        freeze_tabular=frozen,
        dim=2
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
    criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,      # one full cosine cycle over all epochs
        eta_min=1e-6       # minimum learning rate
    )

    acc_tab = Accuracy(num_classes=286, task="multiclass")
    acc_img = Accuracy(num_classes=286, task="multiclass")
    acc_train = Accuracy(num_classes=286, task="multiclass")

    for epoch in range(epochs):
        epoch_loss = []
        epoch_val_loss = []

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"EPOCH: {epoch} | LR: {current_lr:.2e}")

        classifier.train()
        for data in tqdm(train_loader):
            optimizer.zero_grad()
            logits, img_logits, tab_logits = classifier(data['x'].to(device), 
                                data['edge_attr'].to(device),
                                data['mask'].to(device),
                                data['image'].float().to(device))
            preds = torch.softmax(logits.detach(), dim=1)
            acc_train.update(preds.cpu(), data['label'].cpu().long())  
            img_unimodal = criterion(img_logits.squeeze()[:batch_size], data['label'][:batch_size].to(device).long())
            tab_unimodal = criterion(tab_logits[:batch_size], data['label'][:batch_size].to(device).long())
            unimodal_loss = img_unimodal + tab_unimodal
            unimodal_loss.backward(retain_graph=True)
            for name, parms in classifier.named_parameters():
                layer = str(name).split('.')[0]
                if 'mutlimodal' in layer:
                    parms.grad = None
            loss_more = criterion(logits[:batch_size], data['label'][:batch_size].to(device).long())
            loss_less = criterion(logits[batch_size:], data['label'][batch_size:].to(device).long())
            ml_loss = torch.maximum(loss_more - loss_less, torch.tensor(0.0, device=device))
            loss = loss_more + loss_less + loss_weight*ml_loss
            loss.backward()
            nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=40, norm_type=2)
            optimizer.step()
            epoch_loss.append(loss.detach().cpu().numpy())
            wandb.log({"batch_train_loss: ": loss})
        acc_train_metric = acc_train.compute()
        wandb.log({"epoch_train_loss: ": np.mean(np.array(epoch_loss)),
                   "train_acc" : acc_train_metric})

        classifier.eval()
        for data in tqdm(val_loader):
            with torch.no_grad():
                logits, _, _ = classifier(data['x'].to(device), 
                                    data['edge_attr'].to(device),
                                    data['mask'].to(device),
                                    data['image'].float().to(device))
                val_loss = criterion(logits[:batch_size], data['label'][:batch_size].to(device).long())
                preds = torch.softmax(logits.detach(), dim=1)
                acc_tab.update(preds[:batch_size].cpu(), data['label'][:batch_size].cpu().long())  
                acc_img.update(preds[batch_size:].cpu(), data['label'][batch_size:].cpu().long())  
                epoch_val_loss.append(val_loss.detach().cpu().numpy())
                wandb.log({"batch_val_loss": val_loss})
                        
        acc_tab_metric = acc_tab.compute()
        acc_img_metric = acc_img.compute()
        wandb.log({
            "epoch_val_loss": np.mean(epoch_val_loss),
            "acc_img": acc_img_metric.item(),
            "acc_tab": acc_tab_metric.item(),
        })
        scheduler.step()

        torch.save(classifier.state_dict(), os.path.join(classifier_folder, 'epoch' + str(epoch) + '.pth'))
        
        acc_tab.reset()
        acc_img.reset()

if __name__ == '__main__':
    main()