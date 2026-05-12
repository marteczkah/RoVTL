import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import click
import wandb
from tqdm import tqdm
from torchmetrics import Recall, AUROC, Accuracy
from dataset.MMSDataset import MMSDataset, MMSCollator
from utils.utils import set_seed, map_categorical_values
from dataset.attributes import MNMS_ATTRIBUTES_CAT, MNMS_ATTRIBUTES_NUM, MNMS_MAPPING
from tarte_ai import TARTE_TablePreprocessor
from pytorch_lightning import seed_everything
from models.fusion import MutlimodalFusionClassifier
torch.cuda.empty_cache()

ROOT_PATH = "./results"
wandb.init(
   project = "rovtl"
)

@click.command()
@click.option('--csv_path', '-p', help='Path to the csv that filters the h5 with classification training set.', required=False, default="/lustre/groups/shared/ukbb-87065/users/marta.hasny/train_classifier2.csv")
@click.option('--csv_path_val', '-v', help='Path to the csv file with val data information.', required=False, default="/lustre/groups/shared/ukbb-87065-ext/users/marta.hasny/val.h5")
@click.option('--batch_size', '-b', help='Traning batch size.', default = 64, type = int)
@click.option('--epochs', '-e', help='Number of epochs.', default = 50, type = int)
@click.option('--store', '-s', help='Where you want to store the models and results.', required=True, type = str)
@click.option('--frozen', '-f', help='Whether the backbone should be frozen.', required=False, default=False, type = bool)
@click.option('--model', '-m', required=False, default="")
@click.option('--num_workers', '-w', required=False, default=8)
@click.option('--lr', '-l', help='Learning rate.', required=False, default=1e-4, type = float)

def main(csv_path, store, csv_path_val, batch_size, epochs, frozen, num_workers, model, lr):
    print("TRAINING CLASSIFICATION")
    set_seed()
    seed_everything(2022, workers=True)

    store = os.path.join(ROOT_PATH, store)
    os.makedirs(store, exist_ok=True)
    classifier_folder = os.path.join(store, "mms_classifier")
    os.makedirs(classifier_folder, exist_ok=True)

    num_columns = MNMS_ATTRIBUTES_NUM
    cat_columns = MNMS_ATTRIBUTES_CAT

    train_data = MMSDataset(
        csv_path=csv_path,
        cat_columns=cat_columns,
        num_columns=num_columns,
        simulate_missing=True,
        augmentation_rate=0.95,
        classification=True,
        class_type='multiclass'
    )
    
    val_data = MMSDataset(
        csv_path=csv_path_val,
        cat_columns=cat_columns,
        num_columns=num_columns,
        simulate_missing=False,
        augmentation_rate=0,
        classification=True,
        class_type='multiclass'
    )

    tarte_tab_prepper = TARTE_TablePreprocessor()
    attributes = num_columns + cat_columns
    df_train = pd.read_csv(csv_path)[attributes]
    df_train = map_categorical_values(df_train, MNMS_ATTRIBUTES_CAT, MNMS_MAPPING)
    tarte_tab_prepper.fit(df_train)
    collator = MMSCollator(tarte_tab_prepper) 
    
    train_loader = DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = True,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collator
    )
    val_loader = DataLoader(
        val_data,
        batch_size = batch_size,
        shuffle = False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collator
    )

    classifier = MutlimodalFusionClassifier(
        num_classes=5,
        image_checkpoint=model,
        tabular_checkpoint=model,
        freeze_image=frozen,
        freeze_tabular=frozen,
        dim = 3
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

    average = 'macro'
    auc_tab = AUROC(num_classes=5, average=average, task="multiclass")
    auc_img = AUROC(num_classes=5, average=average, task="multiclass")
    acc_tab = Accuracy(num_classes=5, task="multiclass")
    acc_img = Accuracy(num_classes=5, task="multiclass")
    rec_tab = Recall(num_classes=5, average=average, task="multiclass")
    rec_img = Recall(num_classes=5, average=average, task="multiclass")
    auc_train = AUROC(num_classes=5, average=average, task="multiclass")

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
            b = logits.shape[0] // 2
            preds = torch.softmax(logits.detach(), dim=1)
            auc_train.update(preds.cpu(), data['label'].cpu().long())  
            img_unimodal = criterion(img_logits.squeeze()[:b], data['label'][:b].to(device))
            tab_unimodal = criterion(tab_logits[:b], data['label'][:b].to(device))
            unimodal_loss = img_unimodal + tab_unimodal
            unimodal_loss.backward(retain_graph=True)
            for name, parms in classifier.named_parameters():
                layer = str(name).split('.')[0]
                if 'mutlimodal' in layer:
                    parms.grad = None
            loss_more = criterion(logits[:b], data['label'][:b].to(device))
            loss_less = criterion(logits[b:], data['label'][b:].to(device))
            tabmofe_loss = torch.maximum(loss_more - loss_less, torch.tensor(0.0, device=device))
            loss = loss_more + loss_less + tabmofe_loss
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
                b = logits.shape[0]
                val_loss = criterion(logits, data['label'].to(device))
                preds = torch.softmax(logits.detach(), dim=1)
                auc_tab.update(preds[:b//2].cpu(), data['label'][:b//2].cpu().long())  
                acc_tab.update(preds[:b//2].cpu(), data['label'][:b//2].cpu().long())  
                rec_tab.update(preds[:b//2].cpu(), data['label'][:b//2].cpu().long())  
                auc_img.update(preds[b//2:].cpu(), data['label'][b//2:].cpu().long())  
                acc_img.update(preds[b//2:].cpu(), data['label'][b//2:].cpu().long())  
                rec_img.update(preds[b//2:].cpu(), data['label'][b//2:].cpu().long())  
                epoch_val_loss.append(val_loss.detach().cpu().numpy())
                wandb.log({"batch_val_loss": val_loss})
                        
        auc_tab_metric = auc_tab.compute()
        acc_tab_metric = acc_tab.compute()
        rec_tab_metric = rec_tab.compute()
        auc_img_metric = auc_img.compute()
        acc_img_metric = acc_img.compute()
        rec_img_metric = rec_img.compute()

        wandb.log({
            "epoch_val_loss": np.mean(epoch_val_loss),
            "auc_img": auc_img_metric.item(),
            "acc_img": acc_img_metric.item(),
            "rec_img": rec_img_metric.item(),
            "auc_tab": auc_tab_metric.item(),
            "acc_tab": acc_tab_metric.item(),
            "rec_tab": rec_tab_metric.item(),
        })

        torch.save(classifier.state_dict(), os.path.join(classifier_folder, 'epoch' + str(epoch) + '.pth'))
        
        auc_tab.reset()
        acc_tab.reset()
        rec_tab.reset()
        auc_img.reset()
        acc_img.reset()
        rec_img.reset()


if __name__ == '__main__':
    main()