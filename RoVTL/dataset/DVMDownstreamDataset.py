import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import random
import os
import torch
from dataset.attributes import CLASSIFICATION_DVM_ATTRIBUTES, CLASSIFICATION_DVM_NUMERICAL

class DVMDownstreamDataset(Dataset):
    def __init__(self, csv_path, simulate_missing = True, augmentation_rate=-1, root='/lustre/groups/shared/ukbb-87065/dataset/cardiac_mri_nifti'):
        self.root = root
        self.info = pd.read_csv(csv_path)
        self.eid = self.info['Adv_ID'].to_numpy()
        self.paths = self.info['Path'].to_numpy()
        self.labels = self.info['label'].to_numpy() 

        img_size=128

        self.transform_augment = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=29, sigma=(0.1, 2.0))],p=0.5),
            transforms.RandomResizedCrop(size=(img_size,img_size), scale=(0.6, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize(size=(img_size,img_size)),
            transforms.Lambda(lambda x : x.float())
        ])
        self.transform = transforms.Compose([
            transforms.Resize(size=(128, 128)),
            transforms.Lambda(lambda x : x.float())
        ])

        self.augmentation_rate = augmentation_rate
        self.simulate_missing = simulate_missing


    def __len__(self):
        return len(self.eid)
    
    def __getitem__(self, index):
        path = self.paths[index]
        npy_path = os.path.splitext(path)[0] + ".npy"
        image = torch.from_numpy(np.load(npy_path))
        if random.random() <= self.augmentation_rate:
            image = self.transform_augment(image.permute(2, 0, 1)) / 255
        else:
            image = self.transform(image.permute(2, 0, 1)) / 255

        tabular = self.info.iloc[index]

        if self.simulate_missing:
            if random.random() < 0.3:
                chosen1 = random.sample(CLASSIFICATION_DVM_ATTRIBUTES, 
                        k=random.randint(1, int(0.2*(len(CLASSIFICATION_DVM_ATTRIBUTES)))))
                chosen2 = random.sample(chosen1, 
                                        k=random.randint(0, len(chosen1)))
            else:
                chosen1 = random.sample(CLASSIFICATION_DVM_ATTRIBUTES, 
                                        k=random.randint(1, len(CLASSIFICATION_DVM_ATTRIBUTES)))
                chosen2 = random.sample(chosen1, 
                                        k=random.randint(0, len(chosen1)))
        else:
            chosen1 = CLASSIFICATION_DVM_ATTRIBUTES
            chosen2 = []
        
        selected_tab1 = {}
        for col in chosen1:
            val = tabular[col]
            if not pd.isna(val):
                if col in CLASSIFICATION_DVM_NUMERICAL:
                    val /= 1000
                selected_tab1[col] = val 
            else:
                selected_tab1[col] = np.nan
                        
        selected_tab2 = {}
        for col in chosen2:
            val = tabular[col]
            if not pd.isna(val):
                if col in CLASSIFICATION_DVM_NUMERICAL:
                    val /= 1000
                selected_tab2[col] = val 
            else:
                selected_tab2[col] = np.nan

        label = self.labels[index]

        return {"img": image, 
                'tabular1': selected_tab1,
                'tabular2': selected_tab2,
                'label' : torch.tensor(label).float(),
                "eid": self.eid[index]}

class DVMDownstreamCollator:
    def __init__(self, data_prep_module):
        self.data_prep_module = data_prep_module

    def __call__(self, batch):
        tabular_list = []
        for b in batch:
            tabular_list.append(b["tabular1"])
        for b in batch:
            tabular_list.append(b["tabular2"])

        df_combined = pd.DataFrame(tabular_list)

        all_expected_cols = (
            getattr(self.data_prep_module, "num_col_names_", []) +
            getattr(self.data_prep_module, "cat_col_names_", [])
        )
        for col in all_expected_cols:
            if col not in df_combined.columns:
                df_combined[col] = np.nan  # fill missing with NaN

        # --- Preprocess once ---
        preprocessed = self.data_prep_module.transform(df_combined)
        idxs, xs, edge_attrs, masks, _ = zip(*preprocessed)
        xs, edge_attrs, masks = map(torch.stack, (xs, edge_attrs, masks))

        # --- Duplicate images and labels accordingly ---
        images = torch.stack([b["img"] for b in batch])
        images = torch.cat([images, images], dim=0)  # match combined rows

        labels = torch.stack([b["label"] for b in batch])
        labels = torch.cat([labels, labels], dim=0)

        eids = [b["eid"] for b in batch]
        eids = eids + eids  # duplicate list

        return {
            "x": xs,                  # node features
            "edge_attr": edge_attrs,  # edge features
            "mask": masks,            # attention mask
            "image": images,          # duplicated images
            "eid": eids,              # duplicated eids
            "label": labels
        }