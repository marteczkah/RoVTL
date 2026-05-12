import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import random
import os
import torch
from dataset.attributes import DVM_ATTRIBUTES, DVM_NUMERICAL

class DVMDataset(Dataset):
    def __init__(self, csv_path, simulate_missing = True, attribute="", augmentation_rate=-1, root='/lustre/groups/shared/ukbb-87065/dataset/cardiac_mri_nifti'):
        self.root = root
        self.info = pd.read_csv(csv_path)
        self.eid = self.info['Adv_ID'].to_numpy()
        self.paths = self.info['Path'].to_numpy()

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
        self.attribute = attribute
        self.simulate_missing = simulate_missing

        if len(attribute) > 0:
            attribute_ind = list(self.metadata_columns).index(attribute)
            self.attributes = self.metadata[:, attribute_ind]

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
            chosen = random.sample(DVM_ATTRIBUTES, 
                                    k=random.randint(1, len(DVM_ATTRIBUTES)))
        else:
            chosen = DVM_ATTRIBUTES
        
        selected_tab = {}
        for col in chosen:
            val = tabular[col]
            if not pd.isna(val):
                if col in DVM_NUMERICAL:
                    val /= 1000
                selected_tab[col] = val 
            else:
                selected_tab[col] = np.nan

        return {"img": image, 
                'tabular': selected_tab,
                "eid": self.eid[index]}
    
class DVMContrastiveCollator:
    def __init__(self, data_prep_module):
        self.data_prep_module = data_prep_module

    def __call__(self, batch):
        # 1. Rebuild a DataFrame for tabular features
        tab = [b["tabular"] for b in batch]
        df_batch = pd.DataFrame(tab)

        # 2. Run TARTE preprocessor
        preprocessed = self.data_prep_module.transform(df_batch)
        # preprocessed is a list of tuples: (idx, x, edge_attr, mask, y)

        # 3. Unpack
        idxs, xs, edge_attrs, masks, ys = zip(*preprocessed)

        # 4. Stack into tensors
        xs = torch.stack(xs)
        edge_attrs = torch.stack(edge_attrs)
        masks = torch.stack(masks)

        # 5. Collect images
        images = torch.stack([b["img"] for b in batch])


        return {
            "x": xs,                      # node features
            "edge_attr": edge_attrs,      # edge features
            "mask": masks,                # attention mask
            "image": images,              # imaging input
            "eid": [b["eid"] for b in batch]
        }     