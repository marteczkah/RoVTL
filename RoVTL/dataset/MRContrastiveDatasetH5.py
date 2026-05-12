import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import random
import torch
from utils.utils import normalize_scan
import h5py
from dataset.attributes import NUMERICAL_ATTRIBUTES, CATEGORICAL_ATTRIBUTES, LABEL_MAPPING, DISEASE_MAPPING

MODEL_ATTRIBUTES = [
    DISEASE_MAPPING.get(col, col)   # rename if mapping exists, otherwise keep original
    for col in CATEGORICAL_ATTRIBUTES
] + NUMERICAL_ATTRIBUTES

class MRContrastiveDatasetH5(Dataset):
    def __init__(self, h5_path, simulate_missing = True, augmentation_rate=-1, root='/lustre/groups/shared/ukbb-87065/dataset/cardiac_mri_nifti'):
        self.root = root
        self.info = h5py.File(h5_path, 'r')
        self.metadata_columns = self.info['metadata'].attrs['columns']
        self.metadata = np.array(self.info['metadata'])
        eid_ind = list(self.metadata_columns).index('eid')
        cad_ind = list(self.metadata_columns).index('no_cad')
        self.eid = self.info['metadata'][:, eid_ind]
        self.transform_augment = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(128, scale=(0.6, 1.0)),
                transforms.RandomRotation(45),
             ]
        )
        self.augmentation_rate = augmentation_rate
        self.simulate_missing = simulate_missing


    def __len__(self):
        return len(self.eid)
    
    def __getitem__(self, index):
        scan_org = self.info["images"][index]
        scan_org = normalize_scan(scan_org)
        if random.random() <= self.augmentation_rate:
            scan_org = torch.from_numpy(np.array(scan_org))
            scan_org = self.transform_augment(scan_org)
        else:
            scan_org = torch.from_numpy(np.array(scan_org))
        metadata_row = {col: self.metadata[index, i] for i, col in enumerate(self.metadata_columns)}

        if self.simulate_missing:
            chosen_cats = random.sample(CATEGORICAL_ATTRIBUTES, 
                                    k=random.randint(1, len(CATEGORICAL_ATTRIBUTES)))
            chosen_nums = random.sample(NUMERICAL_ATTRIBUTES, 
                        k=random.randint(1, len(NUMERICAL_ATTRIBUTES)))
        else:
            chosen_cats = CATEGORICAL_ATTRIBUTES
            chosen_nums = NUMERICAL_ATTRIBUTES
            
        categorical = {}
        for col in chosen_cats:
            val = metadata_row[col]
            if col in DISEASE_MAPPING.keys():
                new_col = DISEASE_MAPPING[col]
            else:
                new_col = col
            if not np.isnan(val):
                categorical[new_col] = LABEL_MAPPING[col][val]
            else:
                categorical[new_col] = np.nan

        continuous = {}
        for col in chosen_nums:
            val = metadata_row[col]
            if not np.isnan(val):
                continuous[col] = float(val)
            else:
                continuous[col] = np.nan

        return {"scan": scan_org, 
                "continuous": continuous,
                'categorical': categorical,
                "eid": self.eid[index]}

class DataCollator:
    def __init__(self, data_prep_module):
        self.data_prep_module = data_prep_module

    def __call__(self, batch):
        # 1. Rebuild a DataFrame for tabular features
        cats = [b["categorical"] for b in batch]
        conts = [b["continuous"] for b in batch]
        df_batch = pd.DataFrame([{**c, **ct} for c, ct in zip(cats, conts)])
        df_batch = df_batch.reindex(columns=MODEL_ATTRIBUTES, fill_value=np.nan)

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
        images = torch.stack([b["scan"] for b in batch])

        return {
            "x": xs,                      # node features
            "edge_attr": edge_attrs,      # edge features
            "mask": masks,                # attention mask
            "image": images,              # imaging input
            "eid": [b["eid"] for b in batch],
        }