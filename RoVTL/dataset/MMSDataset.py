import nibabel as nib
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import random
import os
import torch
from utils.utils import normalize_scan, clip_sampling, crop_pad_scan
from dataset.attributes import MNMS_MAPPING

class MMSDataset(Dataset):
    def __init__(self, csv_path, num_columns, cat_columns, simulate_missing, img_only=False, mask_path="", importance=False, augmentation_rate = 0, regression=False, classification=False, attributes=None, class_type='multiclass'):
        self.info = pd.read_csv(csv_path)
        self.files = self.info['file_path'].to_numpy()
        self.eid = self.info['eid'].to_numpy()
        self.img_size = 128
        self.num_columns = num_columns
        self.cat_columns = cat_columns
        self.tabular_columns = self.num_columns + self.cat_columns
        self.phenotype = self.info[self.tabular_columns]
        self.augmentation_rate = augmentation_rate
        self.regression = regression
        self.classification = classification
        self.transform_augment = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
                transforms.RandomRotation(45),
             ]
        )
        self.class_type = class_type
        self.simulate_missing = simulate_missing
        if regression:
            self.attribute = self.info[attributes].to_numpy()
        if classification:
            if class_type == 'multiclass':
                self.label = self.info['label_num'].to_numpy()
            else:
                self.label = self.info['has_cad'].to_numpy()
        if len(mask_path) > 0:
            self.is_masked = True
            mask_data = np.load(mask_path, allow_pickle=True).item()
            eids = mask_data["eid"]
            mask = mask_data["mask"]
            self.columns = mask_data["columns"]
            if importance:
                self.eid_to_mask = {eid: mask for _, eid in enumerate(eids)}
            else:
                self.eid_to_mask = {eid: mask[i] for i, eid in enumerate(eids)}
        else:
            self.is_masked = False
        self.img_only = img_only

    def __len__(self):
        return len(self.eid)
    
    def __getitem__(self, index):
        cur_eid = self.eid[index]
        current_file = self.files[index]
        patient_root = os.path.dirname(current_file)
        seg_file = self.info.iloc[index]['seg_path']
        scan = nib.load(current_file).get_fdata()
        seg = nib.load(seg_file).get_fdata()
        seg = nib.load(os.path.join(patient_root, seg_file)).get_fdata()
        scan_org = clip_sampling(scan, 'original').transpose(2,3,0,1)
        scan_org = crop_pad_scan(scan_org, seg, nh=self.img_size, nw=self.img_size)  
        scan_org = normalize_scan(scan_org)
        scan_org = torch.from_numpy(np.array(scan_org))

        if random.random() <= self.augmentation_rate:
            scan_org = self.transform_augment(scan_org)

        if self.regression:
            attr = self.attribute[index]
            attr = torch.tensor(attr).float()
        else:
            attr = 0

        if self.classification:
            lbl = self.label[index]
            lbl = torch.tensor(lbl).long()
        else:
            lbl = 0
        
        if self.simulate_missing:
            chosen_cols = random.sample(self.tabular_columns, 
                k=random.randint(1, len(self.tabular_columns)))
            chosen_cols2 = random.sample(chosen_cols, 
                k=random.randint(0, len(chosen_cols)))
        elif self.img_only:
            chosen_cols = []
            chosen_cols2 = []
        elif self.is_masked:
            row_mask = self.eid_to_mask[cur_eid]
            cur_columns = [col for col, keep in zip(self.columns, row_mask) if keep]
            chosen_cols = cur_columns
            chosen_cols2 = []
        else:
            chosen_cols = self.tabular_columns
            chosen_cols2 = []

        tabular1 = self.phenotype.iloc[index][chosen_cols]
        attributes1 = {}
        for col in chosen_cols:
            if col in self.cat_columns:
                attributes1[col] = MNMS_MAPPING[col][tabular1[col]]
            else:
                attributes1[col] = tabular1[col]
        tabular2 = self.phenotype.iloc[index][chosen_cols2]
        attributes2 = {}
        for col in chosen_cols2:
            if col in self.cat_columns:
                attributes2[col] = MNMS_MAPPING[col][tabular2[col]]
            else:
                attributes2[col] = tabular2[col]

        return {"scan": scan_org, 
                "eid": cur_eid,
                "attributes1": attributes1,
                "attributes2": attributes2,
                "attribute" : attr,
                'label' : lbl}

class MMSCollator:
    def __init__(self, data_prep_module):
        self.data_prep_module = data_prep_module

    def __call__(self, batch):
        combined_dicts = []
        for b in batch:
            combined_dicts.append(b["attributes1"])
        for b in batch:
            combined_dicts.append(b["attributes2"])

        df_combined = pd.DataFrame(combined_dicts)

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
        images = torch.stack([b["scan"] for b in batch])
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