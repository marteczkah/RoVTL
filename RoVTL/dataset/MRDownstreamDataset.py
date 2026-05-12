import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import random
import torch
from utils.utils import normalize_scan
import h5py
from dataset.attributes import REGRESSION_NUMERICAL_ATTRIBUTES, CLASSIFICATION_NUMERICAL_ATTRIBUTES, CLASSIFICATION_CATEGORICAL_ATTRIBUTES, CATEGORICAL_ATTRIBUTES, CAT_LABELS, NUMERICAL_MAPPING, LABEL_MAPPING, DISEASE_MAPPING

CAD_ATTRIBUTES_PRE = [
    "I20_premr", "I21v2_premr", "I24_premr", "I25_premr"
]

class MRDownstreamDatasetH5(Dataset):
    def __init__(self, 
                 h5_path, 
                 filter_csv, 
                 simulate_missing = True, 
                 attribute="",
                 augmentation_rate=-1, 
                 root='/lustre/groups/shared/ukbb-87065/dataset/cardiac_mri_nifti', 
                 sample_more=0.3,
                 num_present = 1, 
                 missing_tabular=False,
                 selected_num=[],
                 selected_cat=[]):
        self.root = root
        self.csv = pd.read_csv(filter_csv)
        target_eids = self.csv['eid'].to_numpy()
        self.info = h5py.File(h5_path, 'r')
        self.metadata_columns = self.info['metadata'].attrs['columns']
        metadata_ds = self.info['metadata']
        eid_ind = list(self.metadata_columns).index('eid')
        cad_ind = list(self.metadata_columns).index('no_cad')
        h5_eids = metadata_ds[:, eid_ind]
        h5_cads = metadata_ds[:, cad_ind]
        mask = np.isin(h5_eids, target_eids)
        self.filtered_indices = np.where(mask)[0]

        # Store the filtered EIDs and CADs
        self.eid = h5_eids[self.filtered_indices]
        self.cad = h5_cads[self.filtered_indices]
        self.metadata = metadata_ds[self.filtered_indices]
        if len(attribute) > 0:
            attribute_ind = list(self.metadata_columns).index(attribute)
            h5_attributes = metadata_ds[:, attribute_ind]
            self.attributes = h5_attributes[self.filtered_indices]
            self.numerical_attr = REGRESSION_NUMERICAL_ATTRIBUTES
        else:
            self.numerical_attr = CLASSIFICATION_NUMERICAL_ATTRIBUTES
        self.transform_augment = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(128, scale=(0.6, 1.0)),
                transforms.RandomRotation(45),
             ]
        )
        # self.CAD = info[CAD_ATTRIBUTES_POST].to_numpy()
        self.augmentation_rate = augmentation_rate
        self.attribute = attribute
        self.simulate_missing = simulate_missing

        # for z-score normalization
        self.num_means = {}
        self.num_stds = {}
        for col in self.numerical_attr:
            col_idx = list(self.metadata_columns).index(col)
            vals = self.metadata[:, col_idx].astype(float)
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                self.num_means[col] = vals.mean()
                self.num_stds[col] = vals.std() if vals.std() > 0 else 1.0
            else:
                self.num_means[col] = 0.0
                self.num_stds[col] = 1.0
        
        self.sample_more = sample_more
        self.num_present = num_present
        self.missing_tabular = missing_tabular
        self.selected_num = selected_num
        self.selected_cat = selected_cat

    def __len__(self):
        return len(self.eid)
    
    def __getitem__(self, index):
        h5_index = self.filtered_indices[index]
        # Read the image and normalize
        scan_org = self.info["images"][h5_index]
        scan_org = np.array(normalize_scan(scan_org))
        scan_aug = scan_org.copy()
        if random.random() <= self.augmentation_rate:
            scan_org = torch.from_numpy(scan_org)
            scan_org = self.transform_augment(scan_org)
            scan_aug = torch.from_numpy(scan_aug)
            scan_aug = self.transform_augment(scan_aug)
        else:
            scan_org = torch.from_numpy(scan_org)
            scan_aug = torch.from_numpy(scan_aug)
            scan_aug = self.transform_augment(scan_aug)

        metadata_row = {col: self.metadata[index, i] for i, col in enumerate(self.metadata_columns)}
        # select categorical values
        if self.simulate_missing:
            if random.random() < self.sample_more:
                # we sample 0.2 more often than others to account for the hard cases more
                chosen_cats1 = random.sample(CLASSIFICATION_CATEGORICAL_ATTRIBUTES, 
                            k=random.randint(1, int(0.2*len(CLASSIFICATION_CATEGORICAL_ATTRIBUTES))))
                chosen_nums1 = random.sample(self.numerical_attr, 
                            k=random.randint(1, int(0.2*len(self.numerical_attr))))
            else:
                chosen_cats1 = random.sample(CLASSIFICATION_CATEGORICAL_ATTRIBUTES, 
                                        k=random.randint(1, len(CLASSIFICATION_CATEGORICAL_ATTRIBUTES)))
                chosen_nums1 = random.sample(self.numerical_attr, 
                            k=random.randint(1, len(self.numerical_attr)))
            if self.missing_tabular:
                chosen_cats2 = []
                chosen_nums2 = []
            else:
                chosen_cats2 = random.sample(chosen_cats1, 
                                        k=random.randint(0, len(chosen_cats1)))
                chosen_nums2 = random.sample(chosen_nums1, 
                            k=random.randint(0, len(chosen_nums1)))
        else:
            if len(self.selected_cat) != 0 or len(self.selected_num) != 0:
                chosen_cats1 = self.selected_cat
                chosen_nums1 = self.selected_num
            elif self.num_present < 1:
                num_cats_to_keep = max(0, int(len(CLASSIFICATION_CATEGORICAL_ATTRIBUTES) * self.num_present))
                num_nums_to_keep = max(0, int(len(self.numerical_attr) * self.num_present))
                chosen_cats1 = random.sample(CLASSIFICATION_CATEGORICAL_ATTRIBUTES, k=num_cats_to_keep)
                chosen_nums1 = random.sample(self.numerical_attr, k=num_nums_to_keep)
            else:
                chosen_cats1 = CLASSIFICATION_CATEGORICAL_ATTRIBUTES
                chosen_nums1 = self.numerical_attr
            chosen_cats2 = []
            chosen_nums2 = []
            
        categorical1 = {}
        for col in chosen_cats1:
            val = metadata_row[col]
            if col in DISEASE_MAPPING.keys():
                new_col = DISEASE_MAPPING[col]
            else:
                new_col = col
            if not np.isnan(val):
                categorical1[new_col] = LABEL_MAPPING[col][val]
            else:
                categorical1[new_col] = np.nan
        categorical2 = {}
        for col in chosen_cats2:
            val = metadata_row[col]
            if col in DISEASE_MAPPING.keys():
                new_col = DISEASE_MAPPING[col]
            else:
                new_col = col
            if not np.isnan(val):
                categorical2[new_col] = LABEL_MAPPING[col][val]
            else:
                categorical2[new_col] = np.nan
        continuous1 = {}
        for col in chosen_nums1:
            val = metadata_row[col]
            if not np.isnan(val):
                continuous1[col] = float(val)
            else:
                continuous1[col] = np.nan
        continuous2 = {}
        for col in chosen_nums2:
            val = metadata_row[col]
            if not np.isnan(val):
                continuous2[col] = float(val)
            else:
                continuous2[col] = np.nan

        cad = self.cad[index]
        multilabel = [metadata_row[i] for i in CAD_ATTRIBUTES_PRE]
        multilabel = torch.tensor(np.nan_to_num(multilabel, nan=0), dtype=torch.float32)
        if cad == 1:
            has_cad = torch.tensor(0, dtype=torch.float32)
        else:
            has_cad = torch.tensor(1, dtype=torch.float32)
        
        if len(self.attribute) > 0:
            # attr = torch.tensor(self.attributes[index], dtype=torch.float32)
            attr = [metadata_row[i] for i in [self.attribute]]
            attr = torch.tensor(np.nan_to_num(attr, nan=0), dtype=torch.float32)
        else:
            attr = torch.tensor(0, dtype=torch.float32)
        
        tab_missing1 = int(len(chosen_cats1) == 0 and len(chosen_nums1) == 0)
        tab_missing2 = int(len(chosen_cats2) == 0 and len(chosen_nums2) == 0)
        return {"scan": scan_org, 
                'scan_aug' : scan_aug,
                "continuous1": continuous1,
                'categorical1': categorical1,
                "continuous2": continuous2,
                'categorical2': categorical2,
                "eid": self.eid[index],
                'has_cad':has_cad,
                'label':multilabel,
                'attribute':attr,
                "tab_missing1": torch.tensor(tab_missing1, dtype=torch.float32),
                "tab_missing2": torch.tensor(tab_missing2, dtype=torch.float32),
                }
        
    def encode_categorical(self, attributes):
        num_cats = np.array(CAT_LABELS)
        encoded = []
        i = 0
        for key, value in attributes.items():
            if key in CATEGORICAL_ATTRIBUTES:
                if pd.isna(value):
                    value = 0
                if num_cats[i] > 2:
                    one_hot = [-1] * num_cats[i]
                    ind = NUMERICAL_MAPPING[key][value]
                    one_hot[ind] = 1
                    encoded.extend(one_hot)
                else:
                    if int(value) == 1:
                        encoded.append(1)
                    else:
                        encoded.append(-1)
                i += 1
        return encoded
    
class DownstreamCollator:
    def __init__(self, data_prep_module):
        self.data_prep_module = data_prep_module

    def __call__(self, batch):
        combined_dicts = []
        for b in batch:
            combined_dicts.append({**b["categorical1"], **b["continuous1"]})
        for b in batch:
            combined_dicts.append({**b["categorical2"], **b["continuous2"]})

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

        has_cad = torch.stack([b["has_cad"] for b in batch])
        has_cad = torch.cat([has_cad, has_cad], dim=0)

        labels = torch.stack([b["label"] for b in batch])
        labels = torch.cat([labels, labels], dim=0)

        attributes = torch.stack([b["attribute"] for b in batch])
        attributes = torch.cat([attributes, attributes], dim=0)

        tab_missing1 = torch.stack([b["tab_missing1"] for b in batch])
        tab_missing2 = torch.stack([b["tab_missing2"] for b in batch])
        tab_missing = torch.cat([tab_missing1, tab_missing2], dim=0)
        eids = [b["eid"] for b in batch]
        eids = eids + eids  # duplicate list

        return {
            "x": xs,                  # node features
            "edge_attr": edge_attrs,  # edge features
            "mask": masks,            # attention mask
            "image": images,          # duplicated images
            "eid": eids,              # duplicated eids
            "has_cad": has_cad,
            "label": labels,
            "attribute": attributes,
            'tab_missing' : tab_missing
        }