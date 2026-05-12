import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import random
import torchvision.transforms as transforms

def set_seed(seed=2022):
    random.seed(seed)                          # Python built-in RNG
    np.random.seed(seed)                       # NumPy RNG
    torch.manual_seed(seed)                    # PyTorch CPU RNG
    torch.cuda.manual_seed(seed)               # PyTorch current GPU RNG
    torch.cuda.manual_seed_all(seed)  

# t-SNE Generation and Plotting
def generate_tsne(projections, labels, title, save_path):
    projections = np.concatenate(projections, axis=0)  # Combine all projections
    tsne = TSNE(n_components=2, random_state=42, perplexity=50)
    tsne_embeddings = tsne.fit_transform(projections)

    # Create t-SNE plot
    plt.figure(figsize=(6, 4))
    unique_labels = list(set(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        indices = [j for j, lbl in enumerate(labels) if lbl == label]
        plt.scatter(
            tsne_embeddings[indices, 0],
            tsne_embeddings[indices, 1],
            color=colors[i],
            label=label,
            alpha=0.7
        )

    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(loc="best", bbox_to_anchor=(1.05, 1), fontsize='small', borderaxespad=0.)
    plt.tight_layout()

    # Save plot locally and log to wandb
    plt.savefig(save_path, dpi=300)
    plt.close()

def calculate_labels(cont_variables, bin_variables, multiple=False, multiple_thr=0.05):
    # calculate continous similarity
    batch_size = cont_variables.shape[0]
    diag_mask = torch.eye(batch_size, dtype=torch.bool)
    distance_matrix = torch.cdist(cont_variables, cont_variables, p=2)
    continuous_similarity =  1 / (1 + distance_matrix)
    continuous_similarity = continuous_similarity*2-1 
    # calculate binary similarity
    binary_similarity = torch.matmul(bin_variables, bin_variables.T) / len(bin_variables)
    similarity = (continuous_similarity + binary_similarity) / 2
    similarity = similarity[~diag_mask].view(batch_size, -1)
    # mask = torch.zeros_like(similarity)
    labels = []
    for i, sim in enumerate(similarity):
        # if multiple:
        #     max_sim = sim.max()  
        #     positive_indices = (sim >= max_sim - multiple_thr).nonzero(as_tuple=True)[0]
        # else:
        positive_index = torch.argmax(sim)
        labels.append(positive_index)
    return torch.tensor(labels, dtype=torch.long)

def calculate_labels_multimodal(cont_variables, bin_variables):
    # calculate continous similarity
    con0, con1 = cont_variables
    bin0, bin1 = bin_variables
    batch_size = con0.shape[0]
    diag_mask = torch.eye(batch_size, dtype=torch.bool)
    distance_matrix = torch.cdist(con0, con1, p=2)
    continuous_similarity =  1 / (1 + distance_matrix)
    continuous_similarity = continuous_similarity*2-1 
    # calculate binary similarity
    binary_similarity = torch.matmul(bin0, bin1.T) / len(bin_variables)
    similarity = (continuous_similarity + binary_similarity) / 2
    similarity = similarity[~diag_mask].view(batch_size, -1)
    labels = []
    for sim in similarity:
        positive_index = torch.argmax(sim)
        labels.append(positive_index)
    return torch.tensor(labels, dtype=torch.long)

def clip_sampling(scan, type, num_slices=11):
    '''Function to perform clip sampling of the scans, with the output size being HxWx11x10
    Inputs:
        - scan: 4D volume of the cardiac MR of size H x W x num_slices x num_of_frames 
        - type: original (our intended sampling, from 50 frames to 10, stride = 5) or augmented (random sampling) 
    Outputs:
        - clippes_scan: 4D valume clipped to the size H x W x 11 x 10
    '''
    h, w, s, t = scan.shape
    assert type == 'original' or type=='random', "input correct clipping type: 'random' or 'original'"
    clipped_scan = np.zeros((h, w, num_slices, 10))
    num_start = 0
    if s > num_slices:
        if s == 10 or s == 11:
            slices_stride = 2
        else:
            slices_stride = 1
    else:
        slices_stride = 1
    if s < num_slices:
        num_slices = s
    selected_slices = [i for i in range(0, s, slices_stride) if i < num_slices]
    if type == 'original':
        j = 0
        if t == 50:
            for i in range(0, 49, 5):
                if j < 10:
                    clipped_scan[:,:,:len(selected_slices),j] = scan[:,:,selected_slices,i]
                    j+=1
        else:
            stride = t // 10
            j = 0
            for i in range(0, t, stride):
                if j < 10:
                    clipped_scan[:,:,:len(selected_slices),j] = scan[:,:,selected_slices,i]
                    j+=1
    elif type == 'random':
        start = random.randint(0, 40)
        max_stride = (50 - start) // 10
        if max_stride < 5:
            stride = random.randint(1, (50 - start) // 10)
        else:
            stride = random.randint(1, 4)
        j = 0
        for i in range(start, 49, stride):
            if j < 10:
                clipped_scan[:,:,:len(selected_slices),j] = scan[:,:,selected_slices,i]
                j+=1
    return clipped_scan

def check_nan(attribute):
    if torch.isnan(attribute):
        return 0
    else:
        return attribute

# def crop_pad_scan(scan, nh=200, nw=200):
#     s, f, h, w = scan.shape
#     tscan = np.zeros((s,f,nh,nw))
#     if h > nh and w > nw:
#         h_diff = (h - nh) // 2
#         w_diff = (w - nw) // 2
#         tscan[:,:,:,:] = scan[:,:,h_diff:h_diff+nh,w_diff:w_diff+nw]
#     elif h < nh and w > nw:
#         h_diff = (nh - h) // 2
#         w_diff = (w - nw) // 2
#         tscan[:,:,h_diff:h_diff+h,:] = scan[:,:,:,w_diff:w_diff+nw]
#     elif h > nh and w < nw:
#         h_diff = (h - nh) // 2
#         w_diff = (nw - w) // 2
#         tscan[:,:,:,w_diff:w_diff+w] = scan[:,:,h_diff:h_diff+nh,:]
#     else:
#         h_diff = (nh - h) // 2
#         w_diff = (nw - w) // 2
#         tscan[:,:,h_diff:h_diff+h,w_diff:w_diff+w] = scan[:,:,:,:]
#     return tscan

def crop_pad_scan(scan, seg, nh=128, nw=128):
    """
    Crop a 4D scan (slices, frames, height, width) around the center of a 3D segmentation mask.

    Args:
        scan (np.ndarray): 4D array (slices, frames, H, W)
        seg (np.ndarray): 3D array (H, W, slices)
        nh (int): Crop height
        nw (int): Crop width

    Returns:
        tscan (np.ndarray): Cropped 4D scan (slices, frames, nh, nw)
    """
    s, f, h, w = scan.shape
    h_seg, w_seg, s_seg = seg.shape

    center_slice_idx = s_seg // 2
    seg_slice = seg[:, :, center_slice_idx]
    coords = np.argwhere(seg_slice == 1)

    # If middle slice has no labels, search others
    if coords.size == 0:
        found = False
        for i in range(s_seg):
            if i == center_slice_idx:
                continue  # Already checked
            seg_slice = seg[:, :, i]
            coords = np.argwhere(seg_slice == 1)
            if coords.size > 0:
                found = True
                break
        if not found:
            # Fallback to image center
            y_center = h // 2
            x_center = w // 2
        else:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            y_center = (y_min + y_max) // 2
            x_center = (x_min + x_max) // 2
    else:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        y_center = (y_min + y_max) // 2
        x_center = (x_min + x_max) // 2

    # Pad scan if needed
    pad_top = max(nh // 2 - y_center, 0)
    pad_bottom = max((y_center + nh // 2) - h, 0)
    pad_left = max(nw // 2 - x_center, 0)
    pad_right = max((x_center + nw // 2) - w, 0)

    if any([pad_top, pad_bottom, pad_left, pad_right]):
        scan = np.pad(
            scan,
            pad_width=((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=0
        )
        y_center += pad_top
        x_center += pad_left
        h += pad_top + pad_bottom
        w += pad_left + pad_right

    # Compute crop coordinates
    y_start = y_center - nh // 2
    x_start = x_center - nw // 2
    y_start = max(min(y_start, h - nh), 0)
    x_start = max(min(x_start, w - nw), 0)
    y_end = y_start + nh
    x_end = x_start + nw

    # Crop the scan
    tscan = scan[:, :, y_start:y_end, x_start:x_end]
    return tscan

def normalize_scan(scan):
    if len(np.unique(scan)) == 1:
        return scan
    min_val = scan.min()
    max_val = scan.max()
    scan = (scan - min_val) / (max_val - min_val)
    return scan

def normalize_attributes(attributes):
    ind_binary = [i for i in range(13, 23)]
    ind_continuous = [i for i in range(13)]
    ind_continuous.append(23)
    ind_binary.append(24)
    continuous_attributes = attributes[:, ind_continuous]
    continuous_attributes = torch.nn.functional.normalize(continuous_attributes, p=2, dim=1)
    attributes[:, ind_continuous] = continuous_attributes
    for i in ind_continuous:
        if torch.isnan(attributes[0, i]):
            attributes[0, i] = 0  # Replace NaN with 0
    for i in ind_binary:
        if i == 13: #smoking
            if torch.isnan(attributes[0, i]):
                attributes[0, i] = 0  # Replace NaN with 0
            elif attributes[0, i] == 1:
                attributes[0, i] = 0.5
            elif attributes[0, i] == 0:
                attributes[0, i] = -1
            elif attributes[0, i] == 2:
                attributes[0, i] = 1
            else:
                attributes[0,i] = 0
        else:
            if torch.isnan(attributes[0, i]):
                attributes[0, i] = 0  # Replace NaN with 0
            elif attributes[0, i] == 0:
                attributes[0, i] = -1  # Replace 0 with -1
    return attributes

def prepare_labels(labels):
    for i, lbl in enumerate(labels):
        if np.isnan(lbl):
            labels[i] = 0
    return labels

def grab_image_augmentations(img_size = 128) :
    """
    Defines augmentations to be used with images during contrastive training and creates Compose.
    """
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(size=img_size, scale=(0.6,1)),
        transforms.Lambda(lambda x: x.float())
        ])
    return transform

def get_bias(attribute, external=False):
    if external:
        attribute_dict = {
            "LVEF" : 52.01
        }
    attribute_dict = {
        "LVEF" : 57.25,
        "LVEDM" : 85.5,
        "LVEDV" : 155.11,
        "LVESV" : 66.99,
        "LVSV" : 88.13,
        "LVCO" : 5.29,
        "RVEDV" : 146.36,
        "RVESV" : 59.82,
        "RVSV" : 86.54,
        "RVEF" : 59.59,
        "RVCO" : 5.19,
        "MYOEDV" : 81.43,
        "MYOESV" : 82.69,
        # 'Price' : -4.4112e-08,
        'Price': 0,
        'age': 64.99,
        'Stroke volume during PWA': 117.11,
        'bmi': 26.55
    }
    return attribute_dict[attribute]

def map_categorical_values(df, categorical_columns, label_mapping):
    df_mapped = df.copy()
    for col in categorical_columns:
        if col in label_mapping:  # Only map if mapping exists for this column
            df_mapped[col] = df_mapped[col].map(label_mapping[col]).astype(str)
    return df_mapped