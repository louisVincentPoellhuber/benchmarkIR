import numpy as np
import random
import torch
from preprocessing.preprocess_utils import get_pairs_dataloader
import os
from tqdm import tqdm

def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

seed_everything(42)

dataloader = get_pairs_dataloader(
        batch_size=3, 
        dataset_path="/Tmp/lvpoellhuber/datasets/msmarco-doc/train_pairs.pt", 
        pin_memory=True, 
        prefetch_factor=2, 
        num_workers = 4
    )


loop = tqdm(dataloader, leave=True)
for i, batch in enumerate(loop):    
    if i==10526:
        pass
