import os
from torchvision.io.video import read_video
from torchvision.models.video import r3d_18 , R3D_18_Weights 
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, average_precision_score, balanced_accuracy_score
import random
import torch.nn.functional as F
from torchlars import LARS
from torch.optim.lr_scheduler import CosineAnnealingLR


random.seed(7)
torch.manual_seed(7)

device0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device1 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def process_video(file):
    vid, _, _ = read_video(path + file, output_format="TCHW", pts_unit='sec')
    if vid.size(0) == 0:  # Check if the first dimension (time) is zero
        return torch.ones(3, 77, 112, 112)
    return preprocess(vid[:77])

