import sys
from timm import create_model
from fastai.vision.all import *
import gc
import glob
import torch


class CFG:
    seed = 999
    batch_size = 32
    device = 0
    model_name = 'swin_large_patch4_window7_224' #'vit_large_patch32_384'
    img_size = 224
    n_folds = 10
    num_workers = 8  
    
    #training configs
    epoch = 1
    lr = 2e-5
    patience = 0
    
    #save
    folder = 'test1'
    
    #infer
    infer_folder = 'test1' #saved folder
    tta = 5