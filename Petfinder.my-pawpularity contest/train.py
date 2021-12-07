from config import CFG
from preprocess import *
from data import *

import sys
from timm import create_model
from fastai.vision.all import *
import gc
import glob
import torch
import os



for i in range(CFG.n_folds):
    
    print(f'Fold {i} results')
    
    learn = get_learner(fold_num = i)
    
    learn.fit_one_cycle(CFG.epoch, CFG.lr, cbs = [SaveModelCallback(), EarlyStoppingCallback(monitor='petfinder_rmse', comp=np.less, patience = CFG.patience)])
        
    learn = learn.to_fp32()
    
    #set save path and export weights
    os.makedirs(f'./models/{CFG.folder}',exist_ok=True)
    
    learn.path = Path(f'./models/{CFG.folder}')
    
    learn.export(f'{CFG.model_name}_{i}_fold')
    
    del learn
    
    torch.cuda.empty_cache()
    
    gc.collect()
