import torch
from config import CFG
from preprocess import *
import sys
from timm import create_model
from fastai.vision.all import *


def seed_setting():
    seed = CFG.seed
    torch.cuda.set_device(CFG.deevice)
    set_seed(seed, reproducible=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def petfinder_rmse(input,target):
    return 100*torch.sqrt(F.mse_loss(F.sigmoid(input.flatten()),target))


def get_data(fold):
    train_df = stratify_df()
    train_df_f = train_df.copy()
    train_df_f['is_valid'] = (train_df_f['fold'] == fold)

    dls = ImageDataLoaders.from_df(train_df_f,
                                   valid_col = 'is_valid',
                                   seed = CFG.seed,
                                   fn_col = 'path',
                                   label_col = 'norm_score',
                                   y_block = RegressionBlock,
                                   bs = CFG.batch_size,
                                   num_workers = CFG.num_workers,
                                   item_tfms = Resize(CFG.img_size),
                                   batch_tfms = setup_aug_tfms([Brightness(),Contrast(),Hue(),Saturation()]))
    return dls

def get_learner(fold_num):
    data = get_data(fold_num)
    
    model = create_model(CFG.model_name, pretrained=True, num_classes=data.c)
    
    learn = Learner(data, model, loss_func = BCEWithLogitsLossFlat(), metrics=petfinder_rmse).to_fp16()

    return learn

