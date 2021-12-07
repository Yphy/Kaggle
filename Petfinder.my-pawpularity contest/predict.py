from fastai.vision.all import *
from config import CFG
from preprocess import *
import glob
import gc

#saved model path
model_dir = f'./models/{CFG.infer_folder}/'
model_ls = glob.glob(model_dir + '*_fold')


#make dataloader
train_df = stratify_df()
test_df = get_test_df()

dls = ImageDataLoaders.from_df(train_df,
                               valid_pct=0.2,
                               seed=999,
                               fn_col='path',
                               label_col='norm_score',
                               y_block=RegressionBlock,
                               bs = CFG.batch_size,
                               num_workers = CFG.num_workers,
                               item_tfms = Resize(CFG.img_size),
                               batch_tfms=setup_aug_tfms([Brightness(), Contrast(), Hue(), Saturation()]))
test_dl = dls.test_dl(test_df)

# #load weights and predict
all_preds = []
for idx,model in enumerate(model_ls):
    print(f'{idx}/{len(model_ls)} th fold prediction is started')
    learner = load_learner(model,cpu=False)
    tester = learner.dls.test_dl(test_df['path'])
    preds,_ = learner.get_preds(dl=tester)
#     preds,_ = learner.tta(dl=tester,n=CFG.tta, beta=0) #tta option
    all_preds.append(preds)  
    del learner
    torch.cuda.empty_cache()
    gc.collect()
    
preds = (np.mean(np.stack(all_preds),axis=0))*100

test_df['Pawpularity'] = preds
print(test_df.head())
         
test_df.to_csv('submission.csv')
 