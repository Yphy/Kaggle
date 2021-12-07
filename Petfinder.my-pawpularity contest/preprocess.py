from config import CFG
from fastai.vision.all import *
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold


def get_train_df():
    
    dataset_path = Path('./data')
    
    train_df = pd.read_csv(dataset_path/'train.csv')
    
    train_df['path'] = train_df['Id'].map(lambda x: str(dataset_path/'train'/x)+'.jpg')
    
    train_df = train_df.sample(frac=1).reset_index(drop=True) #shuffle
    
    train_df['norm_score'] = train_df['Pawpularity']/100
    
    num_bins = int(np.floor(1+(3.3)*(np.log2(len(train_df))))) #sturges rule
    
    train_df['bins'] = pd.cut(train_df['norm_score'],bins=num_bins, labels=False)
    
    return train_df


def stratify_df():
    
    train_df = get_train_df()
    
    train_df['fold'] = -1
 
    N_FOLDS = CFG.n_folds
    
    strat_kfold = StratifiedKFold(n_splits=N_FOLDS, random_state = CFG.seed, shuffle=True)

    for i, (_, train_index) in enumerate(strat_kfold.split(train_df.index, train_df['bins'])):
        train_df.iloc[train_index,-1] = i

    train_df['fold'] = train_df['fold'].astype('int')
   
    return train_df
    
    
def get_test_df():
    
    dataset_path = Path('./data')
    test_df = pd.read_csv(dataset_path/'test.csv')
    test_df['Pawpularity'] = [1]*len(test_df)
    test_df['path'] = test_df['Id'].map(lambda x:str(dataset_path/'test'/x)+'.jpg')
    
    return test_df