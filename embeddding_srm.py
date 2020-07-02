from os.path import join
import numpy as np
from pandas import read_csv
from itertools import combinations
from sklearn.model_selection import KFold
from brainiak.funcalign.srm import SRM
from brainiak.utils.utils import array_correlation
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import seaborn as sns

n_embs = 4585
emb_names = {'gpt2': 'GPT-2',
             'bart': 'BART',
             'roberta': 'RoBERTa',
             'xlnet': 'XLNet',
             'glove300': 'GloVe (300d)',
             'glove50': 'GloVe (50d)',
             'word2vec': 'word2vec',
             'decoder': 'brain (decoder 129d)',
             'decoder79': 'brain (decoder 79d)',
             'raw79': 'brain (raw 79d)'}

# Load and compile all embeddings
embeddings = {}
for name in emb_names:
    emb_fn = join('data', f'embeddings_{name}.csv')
    emb = read_csv(emb_fn, header=None).values
    assert emb.shape[0] == n_embs
    embeddings[name] = emb
    print(f"Loaded {emb_names[name]} embedding {emb.shape}")


# Get names for all embedding pairs
emb_pairs = list(combinations(emb_names, 2))

pair = emb_pairs[0]
    
    
# Simple z-scoring across cross-validation splits
def zscore_cv(train, test, axis=0):
    train_mean = np.mean(train, axis=axis)
    train_std = np.std(train, axis=axis)
    
    train_z = (train - train_mean) / train_std
    test_z = (test - train_mean) / train_std

    return train_z, test_z

    
# 10-fold cross-validation splitter
cv = KFold(n_splits=10)

for i, (train_ids, test_ids) in enumerate(cv.split(np.arange(n_embs))):
    print(i, len(train_ids), len(test_ids))



    
    
# Initialize SRM with fixed number of features
n_features = 50
srm = SRM(n_iter=10, features=n_features)


# Loop through all pairs of embedding types and get cross-validated SRM
srm_pairs = {}
for pair in emb_pairs:

    # Cross-validation split indices
    srm_tests = {pair[0]: [], pair[1]: []}
    for cv_i, (train_ids, test_ids) in enumerate(cv.split(np.arange(n_embs))):
        
        # Split and z-score embeddings
        emb0_train, emb0_test = zscore_cv(embeddings[pair[0]][train_ids],
                                          embeddings[pair[0]][test_ids])
        emb1_train, emb1_test = zscore_cv(embeddings[pair[1]][train_ids],
                                          embeddings[pair[1]][test_ids])
        
        # Convert NaNs to zero or SRM will freak out
        emb0_train = np.nan_to_num(emb0_train)
        emb1_train = np.nan_to_num(emb1_train)
        emb0_test = np.nan_to_num(emb0_test)
        emb1_test = np.nan_to_num(emb1_test)
        
        # Fit SRM on training embedding pair
        srm.fit([emb0_train.T, emb1_train.T])
        
        # Project test embedding pair into shared space
        srm_test0 = emb0_test.dot(srm.w_[0])
        srm_test1 = emb1_test.dot(srm.w_[1])
        print(f"SRM-aligned {emb_names[pair[0]]} and {emb_names[pair[1]]}"
              f" (CV fold {cv_i})")
        
        # Collect our SRM-aligned test sets
        srm_tests[pair[0]].append(srm_test0)
        srm_tests[pair[1]].append(srm_test1)
        
    # Stack SRM-aligned test embeddings
    srm_pairs[pair] =  {pair[0]: np.vstack(srm_tests[pair[0]]),
                        pair[1]: np.vstack(srm_tests[pair[1]])}

np.save(join('data', 'srm_test_embedding_pairs.npy'), srm_pairs)


# Compute token-wise correlations between embedding types
correlation_pairs = {}
for pair in srm_pairs:
    emb0, emb1 = srm_pairs[pair].values()
    correlation_pairs[pair] = array_correlation(emb0, emb1, axis=1)

# Get square correlation matrix and plot
corr_sq = squareform([np.mean(correlation_pairs[p])
                      for p in emb_pairs],
                     checks=False)
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(corr_sq, vmin=0, vmax=1, cmap='plasma', annot=True,
            fmt='.3f', cbar=True, square=True, ax=ax,
            cbar_kws={'label': 'correlation',
                      'fraction': 0.046, 'pad': 0.04},
            xticklabels=emb_names.values(),
            yticklabels=emb_names.values())
