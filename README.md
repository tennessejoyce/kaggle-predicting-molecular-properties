# kaggle-predicting-molecular-properties
Submission for the Kaggle competition 'Predicting molecular properties'. The challenge was to predict magnetic coupling parameters for arbitrary molecules given only the structure. For each molecule in the training set, we are given coordinates of the various atomic nuclei, as well as the target, which is the coupling between various pairs of atoms in the molecule. The main difficulty in this challenge was feature engineering, extracting relevant features from the raw coordinates of the atoms.

# My approach
I used the intuition that the magnetic coupling between a pair of atoms should depend more on the atoms nearby than the ones far away. The first step of my model is a K-nearest neighbor search to find which atoms were nearby. I computed the pairwise distances between each of those atoms, and used those as features. We are also given some auxilliary data (Mulliken charges and magnetic shielding tensors) for molecules in the training set, but not in the test set. I trained a separate model to predict these values, and then used those predictions as features in my main model.

# Interesting code snippets
## Vectorized KNN search
None of the built in functions did what I want.
```
#Vectorized kNN search. For each vector in targets, finds the k nearest vectors in struc.
def kNN(targets, struc, k):
    bsq = np.sum(struc**2,axis=1)
    objective = -2*np.matmul(targets,np.transpose(struc)) + bsq
    return np.argpartition(objective,kth=k-1,axis=1)[:,:k]
```

## XGBoost model

```
        #Fit a model to the training data
        
        params = {
            'max_depth' : 12,
            'n_estimators': 1500,
            'learning_rate': 0.3,
            'gamma': 0.5,
            'colsample_bytree' : 0.5,
            'tree_method': 'gpu_hist'
        }
        
        print('Training FC model...')
        modelFC = XGBRegressor(**params)    #(objective=huber_approx_obj)
        modelFC.fit(trainX,trainYFC_scaled)
        print('Training Diff model...')
        modelDiff = XGBRegressor(**params)
        modelDiff.fit(trainX,trainYDiff_scaled)
```

# Results
Leaderboard score (-1.00221 private, -1.00359 public). Rank 1361/2749.
