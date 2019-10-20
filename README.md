# kaggle-predicting-molecular-properties
Submission for the Kaggle competition 'Predicting molecular properties'. The challenge was to predict magnetic coupling parameters for arbitrary molecules given only the structure. For each molecule in the training set, we are given coordinates of the various atomic nuclei, as well as the target, which is the coupling between various pairs of atoms in the molecule. The main difficulty in this challenge was feature engineering, extracting relevant features from the raw coordinates of the atoms.

# My approach
I used the intuition that the magnetic coupling between a pair of atoms should depend more on the atoms nearby than the ones far away. The first step of my model is a K-nearest neighbor search to find which atoms were nearby. I computed the pairwise distances between each of those atoms, and used those as features. We are also given some auxilliary data (Mulliken charges and magnetic shielding tensors) for molecules in the training set, but not in the test set. I trained a separate model to predict these values, and then used those predictions as features in my main model.

# KNN search
For each molecule with N atoms, we are given a Nx3 matrix (which I called struc) with their coordinates.
We are also given the magnetic couplings for M pairs of atoms in that molecule.
The indices of the atoms in the pairs are contained in two vectors atom_ids_0 and atom_ids1 of length M.
I pass that information (one molecule at a time) into the following function to compute the
indices of the nearest k atoms to the midpoint of each pair.

```
#Finds the k nearest atoms to the midpoint of a two given atoms.
#Vectorized so that it handles all pairs in a given molecule simultaneously.
def findNearest(atom_ids_0,atom_ids_1,struc,k):
    #Corner case if there are not enough atoms in the molecule.
    if struc.shape[0]<k+2:
        slack = k + 2 - struc.shape[0]
        k = struc.shape[0] -2
    else:
        slack = 0
    #Midpoints between each pair of atoms.
    targets = (struc[atom_ids_0]+ struc[atom_ids_1])/2
    #Find k+2 nearest neighbors to each midpoint
    #(which will probably include the original atom pair itself)
    nearest = kNN(targets,struc,k=k+2)
    out = []
    for i in range(nearest.shape[0]):
        temp = nearest[i]
        #First k elements which are not the original 2 atoms.
        temp = temp[(temp!=atom_ids_0[i])*(temp!=atom_ids_1[i])][:k]
        #If there were not enough atoms in the molecules, fill the rest with -1.
        if slack>0:
            temp = np.concatenate((temp,np.zeros(slack)-1))
        out.append(temp)
    return np.array(out)
```
The actual KNN search is handled in numpy by the following custom function,
```
#Vectorized kNN search. For each vector in a, finds the k nearest vectors in b.
def kNN(a, b, k):
    #Magnitudes of vectors in b
    b_squared = np.sum(b**2,axis=1)
    #Squared Euclidean distance without the the a_squared term
    objective = -2*np.matmul(a,np.transpose(b)) + b_squared
    #Use a partial sort to find the indices in b of the k smallest elements for each a.
    return np.argpartition(objective,kth=k-1,axis=1)[:,:k]
```
These indices are added to the training dataframe as atom_id_2 through atom_id_(k+1).
I use pandas merge function on these columns to compute the pairwise (inverse) distance matrix
among those k+2 atoms.



These distances are the most important features in the final model.


# XGBoost model


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
