# kaggle-predicting-molecular-properties
Submission for the Kaggle competition 'Predicting molecular properties'. The challenge was to predict magnetic coupling parameters for arbitrary molecules given only the structure. For each molecule in the training set, we are given coordinates of the various atomic nuclei, as well as the target, which is the coupling between various pairs of atoms in the molecule. The main difficulty in this challenge was feature engineering, extracting relevant features from the raw coordinates of the atoms.

# My approach
I used the intuition that the magnetic coupling between a pair of atoms should depend more on the atoms nearby than the ones far away. The first step of my model is a K-nearest neighbor search to find which atoms were nearby. I computed the pairwise distances between each of those atoms, and used those as features. We are also given some auxilliary data (Mulliken charges and magnetic shielding tensors) for molecules in the training set, but not in the test set. I trained a separate model to predict these values, and then used those predictions as features in my main model.

#Interesting code snippets
##Vectorized KNN search

##

Transforms the raw molecular structure data into a set of features, then trains an XGBoost model.
