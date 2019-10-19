# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from hyperopt import hp, tpe
from hyperopt.fmin import fmin
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import mean_absolute_error,make_scorer
from sklearn.model_selection import train_test_split


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

#Vectorized kNN search. For each vector in targets, finds the k nearest
#vectors in struc.
def kNN(targets, struc, k):
    bsq = np.sum(struc**2,axis=1)
    objective = -2*np.matmul(targets,np.transpose(struc)) + bsq
    return np.argpartition(objective,kth=k-1,axis=1)[:,:k]

#Finds the k nearest atoms to the center of mass of a given two atoms.
#Vectorized so that it handles all pairs in a given molecule simultaneously.
def findNearest(atom_ids_0,atom_ids_1,struc,k):
    #Halfway points between pairs of atoms.
    if struc.shape[0]<k+2:
        slack = k + 2 - struc.shape[0]
        k = struc.shape[0] -2
    else:
        slack = 0
    targets = (struc[atom_ids_0]+ struc[atom_ids_1])/2
    #Find k+2 nearest neighbors to each halfway point
    #(which will probably include the original atom pair itself)
    near3 = kNN(targets,struc,k=k+2)
    #Remove the atom pair from the list, if it's there.
    #If it's not, just take the first k atoms.
    out = []
    for i in range(near3.shape[0]):
        temp = near3[i]
        temp = temp[(temp!=atom_ids_0[i])*(temp!=atom_ids_1[i])][:k]
        if slack>0:
            temp = np.concatenate((temp,np.zeros(slack)-1))
        out.append(temp)
    return np.array(out)
    
    
def calculateDist(i,j,data,structures):
    #Merge structure data into train.
    temp = data.merge(structures,left_on=['molecule_name',f'atom_index_{i}'],
            right_on=['molecule_name','atom_index'],how='left')
    temp = temp.merge(structures,left_on=['molecule_name',f'atom_index_{j}'],
                    right_on=['molecule_name','atom_index'],suffixes=('0','1'), how='left')
    #Calculate the reciprocal distances
    temp['distance'] = (temp['x0']-temp['x1'])**2 +  \
                    (temp['y0']-temp['y1'])**2 + (temp['z0']-temp['z1'])**2
    temp['distance'] = 1/np.sqrt(temp['distance'])
    #Fill missing values with 0
    temp = temp.fillna(0)
    #Sort the data by id
    temp = temp.sort_values('id')
    return np.reshape(temp['distance'].values,[-1])

def calculatemulliken(i,data,mulliken_charges):
    temp = data.merge(mulliken_charges,left_on=['molecule_name',f'atom_index_{i}'],
            right_on=['molecule_name','atom_index'],how='left')
    temp = temp.sort_values('id')
    temp = temp['mulliken_charge']
    temp = temp.fillna(0)
    return np.reshape(temp.values,[-1])

def calculateScalarShielding(i,data,shielding_tensors):
    temp = data.merge(shielding_tensors,left_on=['molecule_name',f'atom_index_{i}'],
            right_on=['molecule_name','atom_index'],how='left')
    temp['scalar_shielding'] = (temp['XX'] + temp['YY'] + temp['ZZ'])/3.0
    temp = temp.sort_values('id')
    temp = temp['scalar_shielding']
    temp = temp.fillna(0)
    return np.reshape(temp.values,[-1])


def processFeatures(infile,printOut=False,additionalData=True):
    #Read in training data
    print("Reading in data...")
    data = pd.read_csv('../input/champs-scalar-coupling/'+infile)
    
    loadFormulas = False
    if loadFormulas:
        #Load in the chemical formulas of the molecules.
        formulas = pd.read_csv('../input/atomcounts/atomCounts.csv',delimiter=';')
        
        #Add chemical formulas to the features dataframe
        features = features.merge(formulas,on='molecule_name')
    
    
    #Load in the structure data.
    structures = pd.read_csv('../input/champs-scalar-coupling/structures.csv')
    #structures.set_index(['molecule_name','atom_index'],inplace=True)
    
    #Group data and structrures by molecule name, to process in chunks.
    groupedData = data.groupby('molecule_name')
    groupedStruc = structures.groupby('molecule_name')
    
    print('Calculating nearest neighbors...')
    #How many neighbors?
    k=3
    #Loop over molecules and calculate nearest neighbors for each pair.
    nearest = []
    for name,group in groupedData:
        #Look up molecular structure
        struc = groupedStruc.get_group(name)[['x','y','z']].values
        #Get atom ids in to 1D numpy arrays.
        id0 = np.reshape(group[['atom_index_0']].values,[-1])
        id1 = np.reshape(group[['atom_index_1']].values,[-1])
        #Find nearest neighbors
        temp = findNearest(id0,id1,struc,k=k)
        #Restructure into a dataframe, naming columns appropriately.
        temp = pd.DataFrame(temp,columns=[f'atom_index_{ki}' for ki in range(2,k+2)])
        #Record the id of each row, in order to merge properly at the end
        temp['id'] = group['id'].values
        nearest.append(temp)
    #Concatenate all the dataframes from each molecule (chunk)
    nearestDF = pd.concat(nearest,ignore_index=True)
    #Merge with the raw dataframe, just adding columns for the new ids.
    data = data.merge(nearestDF, on='id')
    
    
    
    if additionalData:
        #Add in mulliken charges to the structure DataFrame
        print("Loading mulliken charges...")
        mulliken_charges = pd.read_csv('../input/estimatemullikenv1/mulliken_charges_est.csv',
                header=None,index_col=0,names=['mulliken_charge'])
        print(type(mulliken_charges))
        structures['mulliken_charge'] = mulliken_charges['mulliken_charge']
        for i in range(k+2):
                data[f'mulliken_charge_{i}'] = data.merge(structures,
                    left_on=['molecule_name',f'atom_index_{i}'],
                    right_on=['molecule_name','atom_index'],how='left')['mulliken_charge']
        
        #Add in scalar magnetic shieldings
        print("Loading magnetic shielding...")
        scalar_shielding = pd.read_csv('../input/estimatemullikenv1/shielding_scalar_est.csv',
                header=None,index_col=0,names=['scalar_shielding'])
        print(type(mulliken_charges))
        structures['scalar_shielding'] = scalar_shielding['scalar_shielding']
        for i in range(k+2):
                data[f'scalar_shielding_{i}'] = data.merge(structures,
                    left_on=['molecule_name',f'atom_index_{i}'],
                    right_on=['molecule_name','atom_index'],how='left')['scalar_shielding']
        
    print('Calculating distances...')
    #Calculate the reciprocal distance between each pair of atoms.
    for i in range(1,k+2):
        for j in range(i):
            data[f'dist{j}{i}'] = calculateDist(i,j,data,structures)


    #Drop unnecessary rows
    data = data.drop([f'atom_index_{ki}' for ki in range(k+2)]+['molecule_name'],axis=1)
    data = data.set_index('id')
    data = data.fillna(0)
    
    return data
    
recalculateFeatures = True
if recalculateFeatures:
    print('Training features file not found, creating a new one...')
    featuresTrain = processFeatures('train.csv',printOut=True)
    featuresTrain.to_csv('featuresTrain.csv')
else:
    print('Training features file found, loading...')
    featuresTrain = pd.read_csv('../input/molfeaturesv1/featuresTrain.csv',index_col=0)
    
print(list(featuresTrain))


#Read in contrubutions to target
contributions = pd.read_csv('../input/champs-scalar-coupling/scalar_coupling_contributions.csv')
contributions['target'] = featuresTrain.pop('scalar_coupling_constant')
contributions['difference'] = contributions['target'] - contributions['fc']

#Pseudo-Huber loss function to approximate mean absolute error.
delta = 1.0
def huber_approx_obj(y_true,y_pred):
    residual = y_true - y_pred
    radical = np.sqrt(1 + (residual / delta) ** 2)
    grad = -residual / radical
    hess = 1.0 / (radical**3)
    return grad, hess


#For each category in the 'type' column, fit a model to that data.
allTypes = featuresTrain['type'].unique()
print('Unique types:')
print(allTypes)


# fineTune=False
# if fineTune:
#     t = allTypes[0]
#     ids = featuresTrain['type']==t
#     trainX = featuresTrain[ids].iloc[:,1:]
#     trainY = trainX.pop('scalar_coupling_constant').values
#     trainX = trainX.values
    
#     #Define a objective function as a cross-validation score
#     #as a function of hyperparameters.
#     def objective(params):
#         params = {
#             'n_estimators': int(params['n_estimators']),
#             'max_depth': int(params['max_depth']),
#             'gamma': params['gamma'] 
#         }
#             # 'gamma': "{:.3f}".format(params['gamma']),
#             # 'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
        
#         clf = XGBRegressor(learning_rate = 0.3, colsample_bytree = 0.8, **params)
        
#         scorer = make_scorer(mean_absolute_error,greater_is_better=True)
#         score = cross_val_score(clf, trainX, trainY, scoring=scorer, cv=3).mean()
#         print("Mean Absolute Error {:.3f} params {}".format(score, params))
#         return score
    
#     #Define the search space for hyperparameters.
#     space = {
#         'max_depth': hp.quniform('max_depth', 2, 12, 1),
#         'n_estimators': hp.quniform('n_estimators', 100, 1000,1),
#         'gamma': hp.uniform('gamma', 0, 0.5)
#     }
    
#     best = fmin(fn=objective,
#                 space=space,
#                 algo=tpe.suggest,
#                 max_evals=10)
#     print(best)

fitModel=True
if fitModel:
    print('Fitting data...')
    allModelsFC = {}
    allModelsDiff = {}
    allScalersFC = {}
    allScalersDiff = {}
    score = 0
    
    for t in allTypes:
        
        
        ids = featuresTrain['type']==t
        allX = featuresTrain[ids].drop('type',axis=1)
        allYFC = contributions[ids]['fc'].values
        allYDiff = contributions[ids]['difference'].values
        allX = allX.values
        
        print(t)
        print(allX.shape)
        
        validate = True
        if validate:
            trainX, testX, trainYFC, testYFC, trainYDiff, testYDiff = train_test_split(allX,allYFC,allYDiff)
        else:
            trainX, trainYFC, trainYDiff = allX,allYFC,allYDiff
            testX, testYFC, testYDiff = allX,allYFC,allYDiff
        
        #Normalize the data
        scalerFC = StandardScaler()
        scalerDiff = StandardScaler()
        trainYFC_scaled = scalerFC.fit_transform(np.reshape(trainYFC,[-1,1]))
        trainYDiff_scaled = scalerDiff.fit_transform(np.reshape(trainYDiff,[-1,1]))
        
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
        
        #Save both the models and scalers for later
        allModelsFC[t] = modelFC
        allModelsDiff[t] = modelDiff
        allScalersFC[t] = scalerFC
        allScalersDiff[t] = scalerDiff
        
        
        #Calculate the training error
        predYFC = modelFC.predict(trainX)
        predYDiff = modelDiff.predict(trainX)
        predYFC = scalerFC.inverse_transform(np.reshape(predYFC,[-1,1]))[:,0]
        predYDiff = scalerDiff.inverse_transform(np.reshape(predYDiff,[-1,1]))[:,0]
        predY = predYFC + predYDiff
        errorFC = np.log(np.mean(np.abs(predYFC-trainYFC)))
        errorDiff = np.log(np.mean(np.abs(predYDiff-trainYDiff)))
        error = np.log(np.mean(np.abs(predY-trainYFC-trainYDiff)))
        print(f'Train error FC: {errorFC}')
        print(f'Train error Diff: {errorDiff}')
        print(f'Train error Total: {error}')
        
        #Calculate the validation error
        predYFC = modelFC.predict(testX)
        predYDiff = modelDiff.predict(testX)
        predYFC = scalerFC.inverse_transform(np.reshape(predYFC,[-1,1]))[:,0]
        predYDiff = scalerDiff.inverse_transform(np.reshape(predYDiff,[-1,1]))[:,0]
        predY = predYFC + predYDiff
        errorFC = np.log(np.mean(np.abs(predYFC-testYFC)))
        errorDiff = np.log(np.mean(np.abs(predYDiff-testYDiff)))
        error = np.log(np.mean(np.abs(predY-testYFC-testYDiff)))
        print(f'Validation error FC: {errorFC}')
        print(f'Validation error Diff: {errorDiff}')
        print(f'Validation error Total: {error}')
        
        score += error
#Print out the 'leaderboard' score on training data, no cross validation yet.
print(f'Total score: {score/len(allTypes)}')

doTest = True
if doTest:
    #Load in test data
    if recalculateFeatures:
        print('Test features file not found, creating a new one...')
        featuresTest = processFeatures('test.csv',printOut=True)
        featuresTest.to_csv('featuresTest.csv')
    else:
        print('Test features file found, loading...')
        featuresTest = pd.read_csv('../input/molfeaturesv1/featuresTest.csv',index_col=0)
    
    outID = []
    outPred = []
    for t in allTypes:
        print(t)
        #Select out data of the relevant type
        ids = np.array(featuresTest.index[featuresTest['type']==t])
        testX = featuresTest[featuresTest['type']==t].iloc[:,1:].values
        #Look up the models for this type
        modelFC = allModelsFC[t]
        scalerFC = allScalersFC[t]
        modelDiff = allModelsDiff[t]
        scalerDiff = allScalersDiff[t]
        #Make predictions
        predYFC = modelFC.predict(testX)
        predYDiff = modelDiff.predict(testX)
        predYFC = scalerFC.inverse_transform(np.reshape(predYFC,[-1,1]))[:,0]
        predYDiff = scalerDiff.inverse_transform(np.reshape(predYDiff,[-1,1]))[:,0]
        #Record the results
        outID.append(ids)
        outPred.append(predYFC+predYDiff)
    
    outID = np.concatenate(outID)
    outPred = np.concatenate(outPred)
    out = pd.DataFrame(np.transpose(np.array([outID,outPred])),columns=['id','scalar_coupling_constant'])
    #Sort and convert id to int
    out = out.sort_values('id').astype({'id':np.int}).reset_index(drop=True)
    out.to_csv('submission.csv',index=False)
