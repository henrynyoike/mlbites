import numpy as np 
from numpy.typing import ArrayLike
import sys

def make_regression(n_samples:int=100 , 
            n_features:int=100 , 
            n_targets=1 , 
            shuffle:bool = True):

    if n_features > n_samples : 
        raise ValueError("Number of Features is greater that the total number of observations in the dataset")
        sys.exit(0)   
   
    coeff = np.random.randn(n_features , 1)
    bias = np.random.randn()

    # Get the random values of the datasets
    X = np.random.randn(n_samples , n_features)
    y = np.dot(X , coeff ) + bias

    data = (X , y)

    return data

def make_classification(
    n_samples:int = 100, 
    n_features:int = 20,
    n_classes:int = 2, 
    n_repeated:int = 0 , 
    shuffle:bool = True
):
    if n_repeated > n_samples : 
        raise ValueError("Number of Repeated observations is greater that the total number of observations in the dataset")
        sys.exit(0)   
   
    out_shape = (n_samples-n_repeated , n_features) # Samples to be less by n_repeated to make room for repeated observations
    
    # Get the random values of the datasets
    X = np.random.uniform(size=out_shape) # Uniform distribution of samples
    y = np.random.randint(low = 0 , high = n_classes , size=(n_samples , 1) , dtype=int)

    if n_repeated > 0 :
        repeat_set = np.random.uniform(size=(1 , n_features)) # Get one sample to repeat
        repeated = np.repeat(repeat_set , n_repeated , axis=0)
        X = np.append(X , repeated , axis=0) # Append the repeated sample to the main dataset

    if shuffle: 
        np.random.shuffle(X)
    
    data = (X , y)
    return data




