#%%
import numpy as np
from sklearn.datasets import load_boston, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def split_data(dataset_name='boston'):
    random_seed_gen = 32
    if dataset_name == 'boston':
        X, y = load_boston(return_X_y=True)
      
    else:
        X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_seed_gen)   
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=random_seed_gen)
        
    return X_train, X_val, X_test, y_train, y_val, y_test

