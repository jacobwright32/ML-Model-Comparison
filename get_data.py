#%%
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def split_data():
    random_seed_gen = 32
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_seed_gen)   
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=random_seed_gen)
<<<<<<< HEAD
    
    return X_train, X_val, X_test, y_train, y_val, y_test
=======

    return X_train, X_val, X_test, y_train, y_val, y_test
# %%
>>>>>>> 5890a690aa4cb6d17e1de860a9c61d5a36f5e7a3
