#%%
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


def split_data():
    random_seed_gen = 32
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_seed_gen)   
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=random_seed_gen)
    
    return X_train, X_val, X_test, y_train, y_val, y_test