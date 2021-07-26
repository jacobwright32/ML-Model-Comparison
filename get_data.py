#%%
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

random_seed = np.random.randint(1024)

def split_data():
    
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)   
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)
    return X_train, X_val, X_test, y_train, y_val, y_test