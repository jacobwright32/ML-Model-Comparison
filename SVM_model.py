import numpy as np
import pandas as pd
import itertools
from sklearn.svm import SVR
from get_data import split_data


def train_SVR():

    # Getting a boston dataset splited by train, validation, and test set
    X_train, X_val, X_test, y_train, y_val, y_test = split_data()

    # Call the SVC
    keys, values, grid_params = generate_params()
    results = []

    for r in grid_params:
        
        param = {keys[idx]:r[idx] for idx in range(len(r))}
        print(param)
        reg = SVR(**param)
        reg.fit(X_train, y_train)
        
        # Train the model with grid search
        result = {
            'params': param,
            'train_score': reg.score(X_train, y_train),
            'val_score': reg.score(X_val, y_val),
            'test_score': reg.score(X_test, y_test),
        }

        results.append(result)

    return results
    

def generate_params():
    
    params = {
        'kernel' : ['linear', 'rbf'],
        'gamma': [0.1, 1],
        'C': [0.1, 1],
    }

    keys = list(params.keys())
    values = params.values()
    grid_params = list(itertools.product(*values))

    return keys, values, grid_params
    

results = train_SVR()
df = pd.DataFrame(results)

# %%
