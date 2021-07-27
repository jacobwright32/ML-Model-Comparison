import numpy as np
import pandas as pd
from sklearn.svm import SVR
from get_data import split_data


def train_SVR():

    # Getting a boston dataset splited by train, validation, and test set
    X_train, X_val, X_test, y_train, y_val, y_test = split_data()

    # Call the SVC
    #params = generate_params()


    # Train the model with grid search
    reg = SVR()
    reg.fit(X_train, y_train)

    # Validate the model
    result = {
        'train_score': reg.score(X_train, y_train),
        'val_score': reg.score(X_val, y_val),
        'test_score': reg.score(X_test, y_test),
    }

    return result
    

def generate_params():
    params = {
        'kernel' : ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'C': [0.1, 1, 10],
    }
    
    
    


    return params


def generate_params_svr():
    result = train_SVR()
    print(result)



print(generate_params_svr())
print(generate_params_svr())



