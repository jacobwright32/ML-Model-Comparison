from sklearn.svm import SVR
from get_data import split_data


def train_SVR():

    # Getting a boston dataset splited by train, validation, and test set
    X_train, X_val, X_test, y_train, y_val, y_test = split_data()

    # Call the SVC
    params = generate_params()
    reg = SVR(**params)

    # Train the model
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
        #'kernel' : ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        'kernel': 'rbf',
        'gamma': 0.01,
        'C': 10,
    }

    return params


result = train_SVR()
print(result)

