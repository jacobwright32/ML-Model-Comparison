from sklearn.svm import SVR
from get_data import split_data


def train_SVR(**params):

    # Getting a boston dataset splited by train, validation, and test set
    X_train, X_val, X_test, y_train, y_val, y_test = split_data()

    # Call the SVC
    clf = SVR(**params)

    # Train the model
    clf.fit(X_train, y_train)

    # Validate the model
    print(clf.score(X_val, y_val))


params = {
    'kernel' : 'rbf',
    'degree': 3,
    'gamma': 'auto',
    'C': 10
}

train_SVR(**params)

