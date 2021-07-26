#%%
from sklearn import tree
from get_data import split_data


# Getting a boston dataset splited by train, validation, and test set
X_train, X_val, X_test, y_train, y_val, y_test = split_data()


def decision_tree():

    clf = tree.DecisionTreeRegressor()

    # Train the model
    clf = clf.fit(X_train, y_train)
    
    # Validate the model
    result = {
        'train_score': clf.score(X_train, y_train),
        'val_score': clf.score(X_val, y_val),
        'test_score': clf.score(X_test, y_test),
    }

    return result

train_result = decision_tree()


print(train_result)


# %%
