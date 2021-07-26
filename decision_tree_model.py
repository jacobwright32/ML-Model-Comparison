#%%
from sklearn import tree
from get_data import split_data

X_train, X_val, X_test, y_train, y_val, y_test = split_data()


def decision_tree(X, y):

    clf = tree.DecisionTreeRegressor()

    clf = clf.fit(X, y)
    
    clf.score(X, y)

    #tree.plot_tree(clf)

    
decision_tree(X_train, y_train)


# %%
