#%%
from get_data import split_data
from sklearn.neighbors import KNeighborsRegressor

X_train, X_val, X_test, y_train, y_val, y_test = split_data()

def KNN_Model(X, X_val, y, y_val):
    knn = KNeighborsRegressor(n_neighbors=5)
    y_ = knn.fit(X, y)


KNN_Model(X_train, X_val, y_train, y_val)

# %%