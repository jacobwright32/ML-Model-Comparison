#%%
from get_data import split_data
from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor

X_train, X_val, X_test, y_train, y_val, y_test = split_data()
k = 5

def KNN_Model(X, X_val, y, y_val):
    
    knn = neighbors.KNeighborsRegressor(n_neighbors=k)
    knn.fit(X, y)
    return print(knn.score(X_val, y_val))

KNN_Model(X_train, X_val, y_train, y_val)

# %%
# %%
