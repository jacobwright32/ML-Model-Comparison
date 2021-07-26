#%%
from get_data import split_data
from sklearn.neighbors import NearestNeighbors

X_train, X_val, X_test, y_train, y_val, y_test = split_data()

def KNN_Model(X, y):
    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(X,y)
    

    
KNN_Model(X_train, y_train)

# %%
