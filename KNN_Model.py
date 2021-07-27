#%%
from get_data import split_data
from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor

X_train, X_val, X_test, y_train, y_val, y_test = split_data()
<<<<<<< HEAD
k = 2

def KNN_Model(X, X_val, y, y_val):
=======
k = 4
def KNN_Model(X, X_val, y, y_val, k):
>>>>>>> 5890a690aa4cb6d17e1de860a9c61d5a36f5e7a3
    
    knn = neighbors.KNeighborsRegressor(n_neighbors=k)
    knn.fit(X, y)
    return knn.score(X_val, y_val)

def hyper_para():
    scores = {}
    for k in range(1,30):
            scores['k:' + str(k)] = KNN_Model(X_train, X_val, y_train, y_val, k)
    return sorted(scores.items(), key=lambda item: item[1])

knn_scores = hyper_para()
print(knn_scores)
# %%


