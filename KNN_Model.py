#%%
from numpy.lib.npyio import savez_compressed
from get_data import split_data
from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor

X_train, X_val, X_test, y_train, y_val, y_test = split_data()
k = 4
leaf = 30
def KNN_Model(X_train, X_val, X_test, y_train, y_val, y_test, k, leaf):
    knn = neighbors.KNeighborsRegressor(n_neighbors=k,leaf_size=leaf)
    knn.fit(X_train, y_train)    
    score = {
                   'param' : [k,leaf],
            'train_score': knn.score(X_train, y_train),
            'val_score': knn.score(X_val, y_val),
            'test_score': knn.score(X_test, y_test)
        }
    
    return score

def hyper_para():
    scores = []
    for k in range(1,30):
        for leaf in range(1,30):
            scores.append(KNN_Model(X_train, X_val, X_test, y_train, y_val, y_test, k, leaf))
    return scores

knn_scores = hyper_para()
print(knn_scores)
# %%

# %%
