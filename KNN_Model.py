#%%
from numpy.lib.npyio import savez_compressed
from get_data import split_data
import itertools
from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd

X_train, X_val, X_test, y_train, y_val, y_test = split_data()


def KNN_Model():
        
    keys, values, grid_params = generate_params()
    results = []
    
    for r in grid_params:
        param = {keys[idx]:r[idx] for idx in range(len(r))}
        print(param)
        knn = neighbors.KNeighborsRegressor(**param)
        knn.fit(X_train, y_train)    
        
        result = {
                    'param' : param,
                'train_score': knn.score(X_train, y_train),
                'val_score': knn.score(X_val, y_val),
                'test_score': knn.score(X_test, y_test)
            }
    results.append(result)
    
    return results

def generate_params():
    scores = []
    
    params = {
        'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        'leaf_size': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    }
    
    keys = list(params.keys())
    values = params.values()
    grid_params = list(itertools.product(*values))    
    return keys, values, grid_params

results = KNN_Model()

df = pd.DataFrame(results)

print(df)
# %%

