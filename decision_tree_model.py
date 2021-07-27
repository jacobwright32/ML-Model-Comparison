#%%
from sklearn import tree
import pandas as pd
import itertools
from get_data import split_data


# Getting a boston dataset splited by train, validation, and test set
X_train, X_val, X_test, y_train, y_val, y_test = split_data()


def decision_tree():

    keys, values, grid_params = generate_params()
    results = []

    for r in grid_params:
            
            param = {keys[idx]:r[idx] for idx in range(len(r))}
            print(param)
            reg = tree.DecisionTreeRegressor(**param)
            reg.fit(X_train, y_train)
            
            # Train the model with grid search
            result = {
                'params': param,
                'train_score': reg.score(X_train, y_train),
                'val_score': reg.score(X_val, y_val),
                'test_score': reg.score(X_test, y_test),
            }

            results.append(result)

    return results


def generate_params():
    
    params = {
        'criterion' : ['mse', 'friedman_mse', 'mae', 'poisson'],
        'splitter': ['best', 'random'],
        'max_depth': [1, 2, None],
#        'min_samples_split': [2, 0.2, 0.4],
#        'min_samples_leaf': [1, 0.1, 0.2],
#        'min_weight_fraction_leaf': [0.0, 0.1, 0.3],
        'max_features': ['auto', 'sqrt', 'log2', None],
#        'random_state': [1, 2, 3, None],
        'max_leaf_nodes': [2, 3, 4, None],
#        'min_impurity_decrease': [0.0, 0.1, 0.4],
#        'min_impurity_split': [0, 0.2, 0.4, None],
#        'ccp_alpha': [0.0, 0.1]
    }

    keys = list(params.keys())
    values = params.values()
    grid_params = list(itertools.product(*values))

    return keys, values, grid_params




results = decision_tree()

df = pd.DataFrame(results)

print(df)
# %%
