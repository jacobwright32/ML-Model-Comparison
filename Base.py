#%%
import pandas as pd
import timeit
import itertools
from get_data import split_data


class Base:

    def __init__(self, model):
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = split_data()
        self.model = model
        self.params = {}
        

    def generate_params(self, params):
        
        keys = list(params.keys())
        values = params.values()
        grid_params = list(itertools.product(*values))

        return keys, values, grid_params


    def fit(self):
        keys, values, grid_params = self.generate_params(self.params)
        results = []
        
        for r in grid_params:
                param = {keys[idx]:r[idx] for idx in range(len(r))}
                reg = self.model(**param)
                start = timeit.timeit()
                reg.fit(self.X_train, self.y_train)
                end = timeit.timeit()
                # Train the model with grid search
                result = {
                    'time_taken' : (end-start),
                    'params': param,
                    'train_score': reg.score(self.X_train, self.y_train),
                    'val_score': reg.score(self.X_val, self.y_val),
                    'test_score': reg.score(self.X_test, self.y_test),
                }

                results.append(result)

        return results



    def best_model(self):
        
        results = self.fit()
        results_df = pd.DataFrame(results)
        best_result = results_df[results_df['val_score'] == results_df['val_score'].max()].iloc[0]
        best_params = best_result.params
        best_time = best_result.time_taken
        
        return self.model(**best_params), best_time

        

# %%
