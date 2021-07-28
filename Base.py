import itertools
from get_data import split_data


class Base:
    def __init__(self, model):
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = split_data()
        self.model = model


    def generate_params(self, params):
        
        keys = list(params.keys())
        values = params.values()
        grid_params = list(itertools.product(*values))

        return keys, values, grid_params


    def fit(self):
        keys, values, grid_params = self.generate_params()
        results = []
        
        for r in grid_params:
                
                param = {keys[idx]:r[idx] for idx in range(len(r))}
                print(param)
                reg = self.model(**param)
                reg.fit(self.X_train, self.y_train)
                
                # Train the model with grid search
                result = {
                    'params': param,
                    'train_score': reg.score(self.X_train, self.y_train),
                    'val_score': reg.score(self.X_val, self.y_val),
                    'test_score': reg.score(self.X_test, self.y_test),
                }

                results.append(result)

        return results


        