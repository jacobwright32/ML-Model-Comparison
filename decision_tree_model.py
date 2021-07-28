from sklearn import tree
from Base import Base


class Decision_tree(Base):
    def __init__(self, params):
        super().__init__(tree, params)
    
        self.params = {
                'criterion' : ['mse', 'friedman_mse', 'mae', 'poisson'],
                'splitter': ['best', 'random'],
        #        'max_depth': [1, 2, None],
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
