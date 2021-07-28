#%%
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from Base import Base

class KNN_model(Base):

    def __init__(self):
        super().__init__(KNeighborsRegressor)
        self.params = {
        'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        'leaf_size': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    }
# %%
