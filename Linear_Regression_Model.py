#%%
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from Base import Base


class Linear_model(Base):

    def __init__(self):
        super().__init__(Ridge)
        self.params = {
        'sample_weight' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        }
# %%
