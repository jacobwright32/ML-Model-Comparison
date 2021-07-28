#%%
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from Base import Base
x = LinearRegression()

class Linear_model(Base):

    def __init__(self):
        super().__init__(LinearRegression)
        self.params = {
        'normalize' : 'True',
        'fit_intercept' : [True],
        'n_jobs': -1,
        'positive': [True, False]
    }