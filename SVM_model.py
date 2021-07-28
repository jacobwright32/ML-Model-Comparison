#%%
from sklearn.svm import SVR
from Base import Base

class SVR_model(Base):

    def __init__(self, params):
        super().__init__(SVR, params)


params = {
            'kernel' : ['linear', 'rbf'],
            'gamma': [0.1, 1],
            'C': [0.1, 1],
        }

svr = SVR_model(params)
results = svr.fit()
print(results)