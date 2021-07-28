#%%
from sklearn.svm import SVR
from Base import Base

class SVR_model(Base):

    def __init__(self):
        super().__init__(SVR)
        self. params = {
            'kernel' : ['linear', 'rbf'],
            'gamma': [0.1, 1],
            'C': [0.1, 1],
        }




