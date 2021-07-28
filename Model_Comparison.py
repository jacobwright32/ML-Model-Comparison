#%%
from KNN_Model import KNN_model
from SVM_model import SVR_model
from decision_tree_model import Decision_tree
from get_data import split_data


best_knn = KNN_model().best_model()
best_decision_tree = Decision_tree().best_model()
best_svr = SVR_model().best_model()


# %%
