#%%
from numpy.lib.npyio import _savez_compressed_dispatcher
from KNN_Model import hyper_para
from SVM_model import SVR_model
from decision_tree_model import decision_tree

knn_scores = hyper_para()
SVR_scores = SVR_model()
Decision_Tree_scores = decision_tree()
print(SVR_scores)
print(knn_scores)
# %%
q1