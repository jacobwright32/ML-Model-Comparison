#%%
from KNN_Model import hyper_para
from SVM_model import SVR_model

knn_scores = hyper_para()
print(knn_scores[len(knn_scores)-1][1])



svr = SVR_model()
results = svr.fit()
print(results)