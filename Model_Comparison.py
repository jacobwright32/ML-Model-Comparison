#%%
from Linear_Regression_Model import Linear_model
from KNN_Model import KNN_model
from SVM_model import SVR_model
from decision_tree_model import Decision_tree
from get_data import split_data


#best_linear, best_linear_time = Linear_model().best_model()
best_knn, best_knn_time = KNN_model().best_model()
best_dt, best_dt_time = Decision_tree().best_model()
best_svr, best_svr_time = SVR_model().best_model()


X_train, X_val, X_test, y_train, y_val, y_test = split_data()
best_knn.fit(X_train, y_train)
best_dt.fit(X_train, y_train)
best_svr.fit(X_train, y_train)

knn_score = best_knn.score(X_val, y_val)
dt_score = best_dt.score(X_val, y_val)
svr_score = best_svr.score(X_val, y_val)

print(knn_score, dt_score, svr_score)
