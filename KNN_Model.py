#%%
from get_data import split_data
from sklearn.neighbors import KNeighborsClassifier
X_train, X_val, X_test, y_train, y_val, y_test = split_data()

def KNN_Model(X,X_val, y, y_val):
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X,y)
    y_pred = knn.predict(X_val)
    knn.score(y_val,y_pred)


KNN_Model(X_train, X_val, y_train, y_val)

# %%
