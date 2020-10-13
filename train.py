from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LassoCV,RidgeCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

data = datasets.load_boston()
x_train,y_train,columns,data_desc = data['data'],data['target'],data['feature_names'],data['DESCR']
x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,test_size=0.8)

model = LassoCV(cv=5,max_iter=1000,random_state=2020)
model.fit(x_train,y_train)
print('Lasso:',model.score(x_test,y_test))

model = RidgeCV(cv=5)
model.fit(x_train,y_train)
print('Ridge:',model.score(x_test,y_test))

model = SVR(max_iter=1000)
model.fit(x_train,y_train)
print('SVR:',model.score(x_test,y_test))

model = RandomForestRegressor(n_estimators=500,max_depth=4,random_state=2020)
model.fit(x_train,y_train)
print('RF:',model.score(x_test,y_test))





