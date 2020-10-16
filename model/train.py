from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LassoCV,RidgeCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

def train_model():

    try:
        x_train = pd.read_csv('data/train.csv')
        y_train = x_train['target']

        x_test = pd.read_csv('data/test.csv')
        y_test = x_test['target']
        print('... ')
    except:
        data = datasets.load_boston()
        x_train,y_train,columns,data_desc = data['data'],data['target'],data['feature_names'],data['DESCR']
        x_train = pd.DataFrame(x_train,columns=columns)
        x_train['target'] = y_train

        x_train,x_test = train_test_split(x_train,test_size=0.8)
        x_train.to_csv('data/x_train.csv')
        x_test.to_csv('data/x_test.csv')

        y_train = x_train['target']
        y_test = x_test['target']

        x_train.drop('target',axis=1,inplace=True)
        x_test.drop('target',axis=1,inplace=True)

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

    return model

train_model()






