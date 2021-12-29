#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import math
import seaborn as sns
import scipy.stats as stats


#Reading in and concatenating red and white datasets 


df1=pd.read_csv('/Users/manujavangula/Desktop/winequality-red.csv')
df1.head()
df2=pd.read_csv('/Users/manujavangula/Desktop/winequality-white.csv')
df2.head()
df3 = df1.append(df2, ignore_index=True)



#------------------------Preprocessing------------------------#
#Handling missing, duplicate, and outlier values
df3.isnull().sum()
df3.drop_duplicates(inplace=True)
df3.shape
z_scores = stats.zscore(df3)
abs_z_scores = np.abs(z_scores)
filtered = (abs_z_scores < 3).all(axis=1)
df4 = pd.DataFrame(df3[filtered])
df4.shape


#Normalization

from sklearn.preprocessing import StandardScaler
df5=df4
df5_columns=list(df5.columns.values)
scaler=StandardScaler()
scaler.fit(df5)
df6=scaler.transform(df5)
df6=pd.DataFrame(df6, columns=df5_columns)
df6


#Split into train and test data
X=df6[['alcohol', 'citric acid','free sulfur dioxide', 'density', 'volatile acidity', 'chlorides']]
X.head()

y=df6['quality']
y.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=10)




from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import math
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


#----------------------Linear Regression-------------------#

lr_clf=LinearRegression(normalize=True)
lr_clf.fit(X_train,y_train)
y_pred=lr_clf.predict(X_test)
y_true=y_test
mse_lr=mean_squared_error(y_true, y_pred)
mae_lr=mean_absolute_error(y_true, y_pred)
r2_lr=r2_score(y_true, y_pred)





#----------------------Ridge Regression-------------------#

clf = Ridge(alpha=1.0)
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
y_true=y_test
mse_r=mean_squared_error(y_true, y_pred)
mae_r=mean_absolute_error(y_true, y_pred)
r2_r=r2_score(y_true, y_pred)


#----------------------Lasso Regression-------------------#


las = Lasso(alpha=0.1)
las.fit(X_train, y_train)
y_pred=las.predict(X_test)
y_true=y_test
mse_l=mean_squared_error(y_true, y_pred)
mae_l=mean_absolute_error(y_true, y_pred)
r2_l=r2_score(y_true, y_pred)


#------------------Random Forest Regression---------------#


regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X_train, y_train)
regr.score(X_test, y_test)
y_pred=regr.predict(X_test)
y_true=y_test
mse_rf=mean_squared_error(y_true, y_pred)
mae_rf=mean_absolute_error(y_true, y_pred)
r2_rf=r2_score(y_true, y_pred)


#-------------K-Nearest Neighbors Regressor---------------#


knn=KNeighborsRegressor(n_neighbors=20)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
y_pred=knn.predict(X_test)
y_true=y_test
mse_knn=mean_squared_error(y_true, y_pred)
mae_knn=mean_absolute_error(y_true, y_pred)
r2_knn=r2_score(y_true, y_pred)


#--------Metrics: MSE, MAE, R-squared-----------------#



data=[['Linear Regression', mse_lr,  mae_lr, r2_lr], ['Ridge', mse_r, mae_r, r2_lr], 
     ['Lasso', mse_l,  mae_l, r2_l], ['Random Forest Regressor', mse_rf,  mae_rf, r2_rf],
     ['K Nearest Neighbor', mse_knn, mae_knn, r2_knn]]
metrics_df=pd.DataFrame(data, columns=['Model', 'Mean Squared Error',
                                       'Mean Absolute Error', 'R2 Score'])
metrics_df


#-------------Hyperparameter Tuning---------------#
#Determine best model and best parameters

def find_best_model_using_gridsearchcv(X,y):
    algos={
        'linear regression': {
            'model': LinearRegression(),
            'params':{
                'normalize': [True, False]
                
            }
        }, 
        'KNN':{
            'model': KNeighborsRegressor(),
            'params': {
                'n_neighbors': [3,5,11,19],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
                
            }
        },
        'Ridge':{
            'model': Ridge(),
            'params':{
                'alpha': [1,0.1,0.01,0.001,0.0001,0] ,
                'fit_intercept': [True, False],
                'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
            }
        }, 
        'Lasso':{
            'model':Lasso(), 
            'params':{
                'alpha': (np.logspace(-8, 8, 100))
            }
        },
        'Random Forest':{
            'model':RandomForestRegressor(), 
            'params':{
                'n_estimators': [200, 500],
                'max_features': ['auto', 'sqrt', 'log2'],
                'max_depth' : [4,5,6,7,8]
            }
        }
                
    }
    scores=[]
    cv=ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs=GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name, 
            'best_score': gs.best_score_, 
            'best_params': gs.best_params_
        })
    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])



find_best_model_using_gridsearchcv(X,y)
(find_best_model_using_gridsearchcv(X,y)).to_csv('GridSearchResults.csv')


#-------------Tuned Random Forest Regressor------#


regr2 = RandomForestRegressor(max_depth=8, max_features='log2',n_estimators=500,random_state=0)
regr2.fit(X_train, y_train)
regr2.score(X_test, y_test)
y_pred=regr2.predict(X_test)
y_true=y_test
mse_rf2=mean_squared_error(y_true, y_pred)
mae_rf2=mean_absolute_error(y_true, y_pred)
r2_rf2=r2_score(y_true, y_pred)


#Pre-Tuning and Tuned Random Forest Metrics


data=[['Tuned Random Forest Regressor', mse_rf2, mae_rf2, r2_rf2]]
tunedrf_df=pd.DataFrame(data, columns=['Model', 'Mean Squared Error',
                                       'Mean Absolute Error', 'R2 Score'])
oldrf_df=metrics_df.loc[metrics_df['Model']=='Random Forest Regressor']
df7 = oldrf_df.append(tunedrf_df, ignore_index=True)
df7



y_pred=regr.predict(X_test)
comp_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

