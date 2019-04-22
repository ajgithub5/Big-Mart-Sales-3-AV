# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 23:41:56 2019

@author: AJAY
"""


# coding: utf-8

import pandas as pd
import numpy as np

train=pd.read_csv("Y:\\Projects\\Big Mart Sales 3 -AV HACK\\train.csv")
test=pd.read_csv("Y:\\Projects\\Big Mart Sales 3 -AV HACK\\test.csv")

train.head()
test.head()

train['source']='train'
test['source']='test'

data=pd.concat([train,test],ignore_index=True)
print (train.shape,test.shape,data.shape)

data.head(n=10)
data.apply(lambda x:sum(x.isnull()))

data.describe()
data.apply(lambda x:len(x.unique()))

#filter categorical columns
categorical_columns=[x for x in data.dtypes.index if data.dtypes[x]=='object']

#Exclude ID cols and source
categorical_columns=[x for x in categorical_columns if x not in ['Item_Identifier','source','Outlet_Identifier']]

#Print frequency of categories
for col in categorical_columns:
    print('\n frequency of categories for variable %s'%col)
    print(data[col].value_counts())

#Item_Fat_Content: Some of ‘Low Fat’ values mis-coded as ‘low fat’ and ‘LF’. Also, some of ‘Regular’ are mentioned as ‘regular’.
#Item_Type: Not all categories have substantial numbers. It looks like combining them can give better results.
#Outlet_Type: Supermarket Type2 and Type3 can be combined. But we should check if that’s a good idea before doing it.

#Data CLeaning (imputing missing values)

def missing(x):
    return (sum(x.isnull()))

print ("Missing values per column:")
print(data.apply(missing,axis=0))

data.apply(lambda x:sum(x.isnull()))

#Determine the average weight per item:
item_avgwt=data.pivot_table(values='Item_Weight',index='Item_Identifier')
item_avgwt

#Get a boolean variable specifying missing Item_Weight values
miss_bool = data['Item_Weight'].isnull()

#Impute data and check #missing values before and after imputation to confirm
print('original #missing %d'%sum(miss_bool))
data.loc[miss_bool,'Item_Weight']=data.loc[miss_bool,"Item_Identifier"].apply(lambda x:item_avgwt[x])
print('final #missing:%d'%sum(data['Item_Weight'].isnull()))

#Import mode function:
from scipy.stats import mode

crosstable=pd.crosstab(index=data['Outlet_Size'],columns=data['Outlet_Type'])
crosstable.index=['high','medium','small']
crosstable

data.pivot_table(values="Item_Outlet_Sales",index='Outlet_Type')

#as there is significant difference between mean sales of different store types there is no point in combining them.

#Modify Item Visibility. Determine average visibility of a product
visibility_avg=data.pivot_table(values='Item_Visibility',index='Item_Identifier')

#Impute 0 values with mean visibility of that product:
miss_bool = (data['Item_Visibility'] == 0)

print ('Number of 0 values initially: %d'%sum(miss_bool))
data.loc[miss_bool,'Item_Visibility']=data.loc[miss_bool,'Item_Identifier'].apply(lambda x:visibility_avg[x])
print ('Number of 0 values after modification: %d'%sum(data['Item_Visibility'] == 0))

#Determine another variable with means ratio
data['Item_Visibility_MeanRatio'] = data.apply(lambda x: x['Item_Visibility']/visibility_avg[x['Item_Identifier']], axis=1)
print(data['Item_Visibility_MeanRatio'].describe())
data['Item_Type_Combined']=data['Item_Identifier'].apply(lambda x: x[0:2])

#Rename them to more intuitive categories:
data['Item_Type_Combined'].head()

data['Item_Type_Combined']=data['Item_Type_Combined'].map({'FD':'Food','NC':'Non-Consumable','DR':'Drinks'})
data['Item_Type_Combined'].value_counts()

data['Outlet_Years']=2013-data['Outlet_Establishment_Year']
data['Outlet_Years'].describe()

#Change categories of low fat:
print("Original Categories:")
print(data['Item_Fat_Content'].value_counts())
print("Modified Categories:")
data['Item_Fat_Content']=data['Item_Fat_Content'].replace({'LF':'Low Fat','Low Fat':'Low Fat','low fat':'Low Fat',
                                                          'Regular':'Regular','reg':'Regular'})
print(data['Item_Fat_Content'].value_counts())

#Modify Item Fat content to Non Edible where Item Type Combined is 'Non Consumable'
data.loc[data['Item_Type_Combined']=='Non-Consumable','Item_Fat_Content']='Non-Edible'
print(data['Item_Fat_Content'].value_counts())

#Import library:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()#New variable for outlet
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
le = LabelEncoder()
for i in var_mod:
    data[i] = le.fit_transform(data[i])

#One Hot Coding:
data = pd.get_dummies(data, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',
                                     'Item_Type_Combined','Outlet'])
data.dtypes
data[['Item_Fat_Content_0','Item_Fat_Content_1','Item_Fat_Content_2']].head(10)

#Drop the columns which have been converted to different types:
data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)

#Divide into test and train:
train = data.loc[data['source']=="train"]
test = data.loc[data['source']=="test"]

#Drop unnecessary columns:
test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)

#Export files as modified versions:
train.to_csv("C:\\Users\AJAY\\Documents\\BIG MART SALES\\train_modified.csv",index=False)
test.to_csv("C:\\Users\AJAY\\Documents\\BIG MART SALES\\test_modified.csv",index=False)

#Data Modelling - Baseline Model
#Mean based:
mean_sales = train['Item_Outlet_Sales'].mean()
mean_sales

#Define a dataframe with IDs for submission:
base1 = test[['Item_Identifier','Outlet_Identifier']]
base1['Item_Outlet_Sales'] = mean_sales

#Export submission file
base1.to_csv("alg0.csv",index=False)

#Define target and ID columns:
target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier','Outlet_Identifier']
from sklearn import cross_validation, metrics
def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])

    #Perform cross-validation:
    cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20, scoring='neg_mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    
    #Print model report:
    print ("\nModel Report:")
    print ("RMSE : %.4g" %np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)))
    print ("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" %(np.mean(cv_score),np.std(cv_score),
                                                                             np.min(cv_score),np.max(cv_score)))
    
    #Predict on testing data:
    dtest[target] = alg.predict(dtest[predictors])
    
    #Export submission file:
    IDcol.append(target)
    submission = pd.DataFrame({ x: dtest[x] for x in IDcol})
    submission.to_csv(filename, index=False)


from sklearn.linear_model import LinearRegression, Ridge, Lasso
predictors = [x for x in train.columns if x not in [target]+IDcol]
# print predictors
alg1 = LinearRegression(normalize=True)
modelfit(alg1, train, test, predictors, target, IDcol, 'alg1.csv')
coef1 = pd.Series(alg1.coef_, predictors).sort_values()

predictors = [x for x in train.columns if x not in [target]+IDcol]
alg2 = Ridge(alpha=0.05,normalize=True)
modelfit(alg2, train, test, predictors, target, IDcol, 'alg2.csv')
coef2 = pd.Series(alg2.coef_, predictors).sort_values()
coef2.plot(kind='bar', title='Model Coefficients')

from sklearn.tree import DecisionTreeRegressor
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg3 = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
modelfit(alg3, train, test, predictors, target, IDcol, 'alg3.csv')
coef3 = pd.Series(alg3.feature_importances_, predictors).sort_values(ascending=False)
coef3.plot(kind='bar', title='Feature Importances')


from sklearn.ensemble import RandomForestRegressor
predictors = [x for x in train.columns if x not in [target]+IDcol]
alg5 = RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4)
modelfit(alg5, train, test, predictors, target, IDcol, 'alg5.csv')
coef5 = pd.Series(alg5.feature_importances_, predictors).sort_values(ascending=False)
coef5.plot(kind='bar', title='Feature Importances')


predictors = [x for x in train.columns if x not in [target]+IDcol]
alg6 = RandomForestRegressor(n_estimators=400,max_depth=6, min_samples_leaf=100,n_jobs=4)
modelfit(alg6, train, test, predictors, target, IDcol, 'alg6.csv')
coef6 = pd.Series(alg6.feature_importances_, predictors).sort_values(ascending=False)
coef6.plot(kind='bar', title='Feature Importances')


