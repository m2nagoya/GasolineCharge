# -*- coding: utf-8 -*-

import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import re
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from datetime import datetime
from fbprophet import Prophet

### 日数算出
def convert (year,month,day) :
    dt1 = datetime(int(year),1,1)
    dt2 = datetime(int(year),int(month),int(day))
    dt3 = dt2 - dt1
    return dt3.days

### CSV読み込み
# train = pd.read_csv("data/train2.csv",index_col='year',encoding="SHIFT-JIS")
train = pd.read_csv("data/train.csv")
# test  = pd.read_csv("data/test.csv")

### 文字列置換
train = train.rename(columns={u'調査日':'ds',u'徳  島':'y'})

### 日数に変換
# for j, columns in enumerate(train.columns) :
#     if columns != 'year' :
#         for i, md in enumerate(train[columns]) :
#             for year in train['year'] :
#             # for year in train.index :
#                 if isinstance(md,(int,float,str)) is False :
#                     md = md.encode('utf-8')
#                 if pd.isnull(md) is False :
#                     month = re.sub(r'月.*','',md)
#                     day = re.sub(r'日.*','',re.sub(r'.*月','',md))
#                     sum = convert(int(year),int(month),int(day))
#                     train.iat[i,j] = int(sum)

## Category Encorder
# for column in ['2019']:
#     le = LabelEncoder()
#     le.fit(train[column])
#     train[column] = le.transform(train[column])

### OneHot Encording
# oh_area = pd.get_dummies(train.area)
# train.drop(['area'], axis=1, inplace=True)
# train = pd.concat([train,oh_area], axis=1)
# _, i = np.unique(train.columns, return_index=True)
# train = train.iloc[:, i]

### データセットの作成 (説明変数 -> X, 目的変数 -> Y)
train = train[['ds','y']]

# ### 履歴をシフト
# for i in range(1,6):
#     train['sft%s'%i] = train['tokushima'].shift(i)
#
# ### 1階微分
# train['drv1'] = train['sft1'].diff(1)
#
# ### 2階微分
# train['drv2'] = train['sft1'].diff(1).diff(1)
#
# ### 移動平均値
# train['mean'] = train['sft1'].rolling(6).mean()

### NaNの削除
train = train.dropna()

### int型に変換
# for columns in train.columns :
#     if columns is not 'mean' :
#         train[columns] = train[columns].astype(int)

# X = train.drop('tokushima', axis=1)
# y = train['tokushima']
# print('X shape: {}, y shape: {}'.format(X.shape, y.shape))

### データセットの分割
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
# X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

# print("LinearRegression")
# model = LinearRegression()
# model.fit(X_train,y_train)
# print(model.score(X_train,y_train))

# print("LogisticRegression")
# model = LogisticRegression()
# model.fit(X_train,y_train)
# print(model.score(X_train,y_train))
#
# print("SVM")
# model = SVC()
# model.fit(X_train, y_train)
# predicted = model.predict(X_test)
# print(metrics.accuracy_score(y_test, predicted))
# print("GridSearch")
# best_score = 0
# for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
#     for C in [0.001, 0.01, 0.1, 1, 10, 100]:
#         print(str(gamma) + "," + str(C))
#         svm = SVC(gamma=gamma, C=C)
#         svm.fit(X_train, y_train.values.ravel())
#         score = svm.score(X_test, y_test)
#         if score > best_score:
#             best_score = score
#             best_parameters = {'C':C, 'gamma':gamma}
# print("Best score: " + str(best_score))
# print("Best parameters: " + str(best_parameters))

# print("RandomForest")
# model = RandomForest(n_estimators=100).fit(X_train, y_train)
# print(model.score(X_test, y_test))

# print("LightGBM")
# train = lgb.Dataset(X_train, y_train)
# test = lgb.Dataset(X_validation, y_validation, reference=train)
# params = {
#         'task' : 'train',
#         'boosting_type' : 'gbdt',
#         'objective' : 'regression',
#         'metric' : 'mae',
#         'num_leaves' : 31,
#         'learning_rate' : 0.1,
#         'feature_fraction' : 0.9,
#         'bagging_fraction' : 0.8,
#         'bagging_freq': 5,
#         'verbose' : 0,
#         'min_chile_samples' : 3
# }
# evaluation_results = {}
# model = lgb.train(params,
#                 train,
#                 num_boost_round=100,
#                 valid_sets=test,
#                 valid_names='test',
#                 evals_result=evaluation_results,
#                 early_stopping_rounds=10
#                 )
# y_pred = model.predict(X_test, num_iteration=model.best_iteration)
# y_graph = np.concatenate([np.array([None for i in range(len(y_train)+len(y_validation))]) , y_pred])
# y_graph = pd.DataFrame(y_graph, index=X.index)
# plt.figure(figsize=(10,5))
# plt.plot(y, label='original')
# plt.plot(y_graph, '--', label='predict')
# plt.legend()

print("prophet")
# pd.get_dummies(train['y'])
train['origin'] = train['y']
train['y'] = np.log(train['y'])
print(train)
model = Prophet()
model.fit(train)
future_data = model.make_future_dataframe(periods=12, freq='m')
forecast_data = model.predict(future_data)
model.plot(forecast_data)
model.plot_components(forecast_data)
plt.show()
