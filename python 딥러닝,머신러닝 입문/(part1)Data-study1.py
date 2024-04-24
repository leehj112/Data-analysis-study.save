# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:19:39 2024

@author: leehj
"""

x = [-3, 31, -11, 4, 0, 22, -2, -5, -25, -14]
y = [-2, 32, -10, 5, 1, 23, -1, -4, -24, -13]


import matplotlib.pyplot as plt 
plt.plot(x, y)
plt.show()


#%% 

import pandas as pd
df = pd.DataFrame({'X':x, 'Y':y}) 
df.shape 

# x리스트 안에 X열의 데이터가 되고 y리스트 Y 열의 데이터로 변환 

#%%

df.head() 


"""
    X   Y
0  -3  -2
1  31  32
2 -11 -10
3   4   5
4   0   1
"""


#%% 
# (10, 2)

train_features =['X']
target_cols = ['Y']
x_train = df.loc[:, train_features]
y_train = df.loc[:, target_cols]
print(x_train.shape, y_train.shape) 
# (10, 1) (10, 1)
#%% 
# 모델 학습 

from sklearn.linear_model import LinearRegression
lr = LinearRegression() 
lr.fit(x_train, y_train) 

#  LinearRegression
# 모델 인스턴스 객체를 생성하고 lr변수에 저장 
# fit 메소드 입력 데이터를 모델에 전달하여 학습 

#%% 
# lr 모델 인스턴스 객체의 coef_ 속성으로 부터 x 변수의 회귀 계수를 얻을 수 있다. 
# intercept_ 속성은 상수항(y 절편)
lr.coef_, lr.intercept_

# (array([[1.]]), array([1.]))

#%% 
print("기울기:", lr.coef_[0][0]) 
print("y절편:", lr.intercept_[0])


""" 
 기울기: 0.9999999999999999
y절편: 0.9999999999999999
"""

#%% 

import numpy as np 
x_new = np.array(11).reshape(1,1) 
lr.predict(x_new) 

# array([[12.]])

#%% 

x_test = np.arange(11, 16, 1).reshape(-1, 1)
x_test 

"""
array([[11],
       [12],
       [13],
       [14],
       [15]])
"""

y_pred = lr.predict(x_test)
y_pred 

"""
array([[12.],
       [13.],
       [14.],
       [15.],
       [16.]])
"""


















#%% 
# Classification 붓꼿의 품종 판별 
# 붓꼿에 데이터셋을 학습하여 품종을 판별하는 모델 
# 분류 모델 구조화, 모델 학습 및 성능 개선 프로세스 


import pandas as pd 
import numpy as np 

from sklearn import datasets
iris = datasets.load_iris() 

iris.keys() 

"""
Out[21]: dict_keys(['data', 'target', 'frame', 'target_names',
                    'DESCR', 'feature_names', 'filename', 'data_module'])

"""
# DESCR 키를 이용하여 데이터 셋 설명 출력 
print(iris['DESCR']) 

"""
.. _iris_dataset:

Iris plants dataset
--------------------

**Data Set Characteristics:**

:Number of Instances: 150 (50 in each of three classes)   # 150개에 붓꽃 샘플 데이터 
:Number of Attributes: 4 numeric, predictive attributes and the class
:Attribute Information:
    - sepal length in cm                    # 4개의 피처에는 꽃바침과 곷잎에 대한 각각의 가로 길이,세로 길이 
    - sepal width in cm
    - petal length in cm
    - petal width in cm
    - class:
            - Iris-Setosa                   # 3가지 범주에 속하는 붓꽃 품종                
            - Iris-Versicolour
            - Iris-Virginica

:Summary Statistics:

============== ==== ==== ======= ===== ====================
                Min  Max   Mean    SD   Class Correlation
============== ==== ==== ======= ===== ====================
sepal length:   4.3  7.9   5.84   0.83    0.7826
sepal width:    2.0  4.4   3.05   0.43   -0.4194
petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
============== ==== ==== ======= ===== ====================

:Missing Attribute Values: None
:Class Distribution: 33.3% for each of 3 classes.
:Creator: R.A. Fisher
:Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
:Date: July, 1988

The famous Iris database, first used by Sir R.A. Fisher. The dataset is taken
from Fisher's paper. Note that it's the same as in R, but not as in the UCI
Machine Learning Repository, which has two wrong data points.


"""


#%% 
# taeget 키를 이용하여 목표 변수 데이터를 확인 
# 150개 1차원 배열에 들어 있다. class:0, 1, 2 각각 50개씩 들어 있다. 

print("데이터셋 크기", iris['target'].shape) 

print("데이터셋 내용:\n", iris['target']) 

"""
데이터셋 크기 (150,)
데이터셋 내용:
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
"""

#%% 
# 데이터 속성의 데이터셋 크기 
print("데이터셋 크기:", iris['data'].shape)

# data속성의 데이터셋 내용 
print("데이터셋 내용:\n", iris['data'][:7, :]) 

"""
데이터셋 크기: (150, 4)
데이터셋 내용:
 [[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 [4.7 3.2 1.3 0.2]
 [4.6 3.1 1.5 0.2]
 [5.  3.6 1.4 0.2]
 [5.4 3.9 1.7 0.4]
 [4.6 3.4 1.4 0.3]]
"""

#%% 
# 데이터 프레임 변환 
df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
print("데이터프레임의 형태:", df.shape)
df.head() 

"""
         sepal length (cm)  sepal width (cm)       petal length (cm)   petal width (cm)
0                5.1               3.5                1.4               0.2
1                4.9               3.0                1.4               0.2
2                4.7               3.2                1.3               0.2
3                4.6               3.1                1.5               0.2
4                5.0               3.6                1.4               0.

"""

#%%
# 2행 추출 
df.columns = ['sepal_length','sepal_width','petal_length','petal_width']
df.head(2)

"""
      sepal_length  sepal_width  petal_length  petal_width
0           5.1          3.5           1.4          0.2
1           4.9          3.0           1.4          0.2
"""

#%% 
# Target: 오른쪽 열을 새롭게 추가 

df['Target'] = iris['target']
print('데이터셋의 크기:', df.shape)
df.head() 

"""
      sepal_length  sepal_width  petal_length  petal_width  Target
0           5.1          3.5           1.4          0.2       0
1           4.9          3.0           1.4          0.2       0
2           4.7          3.2           1.3          0.2       0
3           4.6          3.1           1.5          0.2       0
4           5.0          3.6           1.4          0.2       0
"""

#%% 
# 데이터 기본 정보
df.info() 
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 5 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   sepal_length  150 non-null    float64
 1   sepal_width   150 non-null    float64
 2   petal_length  150 non-null    float64
 3   petal_width   150 non-null    float64
 4   Target        150 non-null    int32  
dtypes: float64(4), int32(1)
memory usage: 5.4 KB
"""

#%%
df.describe() 
"""
        sepal_length  sepal_width  petal_length  petal_width      Target
count    150.000000   150.000000    150.000000   150.000000  150.000000
mean       5.843333     3.057333      3.758000     1.199333    1.000000
std        0.828066     0.435866      1.765298     0.762238    0.819232
min        4.300000     2.000000      1.000000     0.100000    0.000000
25%        5.100000     2.800000      1.600000     0.300000    0.000000
50%        5.800000     3.000000      4.350000     1.300000    1.000000
75%        6.400000     3.300000      5.100000     1.800000    2.000000
max        7.900000     4.400000      6.900000     2.500000    2.000000
"""

#%%
# 결측값 확인
df.isnull().sum() 
"""
sepal_length    0
sepal_width     0
petal_length    0
petal_width     0
Target          0
dtype: int64
"""

#%% 
# 중복 데이터 확인
df.duplicated().sum() 
# 1

#%%
# 중복 데이터 출력  
df.loc[df.duplicated(), :] 

"""
             sepal_length  sepal_width  petal_length  petal_width  Target
       142           5.8          2.7           5.1          1.9       2
"""

#%%
# 중복 데이터 모두 출력 
df.loc[(df.sepal_length==5.8)&(df.petal_width==1.9),:] 

"""
     sepal_length  sepal_width  petal_length  petal_width  Target
101           5.8          2.7           5.1          1.9       2
142           5.8          2.7           5.1          1.9       2

"""

#%% 
# 중북 데이터 제거 
df = df.drop_duplicates() 
df.loc[(df.sepal_length == 5.8) & (df.petal_width==1.9), :]

"""
 sepal_length  sepal_width  petal_length  petal_width  Target
101           5.8          2.7           5.1          1.9       2
"""
#%% 
# 데이터 시각화 
import  matplotlib.pyplot as plt 
import  seaborn as sns 
sns.set(font_scale=1.2)

sns.heatmap(data=df.corr(), square = True, annot = True, cbar = True)
plt.show() 

#%% 
# Target열: class 50 개씩 
df['Target'].value_counts() 
"""
Target
0    50
1    50
2    49
Name: count, dtype: int64
"""

#%% 
# hist 함수 
plt.hist(x='sepal_length', data=df)
plt.show() 



#%% 
# displot 함수 
sns.displot(x='sepal_width', kind='hist', data = df)
plt.show() 

#%%
# displot 함수 사용하여 -> kde 밀도 함수 그래프
sns.displot(x='petal_width', kind='kde', data = df)
plt.show() 

#%% 
# 꽃바침 길이 class0 이 제일 낮음(그래프) 
sns.displot(x='sepal_length', hue='Target', kind='kde', data = df)
plt.show() 

#%%
for col in['sepal_width','petal_length','potal_width']:
    sns.displot(x=col, hue='Target', kind='kde', data = df)

plt.show() 

#%% 
# paitplot을 사용하여 서로 다른 피 간에 관계를 나타내는 그래프 
# 데이터 분포를 품종별로 나타냄 

sns.pairplot(df, hue = 'Target', size = 2.5, diag_kind = 'kde')
plt.show() 

#%% 
from sklearn.model_selection import train_test_split
x_data = df.loc[:, 'sepal_length':'petal_width']
y_data = df.loc[:, 'Target']

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    test_size=0.2,               # 테스트 20% , rest: 80% 
                                                    shuffle=True,   # 지정한이유   # 무작위로 섞어서 데이터 추출 
                                                    random_state=20)             # 이 무작위로 추출한 데이터를 일정한 기준으로 분할 

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape) 

"""
(119, 4) (119,)   # 149개 샘플 중 80% 119개 훈련 데이터 
(30, 4) (30,)     # 30개 샘플 중 20% 30개 테스트 데이터 
"""













#%% 
# 분류 알고리즘1 KNN
# 예측하려는 x값이 주어질때 기존에 있는 비슷한 k값 개의 이웃을 찾는다
# x를 둘러싼 k개의 가장 가까운 이웃을 찾고, 이웃 데이터가 가장 많이 속해 있는 목표 class를 예측값으로 결정 
# add: k에 따라 값이 달라질 수 있음 

#%% 
# k = 7로 하는 knn모델 정의 
# fit 매소드 룬련 데이터를 입력 --> 모델 학습 

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train, y_train)
y_knn_pred = knn.predict(x_test)
print("예측값", y_knn_pred[:5]) 

# 예측값 [0 1 1 2 1]

#%% 
# 성능 평가
from sklearn.metrics import accuracy_score 
knn_acc = accuracy_score(y_test, y_knn_pred)
print("Accuracy:%.4f" % knn_acc) 

# 정확도 산출 
# Accuracy:0.9667  -> 93.33% 






#%% 
# 분류 알고리즘2 SVM
# dataset에 각 피처 열 백터들이 고유 축을 갖는 백터 공간을 이룬다고 가정 
# 각 데이터가 속하는 목표 클래스별로 군집을 이룬다고 생각해 본다
# 이때 각 군집까지에 거리를 최대한 멀리 유지하는 경계면을 찾고 각 군집을 서로 확연하게 구분 
# 각 군집을 구분하는 경계면을 찾으면 new data가 주어졌을 때 백터 공간의 좌표에서 어느 군집에 속하는지 분류를 할 수 있는 알고리즘
# svm 모듈에서 SVC 인스터스 객체를 통해 모델을 학습



#%% 
# 분류알고리즘3 로지스틱 회귀 
# 1: near ==> class 
# 0: near ==> x 

from sklearn.linear_model import LogisticRegression
lrc = LogisticRegression() 
lrc.fit(x_train, y_train) 

"""
Out[50]: LogisticRegression()  # locist check 
"""
#%% 
y_lrc_pred = lrc.predict(x_test) 
print("예측값:", y_lrc_pred[:5])

lrc_acc = accuracy_score(y_test, y_lrc_pred) 
print("Accuracy:%.4f" % lrc_acc) 

"""
예측값: [0 1 1 2 1]
Accuracy:1.0000
"""
#%% 
# predict_proba use 각 클래스에 속하는 확률값을 예측 
# 3개의 열과 30개의 행으로 구성된 넘파이 배열 

# 
y_lrc_prob = lrc.predict_proba(x_test)
y_lrc_prob 

"""
array([[9.83138240e-01, 1.68617026e-02, 5.74279548e-08], # 1번 째 행에서 0.98 확률 big --> class 0으로 분류
       [4.60594586e-03, 8.41666386e-01, 1.53727668e-01],
       [1.03265565e-02, 9.20315922e-01, 6.93575219e-02],
       [2.57762694e-05, 5.16128865e-02, 9.48361337e-01],
       [2.39278471e-02, 9.52074353e-01, 2.39977996e-02],
       [2.94377266e-02, 9.25886850e-01, 4.46754238e-02],
       [4.77469705e-06, 1.77365389e-02, 9.82258686e-01],
       [9.80408030e-01, 1.95918567e-02, 1.13624849e-07],
       [1.22276963e-05, 6.68235172e-02, 9.33164255e-01],
       [9.75409888e-01, 2.45900476e-02, 6.39734865e-08],
       [2.73409151e-05, 2.70273040e-02, 9.72945355e-01],
       [1.70554907e-03, 7.50416292e-01, 2.47878158e-01],
       [7.54444245e-04, 4.92528980e-01, 5.06716576e-01],
       [9.84507883e-01, 1.54920889e-02, 2.81178204e-08],
       [9.76897366e-01, 2.31025701e-02, 6.38784843e-08],
       [1.28695718e-03, 2.60123455e-01, 7.38589588e-01],
       [9.91829685e-01, 8.17030426e-03, 1.11423256e-08],
       [4.13706985e-03, 8.61510769e-01, 1.34352161e-01],
       [1.81181786e-04, 7.21481433e-02, 9.27670675e-01],
       [1.94275000e-01, 8.02284548e-01, 3.44045225e-03],
       [3.00102729e-03, 8.17185104e-01, 1.79813869e-01],
       [8.82111123e-04, 2.85598483e-01, 7.13519406e-01],
       [1.46887126e-04, 1.63725340e-01, 8.36127773e-01],
       [9.47634969e-01, 5.23647319e-02, 2.99612897e-07],
       [1.20402645e-03, 5.93658993e-01, 4.05136980e-01],
       [5.41729876e-02, 9.36155619e-01, 9.67139327e-03],
       [3.02482604e-01, 6.95875810e-01, 1.64158657e-03],
       [9.69638269e-01, 3.03616045e-02, 1.26557391e-07],
       [9.36080786e-06, 2.60851245e-02, 9.73905515e-01],
       [2.16599214e-06, 3.16629285e-02, 9.68334906e-01]])
"""










#%%
# 분류 알고리즘 4 -의사결정 나무 
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth=3, random_state=20)
dtc.fit(x_train, y_train) 

"""
Out[62]: DecisionTreeClassifier(max_depth=3, random_state=20)
"""

y_dtc_pred = dtc.predict(x_test)
print("예측값:", y_dtc_pred[:5]) 

dtc_acc = accuracy_score(y_test, y_dtc_pred)
print("Accuracy:%.4f" % dtc_acc) 

"""
예측값: [0 1 1 2 1]
Accuracy:0.9333  # 테스트 데이터 predict 메소드 함수로 93.33% 정확도 도출 
"""


