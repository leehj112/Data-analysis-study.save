# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:15:20 2024

@author: leehj
"""

#!/usr/bin/env python
# coding: utf-8

# # 07-2 머신러닝으로 예측하기

#%%

# 머신러닝 패키지 : 사이킷런
# pip install scikit-learn

#%%
"""
# 모델(model)
학습된 패턴을 저장하는 소프트웨어 객체를 의미한다.
사이킷런에서는 어떤 클래스의 인스턴스 객체가 모델이다.

# 지도학습(supervised learning)
데이터에 있는 각 샘플에 대한 정답을 알고 있는 경우이다.
정답을 보통 타깃(target)이라 한다.

# 입력(input)
입력은 타깃을 맞추기 위해 모델이 재료로 사용하는 데이터이다.

# 비지도학습(un-supervised learning)
입력 데이터는 있지만 타깃이 없는 경우이다.
대표적으로 군집 알고리즘(clustering algorithm)이 있다.
"""

#%%
# ## 모델 훈련하기

import gdown

gdown.download('https://bit.ly/3pK7iuu', './data/ns_book7.csv', quiet=False)


# In[2]:


import pandas as pd

ns_book7 = pd.read_csv('./data/ns_book7.csv', low_memory=False)
ns_book7.head()


# In[3]:


from sklearn.model_selection import train_test_split

# 훈련세트: train_set,75%
# 테스트세트: test_set, 25%
train_set, test_set = train_test_split(ns_book7, random_state=42)


# In[4]:

ns_book7_len = len(ns_book7)
train_set_len = len(train_set)
test_set_len = len(test_set)

print('ns_book7_len :', ns_book7_len) # 376770
print('train_set_len :', train_set_len, round(train_set_len / ns_book7_len, 2)) # 282577 0.75
print('test_set_len :', test_set_len, round(test_set_len / ns_book7_len, 2))    # 94193 0.25

#%%

# 각 데이터셋의 '대출건수'의 평균
ns_book7['대출건수'].mean()  # 11.593438968070707
train_set['대출건수'].mean() # 11.598732380908567
test_set['대출건수'].mean()  # 11.577558841952163

#%%

# '도서권수'로 '대출건수' 예측?

# 데이터셋 -> 머신러닝에 전달한 데이터 추출 
X_train = train_set[['도서권수']] # 입력 : 데이터프레임(2차원)
y_train = train_set['대출건수']   # 정답 : 시리즈(1차원)

print(X_train.shape, y_train.shape) # (282577, 1) (282577,)


# In[6]:

# 선형회귀모델: LinearRegression
from sklearn.linear_model import LinearRegression

# 선형회귀모델 생성
lr = LinearRegression()

# 훈련
lr.fit(X_train, y_train)


#%%

# ## 훈련된 모델을 평가하기: 결정계수
# 평가 : score()

# 검증 데이터:
X_test = test_set[['도서권수']] # 입력
y_test = test_set['대출건수']   # 정답

# 결정계수(Coefficient of Determination)
# R² = 1 - (타깃 - 예측)² / (타깃 - 평균)²

# 결정계수가 1에 가까울 수록 도서권수와 대출건수에 관계가 깊다고 볼수 있다.
r2 = lr.score(X_test, y_test) 
print("결정계수(r2): ", r2) # 0.10025676249337057

# 결과: 0.1은 좋은 점수가 아니다.
# 평균은 타깃의 평균을 의미한다.
# 예측이 평균에 가까워지면 분모와 분자가 같아져 R² 점수는 0이 된다.


#%%

# 기울기: 12.87648822
# 절편: -3.1455454195820653
print(lr.coef_, lr.intercept_) # [12.87648822] -3.1455454195820653

# In[8]:

# 대출건수(y_train)로 대출건수를 훈련하여 예측?
# 즉 정답 데이터로 훈련을 수행
# 결과: 결정계수 1.0
lr2 = LinearRegression()

# y_train은 시리즈 객체이므로 2차원 배열 형태로 변환
X_train2 = y_train.to_frame() # 대출건수: 훈련 데이터
y_test2 = y_test.to_frame()   # 대출건수: 테스트 데이터

# lr.fit(y_train.to_frame(), y_train)
# lr.score(y_test.to_frame(), y_test) # 1.0
lr2.fit(X_train2, y_train)
lr2.score(y_test2, y_test) # 1.0


#%%
# ## 연속적인 값 예측하기: 선형 회귀
# 기울기: 1
# 절편: -1.2647660696529783e-12, 0에 가까운 매우 작은 음수
# y = 1 * x + 0
# 이 식의 x에 어떤 값을 입력하여도 x와 y는 동일한 값이 된다.

print(lr2.coef_, lr2.intercept_) # [1.] -1.2647660696529783e-12
# 기울기: 1.0
# 절편: 0 

# 해설:
# 대출건수로 대출건수를 예측하는 것은 의미가 없다.
# 선형회귀 알고리즘은 입력에 기울기를 곱하고 y축과 만나는 절편을 더하여 예측을 만드는 것이다.    

#%%
###############################################################################
# 평균제곱오차와 평균절댓값오차로 모델 평가하기
# 평균제곱오차 : MSE(Mean Squared Error)
# 평균절대오차 : MAE(Mean Absolute Error)
###############################################################################

from sklearn.metrics import mean_absolute_error, mean_squared_error

#%%

# 예측 : 검증데이터로 '대출건수' 예측
y_pred = lr.predict(X_test)

#%%

# 평균절댓값오차 : MAE
# MAE = 합(절대값(타깃 - 예측)) / n
# MAE = mean_absolute_error(타깃, 예측)
mae = mean_absolute_error(y_test, y_pred) 
print(mae) # 10.358091752853873

# 결과 : 10.358091752853873
# 결과과 타깃의 평균 정도의 오차가 발생

#%%

# 타깃의 평균
y_test.mean() # 11.577558841952163


#%%

# 모집단(전체)의 '대출건수'의 평균을 구함
ns_book7['대출건수'].mean() # 11.593438968070707

#%%

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 폰트설정
# font_files = fm.findSystemFonts(fontpaths=['C:/Windows/Fonts'])
font_files = fm.findSystemFonts(fontpaths=['C:/Users/Solero/AppData/Local/Microsoft/Windows/Fonts'])
for fpath in font_files:
    print(fpath)
    fm.fontManager.addfont(fpath)

# plt.rcParams['font.family'] = 'NanumGothic'
# plt.rcParams['font.family'] = 'NanumSquare'    
plt.rcParams['font.family'] = 'NanumBarunGothic'   

#%%
# 산점도
fig, ax = plt.subplots(figsize=(10,10))
ax.scatter(ns_book7['도서권수'], ns_book7['대출건수'])
ax.set_title("도서권수 대비 대출건수")
ax.set_xlabel("도서권수")
ax.set_ylabel("대출건수")
fig.show()
