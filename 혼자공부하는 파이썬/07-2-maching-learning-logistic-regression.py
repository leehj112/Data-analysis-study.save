# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:15:37 2024

@author: leehj
"""

#!/usr/bin/env python
# coding: utf-8

# # 07-2 머신러닝으로 예측하기

#%%

# 머신러닝 패키지 : 사이킷런
# pip install scikit-learn

#%%
###############################################################################
# ## 카테고리 예측하기: 로지스틱 회귀
# 로지스틱 회귀: 카테고리 예측하기
# 카테고리: 
#   - 이진분류(binary classification) : 2개의 카테고리 구분, 0 or 1
#   - 다중분류(multiclass classification) : 3개 이상의 카테코리로 구분, 0~1
#   - 타겟 카테고리: 클래스(class)
#   - 음성클래스(negative class) : 0
#   - 양성클래스(positive class) : 1
###############################################################################

#%%

import pandas as pd

ns_book7 = pd.read_csv('./data/ns_book7.csv', low_memory=False)
ns_book7.head()


# In[3]:


from sklearn.model_selection import train_test_split

# 훈련세트: 75%
# 테스트세트: 25%
train_set, test_set = train_test_split(ns_book7, random_state=42)


# In[4]:

ns_book7_len = len(ns_book7)
train_set_len = len(train_set)
test_set_len = len(test_set)

print('ns_book7_len :', ns_book7_len) # 376770
print('train_set_len :', train_set_len, round(train_set_len / ns_book7_len, 2)) # 282577 0.75
print('test_set_len :', test_set_len, round(test_set_len / ns_book7_len, 2)) # 94193 0.25

#%%

ns_book7['대출건수'].mean()  # 11.593438968070707
train_set['대출건수'].mean() # 11.598732380908567
test_set['대출건수'].mean()  # 11.577558841952163

#%%

# 훈련 데이터
X_train = train_set[['도서권수']] # 입력
y_train = train_set['대출건수']   # 타깃(정답)

print(X_train.shape, y_train.shape) # (282577, 1) (282577,)

#%%

# 검증 세트
X_test = test_set[['도서권수']] # 예측용 데이터
y_test = test_set['대출건수']   # 정답용 데이터


# In[6]:

# 전체 대출건수 평균
borrow_mean = ns_book7['대출건수'].mean()
print(borrow_mean) # 11.593438968070707

#%%

# 전체 대출건수 평균보다 크면 True 아니면 False

# 훈련 데이터 정답
y_train_c = y_train > borrow_mean 

# 검증 데이터 정답
y_test_c = y_test > borrow_mean


#%%    

# 로지스틱 회귀 모델 : 분류

# '도서권수'로 '대출건수'가 평균보다 높은지 아닌지 예측하는 이진분류
# True(1), False(0)으로 처리됨
from sklearn.linear_model import LogisticRegression

# 모델 생성
logr = LogisticRegression()

# 훈련
logr.fit(X_train, y_train_c)

# 성능측정 : 정확도
# 정확도: 정답을 맞힌 비율
logr.score(X_test, y_test_c) # 0.7106154385145393

# 결과 약 71% 맞춤
# 문제점: y_test_c에 있는 음성 클래스와 양성 클래스 비율이 비슷하지 않다.    

# In[12]:

# 양성 클래스와 음성 클래스의 비율
y_test_c.value_counts()
"""
False    65337
True     28856
Name: 대출건수, dtype: int64
"""

#%%
y_test_c_len = len(y_test_c)
print("검증세트의 건수:", y_test_c_len) # 94193

y_test_counts = y_test_c.value_counts()
print("True:  {}, {}".format(y_test_counts.loc[True], y_test_counts.loc[True] / y_test_c_len))   # 28856, 0.3063497287484208
print("False: {}, {}".format(y_test_counts.loc[False], y_test_counts.loc[False] / y_test_c_len)) # 65337, 0.6936502712515792
# 음성 : 69%
# 양성 : 31%
# X_test에 있는 샘플을 무조건 음성 클래스로 예측해도 69%는 맞출 수 있다.

# In[13]:

# 더미모델(summy model) : 가장 많은 클래스로 무조건 예측을 수행
# 분류모델
from sklearn.dummy import DummyClassifier

dc = DummyClassifier()
dc.fit(X_train, y_train_c)
dc.score(X_test, y_test_c) # 0.6936502712515792

# 결과(0.6936502712515792)는 y_test_c의 분포에서 음성비율(0.6936502712515792)과 같다.
# 그러므로 이 값이 모델을 만들 때 기준점이 된다.
# 적어도 이 점수 보다 높지 않다면 유용한 모델이라고 할 수 없다.

#%%

# 예측
y_pred = logr.predict(X_test)

#%%

# 예측 확률
y_pred_proba = logr.predict_proba(X_test)


#%%

from sklearn.metrics import mean_absolute_error, mean_squared_error

# 평균절댓값오차 : MAE
mae = mean_absolute_error(y_test, y_pred) 
print(mae) # 11.47555550837111

