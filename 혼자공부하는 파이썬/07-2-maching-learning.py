# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:16:22 2024

@author: leehj
"""

#!/usr/bin/env python
# coding: utf-8

# # 07-2 머신러닝으로 예측하기

# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://nbviewer.jupyter.org/github/rickiepark/hg-da/blob/main/07-2.ipynb"><img src="https://jupyter.org/assets/share.png" width="61" />주피터 노트북 뷰어로 보기</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/rickiepark/hg-da/blob/main/07-2.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />구글 코랩(Colab)에서 실행하기</a>
#   </td>
# </table>

# ## 모델 훈련하기

# In[1]:


import gdown

gdown.download('https://bit.ly/3pK7iuu', 'ns_book7.csv', quiet=False)


# In[2]:


import pandas as pd

ns_book7 = pd.read_csv('ns_book7.csv', low_memory=False)
ns_book7.head()


# In[3]:


from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(ns_book7, random_state=42)


# In[4]:


print(len(train_set), len(test_set))


# In[5]:


X_train = train_set[['도서권수']]
y_train = train_set['대출건수']

print(X_train.shape, y_train.shape)


# In[6]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)


# ## 훈련된 모델을 평가하기: 결정계수

# In[7]:


X_test = test_set[['도서권수']]
y_test = test_set['대출건수']

lr.score(X_test, y_test)


# In[8]:


lr.fit(y_train.to_frame(), y_train)
lr.score(y_test.to_frame(), y_test)


# ## 연속적인 값 예측하기: 선형 회귀

# In[9]:


print(lr.coef_, lr.intercept_)


# ## 카테고리 예측하기: 로지스틱 회귀

# In[10]:


borrow_mean = ns_book7['대출건수'].mean()
y_train_c = y_train > borrow_mean
y_test_c = y_test > borrow_mean


# In[11]:


from sklearn.linear_model import LogisticRegression

logr = LogisticRegression()
logr.fit(X_train, y_train_c)
logr.score(X_test, y_test_c)


# In[12]:


y_test_c.value_counts()


# In[13]:


from sklearn.dummy import DummyClassifier

dc = DummyClassifier()
dc.fit(X_train, y_train_c)
dc.score(X_test, y_test_c)


# ## 평균제곱오차와 평균절댓값오차로 모델 평가하기

# In[14]:


lr.fit(X_train, y_train)


# In[15]:


y_pred = lr.predict(X_test)


# In[16]:


from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test, y_pred)


# In[17]:


y_test.mean()
