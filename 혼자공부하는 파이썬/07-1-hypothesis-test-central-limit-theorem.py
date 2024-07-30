# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:14:43 2024

@author: leehj
"""

#!/usr/bin/env python
# coding: utf-8

# # 07-1 통계적으로 추론하기
# 가설검정(Hypothesis Test)
# 순열검정(Permutation Test)

#%%
# 중심극한정리 알아보기

import gdown

gdown.download('https://bit.ly/3pK7iuu', './data/ns_book7.csv', quiet=False)


# In[8]:


import pandas as pd

ns_book7 = pd.read_csv('./data/ns_book7.csv', low_memory=False)
ns_book7.head()


# In[9]:


import matplotlib.pyplot as plt

# '대출건수'의 히스토그램
plt.hist(ns_book7['대출건수'], bins=50)
plt.yscale('log') # 로그 스케일
plt.show()


# In[10]:

import numpy as np    

np.random.seed(42)

# 1000번을 30건씩 샘플링하여 '대출건수'의 평균을 구함
sample_means = []
for _ in range(1000):
    m = ns_book7['대출건수'].sample(30).mean()
    sample_means.append(m)


# In[11]:

# 좌우 대칭이 완벽하지 않지만, 종 모양과 유사한 분포 형성
plt.hist(sample_means, bins=30)
plt.show()


# In[12]:

# 매직넘버 : 30
    
# 1000번을 30건씩 샘플링하여 '대출건수'의 평균을 구함
# 샘플 데이터의 총 평균
np.mean(sample_means) # 11.539900000000001


# In[13]:

# 모집단(전체)의 '대출건수'의 평균을 구함
ns_book7['대출건수'].mean() # 11.593438968070707


# In[14]:

# 1000번을 20건씩 샘플링하여 '대출건수'의 평균을 구함
np.random.seed(42)
sample_means = []
for _ in range(1000):
    m = ns_book7['대출건수'].sample(20).mean()
    sample_means.append(m)
    
np.mean(sample_means) # 11.39945


# In[15]:

#%%

# 1000번을 50건씩 샘플링하여 '대출건수'의 평균을 구함
np.random.seed(42)
sample_means = []
for _ in range(1000):
    m = ns_book7['대출건수'].sample(50).mean()
    sample_means.append(m)
    
np.mean(sample_means) # 11.53212


#%%
# 1000번을 60건씩 샘플링하여 '대출건수'의 평균을 구함
np.random.seed(42)
sample_means = []
for _ in range(1000):
    m = ns_book7['대출건수'].sample(60).mean()
    sample_means.append(m)
    
np.mean(sample_means) # 11.511583333333332

#%%

# 1000번을 40건씩 샘플링하여 '대출건수'의 평균을 구함
np.random.seed(42)
sample_means = []
for _ in range(1000):
    m = ns_book7['대출건수'].sample(40).mean()
    sample_means.append(m)
    
np.mean(sample_means) # 11.5613

#%%

# 샘플의 갯수에 따른 평균값
# 전체 : 11.5934
# 20건 : 11.3994
# 30건 : 11.5399(매직넘버)
# 40건 : 11.5613(전체에 가장 가까움)
# 50건 : 11.5321
# 60건 : 11.5115

#%%

# 샘플(40)의 평균의 표준편차
np.std(sample_means) # 3.0355987564235165

# In[17]:

# 표준오차(Standard Error)    
# 표본 평균의 표준편차 = 모집단의 표준편차 / 제곱근(표본에 포함된 샘플갯수)
np.std(ns_book7['대출건수']) / np.sqrt(40) # 3.048338251806833


#%%

###############################################################################
# 신뢰구간(Confidence Interval)
# ## 모집단의 평균 범위 추정하기: 신뢰구간
###############################################################################

# 전제조건:
# 만약 딱 하나의 표본이 있다면 모집단의 평균을 추정할 수 있는가?
# 신뢰구간은 표본의 파라미터(평균)가 모집단의 평균 속할 것이라 믿는 모집단의 파라미터 범위이다.
    
#%%

# '파이썬' 도서의 대출건수를 사용해 신뢰구간을 계산
python_books_index = ns_book7['주제분류번호'].str.startswith('00') & \
                     ns_book7['도서명'].str.contains('파이썬')
python_books = ns_book7[python_books_index]
python_books.head()


# In[19]:


len(python_books) # 251건


# In[20]:

# '파이썬' 도서의 대출건수 평균
python_mean = np.mean(python_books['대출건수'])
python_mean # 14.749003984063744


# In[21]:

# 중심극한정리의 표준오차
# 표준오차(Standard Error)    
# 표본 평균의 표준편차 = 모집단의 표준편차 / 제곱근(표본에 포함된 샘플갯수)
python_std = np.std(python_books['대출건수'])
python_se = python_std / np.sqrt(len(python_books))
python_se # 0.8041612072427442


# In[22]:

# 누적분포 z-score
from scipy import stats

# 0.975 = 1 - 0.025
# 평균을 중심으로 95% 영역의 좌우 각각 2.5% 구간
stats.norm.ppf(0.975) # 1.959963984540054


# In[23]:

stats.norm.ppf(0.025) # -1.9599639845400545


# In[24]:

# 표본의 평균    
# '파이썬' 도서의 대출건수 평균
# python_mean : 14.749003984063744

# 중심극한정리의 표준오차
# python_se : 0.8041612072427442

# 모집단 평균('대출건수') : 11.593438968070707

# 모집단의 평균 추측
print(python_mean - 1.96 * python_se, python_mean + 1.96 * python_se)
# 13.172848017867965 16.325159950259522
# 13.2에서 16.3 사이에 놓여 있다고 추측


#%%
# ## 통계적 의미 확인하기: 가설검정

# In[25]:


cplus_books_index = ns_book7['주제분류번호'].str.startswith('00') & \
                    ns_book7['도서명'].str.contains('C++', regex=False)
cplus_books = ns_book7[cplus_books_index]
cplus_books.head()


# In[26]:


len(cplus_books)


# In[27]:


cplus_mean = np.mean(cplus_books['대출건수'])
cplus_mean


# In[28]:


cplus_se = np.std(cplus_books['대출건수'])/ np.sqrt(len(cplus_books))
cplus_se


# In[29]:


(python_mean - cplus_mean) / np.sqrt(python_se**2 + cplus_se**2)


# In[30]:


stats.norm.cdf(2.50)


# In[31]:


p_value = (1-0.995)*2
p_value


# In[32]:


t, pvalue = stats.ttest_ind(python_books['대출건수'], cplus_books['대출건수'])
print(t, pvalue)


# ## 정규분포가 아닐 때 가설 검증하기: 순열검정

# In[33]:


def statistic(x, y):
    return np.mean(x) - np.mean(y)


# In[34]:


def permutation_test(x, y):
    # 표본의 평균 차이를 계산합니다.
    obs_diff = statistic(x, y)
    # 두 표본을 합칩니다.
    all = np.append(x, y)
    diffs = []
    np.random.seed(42)
    # 순열 검정을 1000번 반복합니다.
    for _ in range(1000):
        # 전체 인덱스를 섞습니다.
        idx = np.random.permutation(len(all))
        # 랜덤하게 두 그룹으로 나눈 다음 평균 차이를 계산합니다.
        x_ = all[idx[:len(x)]]
        y_ = all[idx[len(x):]]
        diffs.append(statistic(x_, y_))
    # 원본 표본보다 작거나 큰 경우의 p-값을 계산합니다.
    less_pvalue = np.sum(diffs < obs_diff)/1000
    greater_pvalue = np.sum(diffs > obs_diff)/1000
    # 둘 중 작은 p-값을 선택해 2를 곱하여 최종 p-값을 반환합니다.
    return obs_diff, np.minimum(less_pvalue, greater_pvalue) * 2


# In[35]:


permutation_test(python_books['대출건수'], cplus_books['대출건수'])


# In[36]:


# scipy 1.8 버전 이상에서만 실행됩니다.
# res = stats.permutation_test((python_books['대출건수'], cplus_books['대출건수']), 
#                              statistic, random_state=42)
# 결과는 약 3.153 0.0258입니다.
# print(res.statistic, res.pvalue)


# In[36]:


java_books_indx = ns_book7['주제분류번호'].str.startswith('00') & \
                  ns_book7['도서명'].str.contains('자바스크립트')
java_books = ns_book7[java_books_indx]
java_books.head()


# In[37]:


print(len(java_books), np.mean(java_books['대출건수']))


# In[38]:


permutation_test(python_books['대출건수'], java_books['대출건수'])
