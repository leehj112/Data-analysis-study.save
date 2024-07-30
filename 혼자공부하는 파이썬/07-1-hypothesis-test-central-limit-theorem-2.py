# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:12:34 2024

@author: leehj
"""

#!/usr/bin/env python
# coding: utf-8

# # 07-1 통계적으로 추론하기
# 가설검정(Hypothesis Test)  : P397
# 순열검정(Permutation Test) : P402

### 통계적 의미 확인하기: 가설검정
# 귀무가설 또는 영가설(null hypothesis)
#   - H0
#   - 표본 사이에 통계적으로 의미가 없다고 예상되는 가설
#   - 가설: 파이썬과 C++ 도서의 평균 대출건수가 같다.
#
# 대립가설(alternative)
#   - Ha
#   - 표본 사이에 통계적으로 차이가 있다는 가설
#   - 가설: 파이썬과 C++ 도서의 평균 대출건수가 같지 않다.
#    
# 유의수준(significance level)
#   - p-value: 0.05(5%), 0.01(1%)
#   - 일반적으로 많이 사용하는 기준은 정규분포의 양쪽 꼬리 면적의 더해 5%가 되는 지점
#   - p-value가 0.05 미만일때 영가설 기각한다.
#   - z-score에 대한 기준을 유의수준이다.
#
# 유의확률(significance Probability, p-value) 단계
#   - 귀무가설(H0)과 대립가설(Ha) 설정
#   - 적절한 통계량 선택: z-통계량, t-통계량
#   - p-value 계산
# 
# 두 모집단의 평균에 대한 z-score
#   z = ((x1 - x2) - (u1 - u2)) / sqrt((s1**2 / n1) + (s2**2 / n2))
#   x1 : 파이썬의 표본의 평균
#   x2 : C++의 표본의 평균
#   u1 : 파이썬의 모집단의 평균
#   u2 : C++의 모집단의 평균
#   s1 : 파이썬의 표본의 표준편차(표준오차)
#   s2 : C++의 표본의 표준편차(표준오차)
#
# 표준오차(Standard Error)
#   - 공식: 표본 평균의 표준편차 = 모집단의 표준편차 / 제곱근(표본에 포함된 샘플갯수)
#   - 표본의 표준편차와 모집단의 표준편차가 거의 동일(비슷)

#%%
import gdown
gdown.download('https://bit.ly/3pK7iuu', './data/ns_book7.csv', quiet=False)

#%%

import pandas as pd

ns_book7 = pd.read_csv('./data/ns_book7.csv', low_memory=False)
ns_book7.head()

#%%

import matplotlib.pyplot as plt
import numpy as np    

#%%

# 모집단(전체)의 '대출건수'의 평균을 구함
ns_book7['대출건수'].mean() # 11.593438968070707
   
#%%

# '파이썬' 도서의 대출건수를 사용해 신뢰구간을 계산
python_books_index = ns_book7['주제분류번호'].str.startswith('00') & \
                     ns_book7['도서명'].str.contains('파이썬')
python_books = ns_book7[python_books_index]
python_books.head()

#%%
len(python_books) # 251건

#%%

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

#%%

# ## 통계적 의미 확인하기: 가설검정

# In[25]:

# 도서 : C++
cplus_books_index = ns_book7['주제분류번호'].str.startswith('00') & \
                    ns_book7['도서명'].str.contains('C++', regex=False)
cplus_books = ns_book7[cplus_books_index]
cplus_books.head()


# In[26]:

len(cplus_books) # 89


# In[27]:

# 표본의 평균 : C++
cplus_mean = np.mean(cplus_books['대출건수'])
cplus_mean # 11.595505617977528

# In[28]:

# 표준오차(Standard Error)
cplus_se = np.std(cplus_books['대출건수'])/ np.sqrt(len(cplus_books))
cplus_se # 0.9748405650607009


# In[29]:

# python_mean : 14.749003984063744
# cplus_mean : 11.595505617977528
# python_se : 0.8041612072427442
# cplus_se : 0.9748405650607009

# z-score : 2.495408195140708
pc_z_score = (python_mean - cplus_mean) / np.sqrt(python_se**2 + cplus_se**2)
pc_z_score = round(pc_z_score, 2)
print(pc_z_score) # 2.5


# In[30]:

# 누적분포 : 0.9937903346742238
# stats.norm.cdf(2.50)
pc_norm_cdf = stats.norm.cdf(pc_z_score) 
pc_norm_cdf = round(pc_norm_cdf, 3) 
print(pc_norm_cdf) # 0.994


# In[31]:

p_value = (1 - 0.995) * 2
p_value # 0.01

#%%

p_value = (1 - pc_norm_cdf) * 2
p_value # 0.012

#%%

# True = p_value(0.012) < p_level(0.05)
# 파이썬과 C++ 도서의 평균에 차이가 있다.
# 결과 : 영가설을 기각한다.
p_level = 0.05
p_tf = p_value < p_level
print(p_tf) # True


#%%
###############################################################################
# t-검정으로 가설 검증하기
###############################################################################

# t-score, p-value
t, pvalue = stats.ttest_ind(python_books['대출건수'], cplus_books['대출건수'])
print(t, pvalue) # 2.1390005694958574 0.03315179520224784

# t: 2.1390005694958574
# pvalue: 0.03315179520224784

#%%

t_tf = pvalue < p_level
print(t_tf) # True

# True = pvalue(0.033) < p_level(0.05)
# 파이썬과 C++ 도서의 평균에 차이가 있다.
# 결과 : 영가설을 기각한다.

#%%

# [참고]
# t-score로 누적분포?
tpc_norm_cdf = stats.norm.cdf(t) 
tpc_norm_cdf = round(tpc_norm_cdf, 3) 
print(tpc_norm_cdf) # 0.984

p_value2 = (1 - tpc_norm_cdf) * 2
p_value2 # 0.03200000000000003

# t-score로 구한 값: pvalue2(0.03200000000000003)
# z-score로 구한 값: pvalue(0.03315179520224784)


#%%

###############################################################################
# ## 정규분포가 아닐 때 가설 검증하기: 순열검정(미모수검정)
###############################################################################

# 모집단의 분포가 정규분포가 아니거나 모집단의 분포를 알 수 없을 때 사용한다.
#   - 두 표본의 평균의 차이를 계산 후에 두 표본을 섞어서 무작위로 두 그룹을 나눔
#   - 두 그룹은 원래 표본의 크기와 동일하게 만든다.
#   - 두 그룹에서 다시 평균의 차이를 계산
#   - 위 과정을 여러번 반복한다.
#   - 원래 표본의 평균 차이가 무작위로 나눈 그룹의 평균 차이보다
#     크거나 작은 경우를 헤아려 p-value를 계산


# In[33]:

# 두 표본의 평균의 차이를 계산
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
# (3.1534983660862164, 0.022)
# 두 그룹의 평균의 차이: 3.1534983660862164
# p-value: 0.022

# 결과
# p-value(0.022) < 0.05보다 작다.
# 유의수준에 미치지 않으므로 영가설 기각한다.
# 두 도서의 평균 대출건수에는 차이가 있다.

#%%

# In[36]:

# scipy 1.8 버전 이상에서만 실행됩니다.
res = stats.permutation_test((python_books['대출건수'], cplus_books['대출건수']), 
                             statistic, random_state=42)

# 결과는 약 3.153 0.0258입니다.
print(res.statistic, res.pvalue) # 3.1534983660862164 0.0258

#%%

# 람다함수 이용 : lambda x, y : np.mean(x) - np.mean(y)
res1 = stats.permutation_test((python_books['대출건수'], cplus_books['대출건수']), 
                            lambda x, y : np.mean(x) - np.mean(y), random_state=42)

# 결과는 약 3.153 0.0258입니다.
print(res1.statistic, res1.pvalue) # 3.1534983660862164 0.0258


# In[36]:

# 도서대출 평균 비교하기: '파이썬' vs '자바스크립트'
java_books_indx = ns_book7['주제분류번호'].str.startswith('00') & \
                  ns_book7['도서명'].str.contains('자바스크립트')
java_books = ns_book7[java_books_indx]
java_books.head()


# In[37]:

# 파이썬 평균: 14.749003984063744

print(len(java_books), np.mean(java_books['대출건수']))
# 105 15.533333333333333


# In[38]:


permutation_test(python_books['대출건수'], java_books['대출건수'])
# (-0.7843293492695889, 0.566)

# 결과
# p-value(0.566)는 0.05 보다 크다
# 영가설을 기각할 수 없다.
# 파이썬과 자바스크립트 도서 사이의 대출건수 차이는 큰 의미가 없다.
