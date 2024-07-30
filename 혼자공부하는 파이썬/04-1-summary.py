# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:10:38 2024

@author: leehj
"""

#!/usr/bin/env python
# coding: utf-8

# # 04-1 통계로 요약하기

# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://nbviewer.jupyter.org/github/rickiepark/hg-da/blob/main/04-1.ipynb"><img src="https://jupyter.org/assets/share.png" width="61" />주피터 노트북 뷰어로 보기</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/rickiepark/hg-da/blob/main/04-1.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />구글 코랩(Colab)에서 실행하기</a>
#   </td>
# </table>

# ## 기술통계 구하기

# In[1]:


import gdown

gdown.download('https://bit.ly/3736JW1', './data/ns_book6.csv', quiet=False)


# In[2]:


import pandas as pd

ns_book6 = pd.read_csv('./data/ns_book6.csv', low_memory=False)
ns_book6.head()


# In[3]:

# 요약통계
ns_book6.describe()


# In[4]:

sum(ns_book6['도서권수']==0)  # 3206

#%%

# '도서권수'의 비율
ns_book6_len = len(ns_book6)
ns_book6_sum = sum(ns_book6['도서권수']==0)  # 3206

ns_book6_rate = ns_book6_sum / ns_book6_len
print(f"{round(ns_book6_rate * 100, 3)}%") # 0.844%


# In[5]:

# '도서권수'가 0보다 큰 데이터프레임 생성
ns_book7 = ns_book6[ns_book6['도서권수']>0] # 376770


# In[6]:

# 백분위수 범위 조정
# percentiles=[...]
# 30%, 60%, 90%
ns_book7.describe(percentiles=[0.3, 0.6, 0.9])


# In[7]:

# include='object'
# 열의 데이터 타입이 수치가 아닌 데이터 타입의 열의 기술통계
ns_book7_object_describe = ns_book7.describe(include='object')
# count  : 누락값을 제외한 데이터 개수
# unique : 고유한 값의 개수
# top    : 가장 많이 등장하는 값
# freq   : top행에 등장하는 항목의 빈도수
    

#%%
# ## 평균

# $평균 = \dfrac{a + b + c}{3}$
# 
# $평균 = \dfrac{x_1 + x_2 + x_3}{3}$

# In[8]:


x = [10, 20, 30]
sum = 0
for i in range(3):
    sum += x[i]
print("평균:", sum / len(x))


# $평균 = \dfrac{x_1 + x_2 + x_3}{3} = \dfrac{\sum_{i=1}^{3} x_i}{3}$
# 
# $평균 대출건수 = \dfrac{\sum_{i=1}^{376770} x_i}{376770}$

# In[9]:

# '대출건수' 평균값
ns_book7['대출건수'].mean() # 11.593438968070707


# ## 중앙값

# In[10]:

# 중앙값
ns_book7['대출건수'].median()


# In[11]:

# 중앙값
temp_df = pd.DataFrame([1,2,3,4,5])
temp_df.median() # 3

#%%

# 중앙값
# (2 + 3) / 2 -> 2.5
temp_df = pd.DataFrame([1,2,3,4])
temp_df.median() # 2.5


# In[12]:

# 교재 : 172참조
# df.duplicated() 참조
# 중복값을 제거한 중앙값
ns_book7['대출건수'].drop_duplicates().median() # 183.0

#%%

ns_book7_unique = ns_book7['대출건수'].drop_duplicates()
ns_book7_unique.median() # 183.0


#%%
# ## 최솟값, 최댓값

# In[13]:


ns_book7['대출건수'].min() # 0 


# In[14]:


ns_book7['대출건수'].max() # 1765


#%%
# ## 분위수

# In[15]:


ns_book7['대출건수'].quantile(0.25) # 2.0


# In[16]:


ns_book7['대출건수'].quantile([0.25,0.5,0.75])
"""
0.25     2.0
0.50     6.0
0.75    14.0
Name: 대출건수, dtype: float64
"""

# In[17]:

# 보간(interpolation)    
# 두 지점 사이에 놓인 특정 위치의 값을 구하는 방법

# 90%에 해당하는 분위수
pd.Series([1,2,3,4,5,6,7,8,9,10]).quantile(0.9) # 9.1


#%%

pd.Series([1,2,3,4,5]).quantile(0.9) # 4.6
pd.Series([1,2,3,4,5]).quantile(0.9,interpolation='linear') # 4.6


# In[18]:

# A:a = B:b
# b = a * B / A
4 + (0.9-0.75)*(5-4)/(1.0-0.75) # 4.6


# In[19]:

# interpolation='midpoint'
# 분위수에 상관없이 무조건 두 수 사이의 중앙값을 사용
pd.Series([1,2,3,4,5]).quantile(0.9, interpolation='midpoint') # 4.5


# In[20]:

# interpolation='nearest'
# 두 수 중에서 가까운 값을 선택
# 4.6은 5에 더 가깝다.
pd.Series([1,2,3,4,5]).quantile(0.9, interpolation='nearest') # 5


# In[21]:

# 백분위 구하기
borrow_10_flag = ns_book7['대출건수'] < 10


# In[22]:


borrow_10_flag.mean() # 0.6402712530190833


# In[23]:

ns_book7['대출건수'].quantile(0.65) # 10.0

#%%
# ## 분산

# 편차: 평균을 뺀 값
# 편차제곱: 편차를 제곱
# 편차제곱합: 편차제곱의 합계
# 분산: 편차제곱합의 평균(편차제곱합 / n)
# 불편분산 : 편차제곱합 / (n - 1)
# 표준편차: 분산의 제곱근(루트)

#%%

# 분산(Variance): 평균으로부터 데이터가 얼마나 퍼져있는지 나타내는 통계량

#%%

# $ s^2 = \dfrac{\sum_{i=1}^{n}(x_i - \bar{x})^2}{n}$
# 
# $ \bar{x} = \dfrac{\sum_{i=1}^n x_i}{n}$

# In[24]:

# 판다스의 분산 계산 : var()
ns_book7['대출건수'].var() # 371.69563042906674


# ## 표준 편차

# $ s = \sqrt{\dfrac{\sum_{i=1}^{n}(x_i - \bar{x})^2}{n}}$

# In[25]:

# 표준편차 : std()
ns_book7['대출건수'].std() # 19.27


# $\sqrt{4}=2$

# In[26]:

# 넘파이를 이용
import numpy as np

# 편차 : 평균을 뺀 값
diff = ns_book7['대출건수'] - ns_book7['대출건수'].mean()

# 표준편차 = np.sqrt(불편분산)
# 표준편차 = 제곱근(편차제곱의 합 / (n-1))
np.sqrt( np.sum(diff**2) / (len(ns_book7)-1) )


# ## 최빈값

# In[27]:

# 최빈값(mode)
# 데이터에서 가장 많이 등장하는 값(빈도가 높은 값)
ns_book7['도서명'].mode() # 승정원일기


# In[28]:


ns_book7['발행년도'].mode() # 2012


#%%
# ## 데이터프레임에서 기술통계 구하기

# In[29]:
    
# 전체 데이터프레임의 평균
# ns_book7.mean()


#%%
# numeric_only=True : 수치형 열만 연산
ns_book7.mean(numeric_only=True)
"""
번호      202977.476649
발행년도      2008.460076
도서권수         1.145540
대출건수        11.593439
dtype: float64
"""

# In[30]:

# 최빈값
# 컬럼: 첫번째 컬럼('번호') 제외한 모든 컬럼
# 각 컬럼은 서로 관계가 없으며 독립적인 최빈값
ns_book7.loc[:, '도서명':].mode()


# In[31]:


ns_book7.to_csv('./data/ns_book7.csv', index=False)

#%%
# ## 넘파이의 기술통계 함수

# ### 평균 구하기

# In[32]:

# '대출건수' 평균값
ns_book7['대출건수'].mean() # 11.593438968070707

#%%
import numpy as np
np.mean(ns_book7['대출건수']) # 11.593438968070707


# $\dfrac{국어점수 \times 2 + 수학점수}{3}$
# 
# $\dfrac{국어점수 \times 국어가중치 + 수학점수 \times 수학가중치}{국어가중치 + 수학가중치}$ 
# 
# $가중평균 = \dfrac{x_1 \times w_1 + x_2 \times w_2}{w_1 + w_2} = \dfrac{\sum_{i=1}^{2} x_i \times w_i}{\sum_{i=1}^{2} w_i}$

# In[33]:

# 가중평균
# weights : weights = 1 / ns_book7['도서권수']
# '도서권수'의 역수로 
np.average(ns_book7['대출건수'], weights = 1 / ns_book7['도서권수']) # 10.543612175385386


# In[34]:


np.mean(ns_book7['대출건수']/ns_book7['도서권수']) # 9.873029861445774


# In[35]:

# 전체 대출건수 / 전체 도서권수
# 도서에 상관없이 한 권 당 대출건수를 구함
ns_book7['대출건수'].sum()/ns_book7['도서권수'].sum()  # 10.120503701300958


# ### 중앙값 구하기

# In[36]:


np.median(ns_book7['대출건수']) #  6.0


# ### 최솟값, 최댓값 구하기

# In[37]:


# 최솟값
np.min(ns_book7['대출건수']) # 0
 

# In[38]:

# 최댓값
np.max(ns_book7['대출건수']) # 1765


# ### 분위수 구하기

# In[39]:


# interpolation 매개변수가 numpy 1.22(python >= 3.8) 버전부터 method로 바뀜
# Definition : quantile(a, q, axis=None, out=None, overwrite_input=False, 
#                      method='linear', keepdims=False, *, interpolation=None)
# interpolationstr, optional
#      Deprecated name for the method keyword argument.
#      Deprecated since version 1.22.0.

np.quantile(ns_book7['대출건수'], [0.25,0.5,0.75])
# array([ 2.,  6., 14.])

# ### 분산 구하기
# 넘파이(표본분산) : n
# 판다스(불편분산) : n - 1
# 넘파이 : ddof(0, 1)

# In[40]:


# 넘파이 기본 : n
np.var(ns_book7['대출건수']) # 371.6946438971496


# $ s^2 = \dfrac{\sum_{i=1}^{n}(x_i - \bar{x})^2}{n - 1}$

# In[41]:

# 판다스 기본: n - 1
ns_book7['대출건수'].var() # 371.69563042906674

#%%

# 판다스 : 넘파이 방식, n
ns_book7['대출건수'].var(ddof=0) # 371.6946438971496


# In[42]:

# 넘파이 : 판다스 방식, n - 1
np.var(ns_book7['대출건수'], ddof=1) # 371.69563042906674


#%%
# ### 표준 편차 구하기

# 판다스
ns_book7['대출건수'].std() # 19.279409493785508

#%%
# 넘파이
np.std(ns_book7['대출건수']) #  19.27938390865096


#%%
# ### 최빈값 구하기

# In[44]:

# unique() 고유한 값을 추출
# return_counts=True: 등장 횟수
values, counts = np.unique(ns_book7['도서명'], return_counts=True)
max_idx = np.argmax(counts)
values[max_idx] # '승정원일기'


#%%
# ## 확인문제

# 문제4. 
# 241쪽에서 만든 ns_book7 데이터프레임에서 평균 대출건수가 
# 가강 높은 10개의 출판사를 추출하는 명령을 완성하라.
# 판다스의 groupby() 메서드를 사용하라.
# 

# 내림차순: ascending=False
ns_book7[['출판사','대출건수']].groupby('출판사').mean().sort_values('대출건수', ascending=False).head(10)

#%%
ns_book7_t20A = ns_book7[['출판사','대출건수']].groupby('출판사').mean()
ns_book7_t20 = ns_book7[['출판사','대출건수']].groupby('출판사').mean().sort_values('대출건수', ascending=False).head(20)

#%%

# 문제 5. 다음은 25%와 75% 경계에 해당하는 대출건수를 찾아 이 범위에 속한 도서가
# 전체 도서 중 몇 퍼센트를 차지하는지 구하는 명령이다.
# 빈칸을 채워 명령을 완성하라
# ns_book7 데이터프레임의 '대출건수'열에서 quantile() 메서드를 호출한다.

target_range = np.array(ns_book7['대출건수'].quantile(q=[0.25,0.75])) # 2, 14
target_bool_idx = (ns_book7['대출건수'] >= target_range[0]) & (ns_book7['대출건수'] <= target_range[1])
target_bool_idx.sum()/len(ns_book7)*100

# 결과: 51.51737134060568