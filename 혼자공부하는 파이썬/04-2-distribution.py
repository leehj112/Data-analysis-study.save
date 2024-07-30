# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:10:56 2024

@author: leehj
"""

#!/usr/bin/env python
# coding: utf-8

# # 04-2 분포 요약하기

# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://nbviewer.jupyter.org/github/rickiepark/hg-da/blob/main/04-2.ipynb"><img src="https://jupyter.org/assets/share.png" width="61" />주피터 노트북 뷰어로 보기</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/rickiepark/hg-da/blob/main/04-2.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />구글 코랩(Colab)에서 실행하기</a>
#   </td>
# </table>

# ## 산점도 그리기

# In[1]:


import gdown

gdown.download('https://bit.ly/3pK7iuu', './data/ns_book7.csv', quiet=False)


# In[2]:


import pandas as pd

ns_book7 = pd.read_csv('./data/ns_book7.csv', low_memory=False)
ns_book7.head()


# In[3]:

# 맷플롯립(matplotlib)
# pip install matplotlib

import matplotlib.pyplot as plt

#%%
###############################
# 산점도 그리기
# 데이터를 화면에 뿌리듯이 그리는 그래프

# plt.scatter(x, y)
plt.scatter([1,2,3,4], [1,2,3,4])
plt.show()


# In[4]:


plt.scatter(ns_book7['번호'], ns_book7['대출건수'])
plt.show()


# In[5]:


plt.scatter(ns_book7['도서권수'], ns_book7['대출건수'])
plt.show()


# In[6]:

# 투명도: alpha
# alpha : 0.0(투명) ~ 1.0(불투명)
plt.scatter(ns_book7['도서권수'], ns_book7['대출건수'], alpha=0.1)
plt.show()


# In[7]:

# 양의 상관관계 : x축이 증가함에 따라 y축도 증가    
# 도서권수 당 대출건수가 증가함에 따라 대출건수도 증가한다.

average_borrows = ns_book7['대출건수'] / ns_book7['도서권수']

plt.scatter(average_borrows, ns_book7['대출건수'], alpha=0.1)
plt.show()


#%%
# ## 히스토그램(histgram) 그리기
# 데이터의 분포
# 구간: 계급, bins 옵션
# 도수: 데이터의 갯수

# In[8]:

# 구간: bins=5, 5개 구간으로 나눔
# 2.6 = 13 / 5
plt.hist([0,3,5,6,7,7,9,13], bins=5)
plt.show()


# In[9]:

import numpy as np

# 구간값 확인
np.histogram_bin_edges([0,3,5,6,7,7,9,13], bins=5)
# array([ 0. ,  2.6,  5.2,  7.8, 10.4, 13. ])

# In[10]:

# 표준정규분포(Standard Normal Distributtion)    
# 넘파이 randn() 
# 표준정규분포를 따르는 랜덤한 실수를 생성
# 평균: 0
# 표준편차: 1
np.random.seed(42)
random_samples = np.random.randn(1000) # 1000개 난수


# In[11]:

# 평균, 표준편차
print(np.mean(random_samples), np.std(random_samples))
# 평균: 0.01933205582232549
# 표준편차: 0.9787262077473543

# In[12]:


plt.hist(random_samples)
plt.show()


#%%

# 남산도서관 대출 데이터

plt.hist(ns_book7['대출건수'])
plt.show()


# In[14]:

# 구간조정
# 로그 스케일(log scale)
# y축 조정: plt.yscale('log')
plt.hist(ns_book7['대출건수'])
plt.yscale('log')
plt.show()


# In[15]:

# y축 조정: log=True
plt.hist(ns_book7['대출건수'], log=True)
plt.show()

# In[16]:

# 대출건수 : 0이 가장 많다.
plt.hist(ns_book7['대출건수'], bins=100)
plt.yscale('log')
plt.show()


# In[17]:

# len() 함수를 apply() 함수로 지정
# x: '도서명'의 길이
# y: 빈도수
# bins: 구간을 100개로 분할
title_len = ns_book7['도서명'].apply(len)
plt.hist(title_len, bins=100)
plt.show()


# In[18]:

plt.hist(title_len, bins=100)
plt.xscale('log')
plt.show()


#%%
# ## 상자 수염 그림 그리기

# In[19]:


temp = ns_book7[['대출건수','도서권수']]


# In[20]:

# 1번: 대출건수
# 2번: 도서권수
plt.boxplot(temp)
plt.show()


# In[21]:

# 로그 스케일 적용
plt.boxplot(ns_book7[['대출건수','도서권수']])
plt.yscale('log')
plt.show()


# In[22]:

# vert=False: 수평 그리기
plt.boxplot(ns_book7[['대출건수','도서권수']], vert=False)
plt.xscale('log')
plt.show()


# In[23]:

# IQR(interquartile range)
#  - 제1사분위(25%)와 제3사분위(75%) 사이의 거리
#  - 박스 구간
#    
# 수염길이 조정: 
#   - 기보값: 1.5 
#   - whis=10
plt.boxplot(ns_book7[['대출건수','도서권수']], whis=10)
plt.yscale('log')
plt.show()


# In[24]:

# 수염을 백분율로 지정
# whis=(0,100) : 0%~100% 수염으로
plt.boxplot(ns_book7[['대출건수','도서권수']], whis=(0,100))
plt.yscale('log')
plt.show()


# ## 판다스의 그래프 함수

# ### 산점도 그리기

# In[25]:


ns_book7.plot.scatter('도서권수', '대출건수', alpha=0.1)
plt.show()


# ### 히스토그램 그리기

# In[26]:


ns_book7['도서명'].apply(len).plot.hist(bins=100)
plt.show()


# In[27]:


ns_book7['도서명'].apply(len).plot.hist(bins=100)
plt.show()


# ### 상자 수염 그림 그리기

# In[28]:


ns_book7[['대출건수','도서권수']].boxplot()
plt.yscale('log')
plt.show()


# ## 확인문제

# [문제4] ns_book7에서 1980년~2022년에 발행된 도서를 선택하여 히스토그램을 그려라.

selected_rows = (1980 <= ns_book7['발행년도']) & (ns_book7['발행년도'] <= 2022)
plt.hist(ns_book7.loc[selected_rows, '발행년도'])
plt.show()


# #### 5.

# In[30]:


plt.boxplot(ns_book7.loc[selected_rows, '발행년도'])
plt.show()
