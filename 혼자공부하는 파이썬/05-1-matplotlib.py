# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:11:11 2024

@author: leehj
"""

#!/usr/bin/env python
# coding: utf-8

# # 05-1 맷플롯립 기본 요소 알아보기

# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://nbviewer.jupyter.org/github/rickiepark/hg-da/blob/main/05-1.ipynb"><img src="https://jupyter.org/assets/share.png" width="61" />주피터 노트북 뷰어로 보기</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/rickiepark/hg-da/blob/main/05-1.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />구글 코랩(Colab)에서 실행하기</a>
#   </td>
# </table>

# ## `Figure` 클래스

# In[1]:


import gdown
gdown.download('https://bit.ly/3pK7iuu', './data/ns_book7.csv', quiet=False)


# In[2]:


import pandas as pd

ns_book7 = pd.read_csv('./data/ns_book7.csv', low_memory=False)
ns_book7.head() # 376770


# In[3]:


import matplotlib.pyplot as plt

# 산점도
plt.scatter(ns_book7['도서권수'], ns_book7['대출건수'], alpha=0.1)
plt.show()


# In[4]:

# 기본 그래프의 크기(인치) : 너비, 높이
# 인치(1인치) : 2.54cm
print(plt.rcParams['figure.figsize']) 
# [6.0, 4.0] : 63.5 pixel, 62 pixcel


# In[5]:

# figsize=(9, 6)
# 너비 : 9 inch, 61 pixel
# 높이 : 6 inch, 60 pixel
plt.figure(figsize=(9, 6))
plt.scatter(ns_book7['도서권수'], ns_book7['대출건수'], alpha=0.1)
plt.show()


# In[6]:

print(plt.rcParams['figure.dpi'])
# DPI(Dot Per Inch) : 인쇄물의 해상도
# 27인치(1920 Pixel) : 72.0
# PPI(Pixel Per Inch) : 화면 해상도


# In[7]:

dpi = plt.rcParams['figure.dpi']
print("DPI:", dpi)

#%%

# plt.figure(figsize=(900/72, 600/72))
plt.figure(figsize=(900/dpi, 600/dpi))
plt.scatter(ns_book7['도서권수'], ns_book7['대출건수'], alpha=0.1)
plt.show()


# In[8]:


# get_ipython().run_line_magic('config', "InlineBackend.print_figure_kwargs = {'bbox_inches': None}")
plt.figure(figsize=(900/72, 600/72))
plt.scatter(ns_book7['도서권수'], ns_book7['대출건수'], alpha=0.1)
plt.show()


# In[9]:


# get_ipython().run_line_magic('config', "InlineBackend.print_figure_kwargs = {'bbox_inches': 'tight'}")


# In[10]:


plt.figure(dpi=144)
plt.scatter(ns_book7['도서권수'], ns_book7['대출건수'], alpha=0.1)
plt.show()


# ## `rcParams` 객체

# In[11]:

# DPI 기본값 변경
plt.rcParams['figure.dpi'] = 100


# In[12]:


plt.rcParams['scatter.marker']


# In[13]:


plt.rcParams['scatter.marker'] = '*'


# In[14]:


plt.scatter(ns_book7['도서권수'], ns_book7['대출건수'], alpha=0.1)
plt.show()


# In[15]:


plt.scatter(ns_book7['도서권수'], ns_book7['대출건수'], alpha=0.1, marker='+')
plt.show()


#%%
# ## 여러 개의 서브플롯 출력하기
# subplots(nrows: 'int'=1, ncols: 'int'=1)

# In[16]:


# 2개의 서브 플롯 지정
# 2행 1열
# nrows: 2
# ncols: 1
fig, axs = plt.subplots(2) 

# 1번째
axs[0].scatter(ns_book7['도서권수'], ns_book7['대출건수'], alpha=0.1)

# 2번째
axs[1].hist(ns_book7['대출건수'], bins=100)
axs[1].set_yscale('log')

fig.show()


# In[17]:

# 2행 1열
# nrows: 2
# ncols: 1
fig, axs = plt.subplots(2, figsize=(6, 8))

axs[0].scatter(ns_book7['도서권수'], ns_book7['대출건수'], alpha=0.1)
axs[0].set_title('scatter plot')

axs[1].hist(ns_book7['대출건수'], bins=100)
axs[1].set_title('histogram')
axs[1].set_yscale('log')

fig.show()


# In[18]:

# subplots(행, 열)
# 1행 2열
# nrows: 1
# ncols: 2
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

axs[0].scatter(ns_book7['도서권수'], ns_book7['대출건수'], alpha=0.1)
axs[0].set_title('scatter plot')
axs[0].set_xlabel('number of books')
axs[0].set_ylabel('borrow count')

axs[1].hist(ns_book7['대출건수'], bins=100)
axs[1].set_title('histogram')
axs[1].set_yscale('log')
axs[1].set_xlabel('borrow count')
axs[1].set_ylabel('frequency')

fig.show()
