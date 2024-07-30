# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:11:46 2024

@author: leehj
"""

#!/usr/bin/env python
# coding: utf-8

# # 06-1 객체지향 API로 그래프 꾸미기

# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://nbviewer.jupyter.org/github/rickiepark/hg-da/blob/main/06-1.ipynb"><img src="https://jupyter.org/assets/share.png" width="61" />주피터 노트북 뷰어로 보기</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/rickiepark/hg-da/blob/main/06-1.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />구글 코랩(Colab)에서 실행하기</a>
#   </td>
# </table>

# ## pyplot 방식과 객체지향 API 방식

# In[1]:


import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 100


# In[2]:

# figure 객체를 명시적으로 생성하지 않고 사용
# 디폴트 내부에 figure 객체를 가지고 있다.
plt.plot([1, 4, 9, 16])
plt.title('simple line graph')
plt.show()


# In[3]:

# 명시적으로 figure 객체를 얻어서 사용
fig, ax = plt.subplots()
ax.plot([1, 4, 9, 16])
ax.set_title('simple line graph')
fig.show()


# ## 그래프에 한글 출력하기

# 이 노트북은 맷플롯립 그래프에 한글을 쓰기 위해 나눔 폰트를 사용합니다. 코랩의 경우 다음 셀에서 나눔 폰트를 직접 설치합니다.

# In[4]:


# 노트북이 코랩에서 실행 중인지 체크합니다.
import sys
"""
if 'google.colab' in sys.modules:
    get_ipython().system("echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections")
    # 나눔 폰트를 설치합니다.
    get_ipython().system('sudo apt-get -qq -y install fonts-nanum')
    import matplotlib.font_manager as fm
    font_files = fm.findSystemFonts(fontpaths=['/usr/share/fonts/truetype/nanum'])
    for fpath in font_files:
        fm.fontManager.addfont(fpath)
"""


#%%
# 폰트설치
#   - 나눔고딕: NanumGothic.ttf
# 
# 사용자가 설치한 폰트 위치:
# C:/Users/Solero/AppData/Local/Microsoft/Windows/Fonts
#%%
import matplotlib.font_manager as fm
# font_files = fm.findSystemFonts(fontpaths=['C:/Windows/Fonts'])
font_files = fm.findSystemFonts(fontpaths=['C:/Users/Solero/AppData/Local/Microsoft/Windows/Fonts'])
for fpath in font_files:
    print(fpath)
    fm.fontManager.addfont(fpath)


# In[5]:


plt.rcParams['figure.dpi'] = 100


# In[6]:


plt.rcParams['font.family']


# In[7]:


# 나눔고딕 폰트를 사용합니다.
plt.rcParams['font.family'] = 'NanumGothic'

# In[8]:


# 위와 동일하지만 이번에는 나눔바른고딕 폰트로 설정합니다.
plt.rc('font', family='NanumBarunGothic')


# In[9]:


plt.rc('font', family='NanumBarunGothic', size=11)


# In[10]:


print(plt.rcParams['font.family'], plt.rcParams['font.size'])


# In[11]:

from matplotlib.font_manager import findSystemFonts
findSystemFonts()


# In[12]:

plt.rcParams['font.family'] = 'NanumGothic'
# plt.rcParams['font.family'] = 'NanumSquare'    
# plt.rcParams['font.family'] = 'NanumBarunGothic'    

plt.plot([1, 4, 9, 16])
plt.title('간단한 선 그래프')
plt.show()


# In[13]:


plt.rc('font', size=10)


# ## 출판사별 발행 도서 개수 산점도 그리기

# In[14]:


import gdown

gdown.download('https://bit.ly/3pK7iuu', './data/ns_book7.csv', quiet=False)


# In[15]:


import pandas as pd

ns_book7 = pd.read_csv('./data/ns_book7.csv', low_memory=False)
ns_book7.head()


# In[16]:

top_pubs = ns_book7['출판사'].value_counts()

#%%

# 상위 30개 출판사
top30_pubs = ns_book7['출판사'].value_counts()[:30]
top30_pubs


# In[17]:

# 상위 30ㄱ래에 해당하는 출판사는 True, 그렇지 않으면 False
top30_pubs_idx = ns_book7['출판사'].isin(top30_pubs.index)
top30_pubs_idx


# In[18]:


top30_pubs_idx.sum() # 51886


# In[19]:

# 무작위로 1000개 데이터 샘플링
# random_state=42 : seed 고정
ns_book8 = ns_book7[top30_pubs_idx].sample(1000, random_state=42)
ns_book8.head()


# In[20]:

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(ns_book8['발행년도'], ns_book8['출판사'])
ax.set_title('출판사별 발행도서')
fig.show()


# In[21]:

# 마커 사이즈 : 기본 6.0
plt.rcParams['lines.markersize'] # 6.0


# In[22]:

# 마커 사이즈 : s=ns_book8['대출건수']
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(ns_book8['발행년도'], ns_book8['출판사'], s=ns_book8['대출건수'])
ax.set_title('출판사별 발행도서')
fig.show()


# In[23]:

# 마커 꾸미기
# alpha : 투명도
# eggecolors: 마커의 테두리
# linewidths: 마커의 테두리 선의 두께
# c: 산점도의 색상
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(ns_book8['발행년도'], ns_book8['출판사'], 
           linewidths=0.5, edgecolors='k', alpha=0.3,
           s=ns_book8['대출건수']*2, c=ns_book8['대출건수'])
ax.set_title('출판사별 발행도서')
fig.show()


# In[24]:

# cmap : 컬러맵(color map)
#  'jet':
#  - 낮은 값일수록 짙은 파란색
#  - 높은 값일수록 점차 노란색
#
# fig.colorbar(sc) : 컬러막대
fig, ax = plt.subplots(figsize=(10, 8))
sc = ax.scatter(ns_book8['발행년도'], ns_book8['출판사'], 
                linewidths=0.5, edgecolors='k', alpha=0.3,
                s=ns_book8['대출건수']**1.3, c=ns_book8['대출건수'], cmap='jet')
ax.set_title('출판사별 발행도서')
fig.colorbar(sc)
fig.show()
