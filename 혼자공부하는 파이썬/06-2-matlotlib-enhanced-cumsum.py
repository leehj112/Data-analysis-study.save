# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:12:02 2024

@author: leehj
"""

#!/usr/bin/env python
# coding: utf-8

# # 06-2 맷플롯립의 고급 기능 배우기

# ## 실습 준비하기

# 이 노트북은 맷플롯립 그래프에 한글을 쓰기 위해 나눔 폰트를 사용합니다. 코랩의 경우 다음 셀에서 나눔 폰트를 직접 설치합니다.

# ### 데이터값 누적하여 그리기

# In[1]:


# 노트북이 코랩에서 실행 중인지 체크합니다.
"""
import sys
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

#%%
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# font_files = fm.findSystemFonts(fontpaths=['C:/Windows/Fonts'])
font_files = fm.findSystemFonts(fontpaths=['C:/Users/Solero/AppData/Local/Microsoft/Windows/Fonts'])
for fpath in font_files:
    print(fpath)
    fm.fontManager.addfont(fpath)

# plt.rcParams['font.family'] = 'NanumGothic'
# plt.rcParams['font.family'] = 'NanumSquare'    
plt.rcParams['font.family'] = 'NanumBarunGothic'    

plt.plot([1, 4, 9, 16])
plt.title('간단한 선 그래프')
plt.show()

# In[2]:


import matplotlib.pyplot as plt

# 나눔바른고딕 폰트로 설정합니다.
plt.rc('font', family='NanumBarunGothic')

# 그래프 DPI 기본값을 변경합니다.
plt.rcParams['figure.dpi'] = 100


# In[3]:


import gdown

gdown.download('https://bit.ly/3pK7iuu', './data/ns_book7.csv', quiet=False)


# In[4]:


import pandas as pd

ns_book7 = pd.read_csv('./data/ns_book7.csv', low_memory=False)
ns_book7.head()


#%%
# ## 하나의 피겨에 여러 개의 선 그래프 그리기

# In[5]:

pubs_counts = ns_book7['출판사'].value_counts()

#%%    

# 출판사별 상위 30개
top30_pubs = ns_book7['출판사'].value_counts()[:30]

#%%

# 출판사별 상위 30개에 해당하는 데이터는 True
# 그렇지 않으면 False
top30_pubs_idx = ns_book7['출판사'].isin(top30_pubs.index)


# In[6]:

# 행: 상위 30개 출판사    
# 열: ['출판사', '발행년도', '대출건수']
ns_book9 = ns_book7[top30_pubs_idx][['출판사', '발행년도', '대출건수']]

#%%

# 집계 : '출판사', '발행년도'별 총 대출건수
ns_book9 = ns_book9.groupby(by=['출판사', '발행년도']).sum()


# In[7]:

# 인덱스 번호를 재배열
ns_book9 = ns_book9.reset_index()
ns_book9[ns_book9['출판사'] == '황금가지'].head()


# In[8]:


line1 = ns_book9[ns_book9['출판사'] == '황금가지']
line2 = ns_book9[ns_book9['출판사'] == '비룡소']


# In[9]:


#%%

# 피벗 테이블(pivot table)

# In[12]:

# 인덱스 : 출판사
# 피벗 : '발행년도'가 컬럼으로 이동
# 대출건수에 대한 발행년도별로 컬럼이 생성
ns_book10 = ns_book9.pivot_table(index='출판사', columns='발행년도')
ns_book10.head()


# In[13]:

ns_book10.columns[:10]

#%%
"""
MultiIndex([('대출건수', 1947),
            ('대출건수', 1974),
            ('대출건수', 1975),
            ('대출건수', 1976),
            ('대출건수', 1977),
            ('대출건수', 1978),
            ('대출건수', 1979),
            ('대출건수', 1980),
            ('대출건수', 1981),
            ('대출건수', 1982)],
           names=[None, '발행년도'])
"""

# In[14]:

# 상위 10개의 인덱스(출판사) 추출    
top10_pubs = top30_pubs.index[:10]

#%%

# 다단으로 구성된 열에서 컬럼 추출
year_cols0 = ns_book10.columns.get_level_values(0) # None
year_cols = ns_book10.columns.get_level_values(1)  # 발행년도


# In[15]:

# ns_book10 : 상위 10개 출판사 기준 발행년도(피벗)별 대출건수
# year_cols : 발행년도
# top10_pubs: 상위 10개 출판사
# 상위 10개 출판사 순위: 문학동네, 믿음사, 김영사, ... 학지사, 한울

print(top10_pubs)
# ['문학동네', '민음사', '김영사', '웅진씽크빅', '시공사', 
# '창비', '문학과지성사', '위즈덤하우스', '학지사', '한울']

#%%

#%%
# ## 하나의 피겨에 여러 개의 막대 그래프 그리기

# In[16]:



#%%
# ### 데이터값 누적하여 그리기

# In[20]:

# ns_book10 : 상위 30개 출판사의 발행년도별 피벗 테이블 형태    
# top10_pubs : 상위 10개 출판사
# 행 : 상위 5개
# 열 : 2013~2020
top10_2013_2020 = ns_book10.loc[top10_pubs[:5], ('대출건수',2013):('대출건수',2020)]


# In[21]:

# cumsum()은 기본적으로 행을 따라 값을 누적한다.
# axis 매개변수를 1로 지정하면 열 방향으로 누적한다.
top10_2013_2020_cumsum = ns_book10.loc[top10_pubs[:5], ('대출건수',2013):('대출건수',2020)].cumsum()


# In[22]:

# 데이터프레임 전체에 누적 : 
# 행단위: 상위 10개 출판사의 대출건수
# 상위출판사부터 하위 출판사까지 대출건수를 누적
# 가장 하위 출판사가 가장 큰 값을 가지게 되어
# 누적값이 큰 순서부터 그리게 되면
# 그래프를 그릴 때 덮어 쓰지 않고 보이도록 하기위한 처리
ns_book12 = ns_book10.loc[top10_pubs].cumsum()

# In[23]:

# 인덱스의 순으로 반복하면?
# 가장 큰 막대가 이전에 그린 막대를 모두 덮어 씀
ns_book12.index
#%%
"""
# Index(['문학동네', '민음사', '김영사', '웅진씽크빅', '시공사', '창비', 
         '문학과지성사', '위즈덤하우스', '학지사', '한울'],
      dtype='object')
"""
#%%

fig, ax = plt.subplots(figsize=(8, 6))

for i in range(len(ns_book12)):
    bar = ns_book12.iloc[i]     # 행 추출
    label = ns_book12.index[i]  # 출판사 이름 추출
    ax.bar(year_cols, bar, label=label)
    
ax.set_title('년도별 대출건수')
ax.legend(loc='upper left')
ax.set_xlim(1985, 2025)
fig.show()
    
    
#%%
# 
fig, ax = plt.subplots(figsize=(8, 6))

# 인덱스의 역순으로 반복
# 가장 하위 출판사부터 상위 출판사까지 역순으로 그림
for i in reversed(range(len(ns_book12))):
    bar = ns_book12.iloc[i]     # 행 추출
    label = ns_book12.index[i]  # 출판사 이름 추출
    ax.bar(year_cols, bar, label=label)
    print("출판사이름: ", label)
    
ax.set_title('년도별 대출건수')
ax.legend(loc='upper left')
ax.set_xlim(1985, 2025)
fig.show()


#%%
# ## 원 그래프 그리기

# In[24]:

# top30_pubs : 상위 출판사 30개 

# 10개 추출
data = top30_pubs[:10] 

# 10의 인덱스 : 출판사 이름
# labels = top30_pubs.index[:10]
labels = data.index
print(labels)
#%%
"""
['문학동네', '민음사', '김영사', '웅진씽크빅', '시공사', 
 '창비', '문학과지성사', '위즈덤하우스', '학지사', '한울']
"""

# In[25]:

# 원그래프 : pie
# 그려지는 순서: 3시 기준으로 반시계 방향
fig, ax = plt.subplots(figsize=(8, 6))
ax.pie(data, labels=labels)
ax.set_title('출판사 도서비율')
fig.show()


# In[26]:

# startangle : 90(12시 방향부터 시작)
plt.pie([10,9], labels=['A제품', 'B제품'], startangle=90)
plt.title('제품의 매출비율')
plt.show()


# In[27]:

# 비율을 표시하고 부채꼴 강조
# 시작: startangle : 90(12시 방향부터 시작)
# 비율: autopct, 소숫점 이하 1자리까지 표시, 뒤에 %를 붙여라
# 강조: explode([0.1]+[0]*9), '문학동네', 데이터의 갯수만큼 강조의 정도를 리스트로 전달
fig, ax = plt.subplots(figsize=(8, 6))
ax.pie(data, labels=labels, startangle=90, autopct='%.1f%%', explode=[0.1]+[0]*9)
ax.set_title('출판사 도서비율')
fig.show()

#%%

# explode : 첫 번째는 0.1, 나머지는 0인 파이썬 리스트
explode = [0.1] + [0] * 9
print(explode) # [0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

#%%

explode = [0.2, 0, 0, 0, 0, 0.1, 0, 0, 0, 0]
fig, ax = plt.subplots(figsize=(8, 6))
ax.pie(data, labels=labels, startangle=90, autopct='%.1f%%', explode=explode)
ax.set_title('출판사 도서비율')
fig.show()


#%%
# ## 여러 종류의 그래프가 있는 서브플롯 그리기

# In[28]:

# subplots(nrows=1, ncols=1)
# 2행 2열의 그래프
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# 산점도
ns_book8 = ns_book7[top30_pubs_idx].sample(1000, random_state=42)
sc = axes[0, 0].scatter(ns_book8['발행년도'], ns_book8['출판사'], 
                        linewidths=0.5, edgecolors='k', alpha=0.3,
                        s=ns_book8['대출건수'], c=ns_book8['대출건수'], cmap='jet')
axes[0, 0].set_title('출판사별 발행도서')
fig.colorbar(sc, ax=axes[0, 0])

# 스택 선 그래프
axes[0, 1].stackplot(year_cols, ns_book10.loc[top10_pubs].fillna(0), 
                     labels=top10_pubs)
axes[0, 1].set_title('년도별 대출건수')
axes[0, 1].legend(loc='upper left')
axes[0, 1].set_xlim(1985, 2025)

# 스택 막대 그래프
for i in reversed(range(len(ns_book12))):
    bar = ns_book12.iloc[i]     # 행 추출
    label = ns_book12.index[i]  # 출판사 이름 추출
    axes[1, 0].bar(year_cols, bar, label=label)
axes[1, 0].set_title('년도별 대출건수')
axes[1, 0].legend(loc='upper left')
axes[1, 0].set_xlim(1985, 2025)

# 원 그래프
axes[1, 1].pie(data, labels=labels, startangle=90,
               autopct='%.1f%%', explode=[0.1]+[0]*9)
axes[1, 1].set_title('출판사 도서비율')

fig.savefig('all_in_one.png')
fig.show()

#%%
# ## 판다스로 여러 개의 그래프 그리기

# ### 스택 영역 그래프 그리기

# In[29]:


ns_book11a = ns_book9.pivot_table(index='발행년도', columns='출판사', values='대출건수')
ns_book11a.loc[2000:2005]


# In[30]:


import numpy as np

ns_book11 = ns_book7[top30_pubs_idx].pivot_table(
    index='발행년도', columns='출판사', 
    values='대출건수', aggfunc=np.sum)
ns_book11.loc[2000:2005]


# In[31]:


fig, ax = plt.subplots(figsize=(8, 6))
ns_book11[top10_pubs].plot.area(ax=ax, title='년도별 대출건수',
                                xlim=(1985, 2025))
ax.legend(loc='upper left')
fig.show()


#%%

# ### 스택 막대 그래프 그리기

fig, ax = plt.subplots(figsize=(8, 6))
ns_book11.loc[1985:2025, top10_pubs].plot.bar(
    ax=ax, title='년도별 대출건수', stacked=True, width=0.8)
ax.legend(loc='upper left')
fig.show()
