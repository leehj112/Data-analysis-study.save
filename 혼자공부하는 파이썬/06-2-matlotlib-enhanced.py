# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:12:19 2024

@author: leehj
"""

#!/usr/bin/env python
# coding: utf-8

# # 06-2 맷플롯립의 고급 기능 배우기

# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://nbviewer.jupyter.org/github/rickiepark/hg-da/blob/main/06-2.ipynb"><img src="https://jupyter.org/assets/share.png" width="61" />주피터 노트북 뷰어로 보기</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/rickiepark/hg-da/blob/main/06-2.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />구글 코랩(Colab)에서 실행하기</a>
#   </td>
# </table>

# ## 실습 준비하기

# 이 노트북은 맷플롯립 그래프에 한글을 쓰기 위해 나눔 폰트를 사용합니다. 코랩의 경우 다음 셀에서 나눔 폰트를 직접 설치합니다.

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

# 황금가지: 파란색
# 비룡소: 주황색
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(line1['발행년도'], line1['대출건수'])
ax.plot(line2['발행년도'], line2['대출건수'])
ax.set_title('년도별 대출건수')
fig.show()


# In[10]:


fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(line1['발행년도'], line1['대출건수'], label='황금가지')
ax.plot(line2['발행년도'], line2['대출건수'], label='비룡소')
ax.set_title('년도별 대출건수')
ax.legend() # 범례
fig.show()


# In[11]:

# 선 그래프 5개 그리기
fig, ax = plt.subplots(figsize=(8, 6))
for pub in top30_pubs.index[:5]:
    line = ns_book9[ns_book9['출판사'] == pub]
    ax.plot(line['발행년도'], line['대출건수'], label=pub)
ax.set_title('년도별 대출건수')
ax.legend()
ax.set_xlim(1985, 2025)
fig.show()

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

# 스택 영역 그래프(stacked area graph)
fig, ax = plt.subplots(figsize=(8, 6))
ax.stackplot(year_cols, ns_book10.loc[top10_pubs].fillna(0), labels=top10_pubs)
ax.set_title('년도별 대출건수')
ax.legend(loc='upper left')
ax.set_xlim(1985, 2025)
fig.show()


#%%
# ## 하나의 피겨에 여러 개의 막대 그래프 그리기

# In[16]:

# line1 : 황금가지
# line2 : 비룡소
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(line1['발행년도'], line1['대출건수'], label='황금가지')
ax.bar(line2['발행년도'], line2['대출건수'], label='비룡소')
ax.set_title('년도별 대출건수')
ax.legend()
fig.show()


# In[17]:

# 위치이동: 발행년도, 황금가지(-0.2), 비룡소(+0.2)
# width : 막대의 너비, 0.4
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(line1['발행년도']-0.2, line1['대출건수'], width=0.4, label='황금가지')
ax.bar(line2['발행년도']+0.2, line2['대출건수'], width=0.4, label='비룡소')
ax.set_title('년도별 대출건수')
ax.legend()
fig.show()


# In[18]:

# 스택 막대 그래프
height1 = [5, 4, 7, 9, 8]
height2 = [3, 2, 4, 1, 2]

# x : range(5), 0~4
# y : height1, height2
plt.bar(range(5), height1, width=0.5)

# bottom : height1이 끝나는 위치에서 시작
plt.bar(range(5), height2, bottom=height1, width=0.5)
plt.show()

#%%

# 스택 막대 그래프
# 먼저 그린 그래프 위에 덮어 그려서 이전 그래프가 표현되지 않을 수 있다.
height1 = [5, 4, 7, 9, 8]
height2 = [3, 2, 4, 1, 10]
plt.bar(range(5), height1, width=0.5)
plt.bar(range(5), height2, width=0.5)
plt.show()

# In[19]:

# 막대의 길이를 누적해서 그림
# bottom 옵션을 사용하지 않고 처리
height1 = [5, 4, 7, 9, 8]
height2 = [3, 2, 4, 1, 2]

height3 = [a + b for a, b in zip(height1, height2)]

plt.bar(range(5), height3, width=0.5)
plt.bar(range(5), height1, width=0.5)
plt.show()
