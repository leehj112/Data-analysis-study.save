# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:09:05 2024

@author: leehj
"""

#!/usr/bin/env python
# coding: utf-8

# # 02-2 스크래핑 사용하기

# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://nbviewer.jupyter.org/github/rickiepark/hg-da/blob/main/02-2.ipynb"><img src="https://jupyter.org/assets/share.png" width="61" />주피터 노트북 뷰어로 보기</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/rickiepark/hg-da/blob/main/02-2.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />구글 코랩(Colab)에서 실행하기</a>
#   </td>
# </table>

# ## 검색 결과 페이지 가져오기

# In[ ]:


import gdown

gdown.download('https://bit.ly/3q9SZix', './data/20s_best_book.json', quiet=False)


# In[ ]:


import pandas as pd

books_df = pd.read_json('./data/20s_best_book.json')
books_df.head()


# In[ ]:

# 컬럼 선택
books = books_df[['no','ranking','bookname','authors','publisher', 'publication_year','isbn13']]
books.head()


# In[ ]:

# 인덱스: 0, 1 행
# 컬럼: ['bookname','authors']
books_df.loc[[0,1], ['bookname','authors']]


# In[ ]:

# 인덱스: 0, 1 행
# 컬럼: 'bookname': 'authors' 범위
books_df.loc[0:1, 'bookname':'authors']


# In[ ]:

# 인덱스: 전체 행
# 컬럼: 'no'부터 'isbn13'까지
books = books_df.loc[:, 'no':'isbn13']
books.head()


# In[ ]:

# 인덱스: 0부터 끝까지 2스텝
# 컬럼: 'no'부터 'isbn13'까지
# 처음부터 5개의 행을 선택
books_df.loc[::2, 'no':'isbn13'].head()


# In[ ]:


# DH_KEY_TOO_SMALL 에러가 발생하는 경우 다음 코드의 주석을 제거하고 실행하세요.
# https://stackoverflow.com/questions/38015537/python-requests-exceptions-sslerror-dh-key-too-small
# import requests

# requests.packages.urllib3.util.ssl_.DEFAULT_CIPHERS += 'HIGH:!DH:!aNULL'
# try:
#     requests.packages.urllib3.contrib.pyopenssl.DEFAULT_SSL_CIPHER_LIST += 'HIGH:!DH:!aNULL'
# except AttributeError:
#     # no pyopenssl support used / needed / available
#     pass


# In[ ]:


import requests

isbn = 9791190090018      # '우리가 빛의 속도로 갈 수 없다면'의 ISBN
url = 'http://www.yes24.com/Product/Search?domain=BOOK&query={}'

r = requests.get(url.format(isbn))
# r : models.Response

# http://www.yes24.com/Product/Search?domain=BOOK&query=9791190090018

# In[ ]:

# HTML
print(r.text)


# ## 뷰티플수프

# In[ ]:


from bs4 import BeautifulSoup


# In[ ]:


soup = BeautifulSoup(r.text, 'html.parser')


# In[ ]:


prd_link = soup.find('a', attrs={'class':'gd_name'})


# In[ ]:


print(prd_link)


# In[ ]:


print(prd_link['href'])

#%%

print(prd_link.text)


# In[ ]:


# '우리가 빛의 속도로 갈 수 없다면'의 상세 페이지 가져오기
url = 'http://www.yes24.com'+prd_link['href']
r = requests.get(url)


# In[ ]:


print(r.text)


# In[ ]:


soup = BeautifulSoup(r.text, 'html.parser')
prd_detail = soup.find('div', attrs={'id':'infoset_specific'})
print(prd_detail)


# In[ ]:


prd_tr_list = prd_detail.find_all('tr')
print(prd_tr_list)


# In[ ]:


for tr in prd_tr_list:
    if tr.find('th').get_text() == '쪽수, 무게, 크기':
        page_td = tr.find('td').get_text()
        break


# In[ ]:


print(page_td)


# In[ ]:

# 공백 분할
print(page_td.split())    # ['330쪽', '|', '496g', '|', '130*198*30mm']
print(page_td.split()[0]) # 330쪽


# ## 전체 도서의 쪽수 구하기

# In[ ]:


def get_page_cnt(isbn):
    # Yes24 도서 검색 페이지 URL
    url = 'http://www.yes24.com/Product/Search?domain=BOOK&query={}'
    # URL에 ISBN을 넣어 HTML 가져옵니다.
    r = requests.get(url.format(isbn))
    soup = BeautifulSoup(r.text, 'html.parser')   # HTML 파싱
    # 검색 결과에서 해당 도서를 선택합니다.
    prd_info = soup.find('a', attrs={'class':'gd_name'})
    if prd_info == None:
        return ''
    # 도서 상세 페이지를 가져옵니다.
    url = 'http://www.yes24.com'+prd_info['href']
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    # 상품 상세정보 div를 선택합니다.
    prd_detail = soup.find('div', attrs={'id':'infoset_specific'})
    # 테이블에 있는 tr 태그를 가져옵니다.
    prd_tr_list = prd_detail.find_all('tr')
    # 쪽수가 들어 있는 th를 찾아 td에 담긴 값을 반환합니다.
    for tr in prd_tr_list:
        if tr.find('th').get_text() == '쪽수, 무게, 크기':
            return tr.find('td').get_text().split()[0]
    return ''


# In[ ]:


get_page_cnt(9791190090018)

#%%
get_page_cnt(9791165300005)


#%%
# In[ ]:


top10_books = books.head(10)


# In[ ]:


def get_page_cnt2(row):
    isbn = row['isbn13']
    return get_page_cnt(isbn)


# In[ ]:

# 리턴 : Series, 요소 10개
page_count = top10_books.apply(get_page_cnt2, axis=1)
print(page_count)


# In[ ]:


page_count2 = top10_books.apply(lambda row: get_page_cnt(row['isbn13']), axis=1)


# In[ ]:


page_count.name = 'page_count'
print(page_count)


# In[ ]:

# 스크래핑한 책정보에 페이지 수 병합(merge)
top10_with_page_count = pd.merge(top10_books, page_count,
                                 left_index=True, right_index=True)
top10_with_page_count


# ## `merge()` 함수의 매개변수

# In[ ]:


df1 = pd.DataFrame({'col1': ['a','b','c'], 'col2': [1,2,3]})
df1


# In[ ]:


df2 = pd.DataFrame({'col1': ['a','b','d'], 'col3': [10,20,30]})
df2


# In[ ]:

# 두 데이터프레임에 모두 존재하는 요소만 선택
# how('inner') 기본값으로 처리
pd.merge(df1, df2, on='col1')
"""
  col1  col2  col3
0    a     1    10
1    b     2    20
"""

#%%

# how: 'inner', 기본값
pd.merge(df1, df2, how='inner', on='col1')
"""
  col1  col2  col3
0    a     1    10
1    b     2    20
"""

# In[ ]:

# how: 'inner', 기본값
# how: 'left', 왼쪽에 지정된 데이터프레임 기준
pd.merge(df1, df2, how='left', on='col1')
"""
   df1   df1   df2
  col1  col2  col3
0    a     1  10.0
1    b     2  20.0
2    c     3   NaN
"""

# In[ ]:


pd.merge(df1, df2, how='right', on='col1')
"""
#  df2   df1   df2
  col1  col2  col3
0    a   1.0    10
1    b   2.0    20
2    d   NaN    30
"""


# In[ ]:


pd.merge(df1, df2, how='outer', on='col1')
"""
   df1   df1   
   df2         df2
  col1  col2  col3
0    a   1.0  10.0
1    b   2.0  20.0
2    c   3.0   NaN
3    d   NaN  30.0
"""

# In[ ]:

# 병합할 열의 이름이 서로 다른 경우
# 
pd.merge(df1, df2, left_on='col1', right_on='col1')
"""
  col1  col2  col3
0    a     1    10
1    b     2    20
"""

# In[ ]:


pd.merge(df1, df2, left_on='col2', right_index=True)
"""
  col1_x  col2 col1_y  col3
0      a     1      b    20
1      b     2      d    30
"""