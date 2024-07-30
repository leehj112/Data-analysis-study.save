# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:09:26 2024

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

gdown.download('https://bit.ly/3q9SZix', '20s_best_book.json', quiet=False)


# In[ ]:


import pandas as pd

books_df = pd.read_json('20s_best_book.json')
books_df.head()


# In[ ]:


books = books_df[['no','ranking','bookname','authors','publisher',
                 'publication_year','isbn13']]
books.head()


# In[ ]:


books_df.loc[[0,1], ['bookname','authors']]


# In[ ]:


books_df.loc[0:1, 'bookname':'authors']


# In[ ]:


books = books_df.loc[:, 'no':'isbn13']
books.head()


# In[ ]:


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


# In[ ]:


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


print(page_td.split()[0])


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


# In[ ]:


top10_books = books.head(10)


# In[ ]:


def get_page_cnt2(row):
    isbn = row['isbn13']
    return get_page_cnt(isbn)


# In[ ]:


page_count = top10_books.apply(get_page_cnt2, axis=1)
print(page_count)


# In[ ]:


page_count = top10_books.apply(lambda row: get_page_cnt(row['isbn13']), axis=1)


# In[ ]:


page_count.name = 'page_count'
print(page_count)


# In[ ]:


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


pd.merge(df1, df2, on='col1')


# In[ ]:


pd.merge(df1, df2, how='left', on='col1')


# In[ ]:


pd.merge(df1, df2, how='right', on='col1')


# In[ ]:


pd.merge(df1, df2, how='outer', on='col1')


# In[ ]:


pd.merge(df1, df2, left_on='col1', right_on='col1')


# In[ ]:


pd.merge(df1, df2, left_on='col2', right_index=True)
