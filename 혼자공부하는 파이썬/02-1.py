# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:08:54 2024

@author: leehj
"""

#!/usr/bin/env python
# coding: utf-8

# # 02-1 API 사용하기

# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://nbviewer.jupyter.org/github/rickiepark/hg-da/blob/main/02-1.ipynb"><img src="https://jupyter.org/assets/share.png" width="61" />주피터 노트북 뷰어로 보기</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/rickiepark/hg-da/blob/main/02-1.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />구글 코랩(Colab)에서 실행하기</a>
#   </td>
# </table>

# ## 파이썬에서 JSON 데이터 다루기

# In[1]:


# dict
d = {"name": "혼자 공부하는 데이터 분석"}

print(d['name'])


# In[2]:

# 파이썬 객체를 json 문자열로 변환
# json.dumps()

import json


# In[3]:

# dict -> json
d_str = json.dumps(d)
"""    
{"name": "\ud63c\uc790 \uacf5\ubd80\ud558\ub294 \ub370\uc774\ud130 \ubd84\uc11d"}
"""
#%%

# ensure_ascii=False
# 아스키문자 외에 다른 문자는 16진수로 출력되는 막음
d_str = json.dumps(d, ensure_ascii=False)
print(d_str)
"""
{"name": "혼자 공부하는 데이터 분석"}
"""

# In[4]:

print(type(d_str)) # <class 'str'>


# In[5]:

# json 문자열을 파이썬 객체로 변환
# json -> dict
d2 = json.loads(d_str)

print(d2['name'])


# In[6]:


print(type(d2)) # <class 'dict'>


# In[7]:


d3 = json.loads('{"name": "혼자 공부하는 데이터 분석", "author": "박해선", "year": 2022}')

print(d3['name'])
print(d3['author'])
print(d3['year'])


# In[8]:

# 문자열이 길어서 다음 라인으로 연결: 역슬래시(\)
d3 = json.loads('{"name": "혼자 공부하는 데이터 분석", \
                  "author": ["박해선","홍길동"], \
                  "year": 2022}')

print(type(d3['author'])) # <class 'list'>

print(d3['author'][0]) # 박해선
print(d3['author'][1]) # 홍길동


# In[9]:

# 멀티라인 문자열 지정
# json  : 배열[객체, 객체]
# python: 리스트[딕셔너리, 딕셔너리]
d4_str = """
[
  {"name": "혼자 공부하는 데이터 분석", "author": "박해선", "year": 2022},
  {"name": "혼자 공부하는 머신러닝+딥러닝", "author": "박해선", "year": 2020}
]
"""
d4 = json.loads(d4_str)

print(type(d4)) # <class 'list'>
print(d4[0]['name']) # 혼자 공부하는 데이터 분석
print(d4[1]['name']) # 혼자 공부하는 머신러닝+딥러닝


# In[10]:


import pandas as pd

pd_d4 = pd.read_json(d4_str)
print(pd_d4)


# In[11]:


pd_d4_11 = pd.DataFrame(d4)


# ## 파이썬에서 XML 다루기

# In[12]:


x_str = """
<book>
    <name>혼자 공부하는 데이터 분석</name>
    <author>박해선</author>
    <year>2022</year>
</book>
"""


# In[13]:


import xml.etree.ElementTree as et

book = et.fromstring(x_str)


# In[14]:


print(type(book))


# In[15]:


print(book.tag)


# In[16]:


book_childs = list(book)

print(book_childs)


# In[17]:

    
# etree.ElementTree.Element
name, author, year = book_childs

print(name.text)
print(author.text)
print(year.text)


# In[18]:


name = book.findtext('name')
author = book.findtext('author')
year = book.findtext('year')

print(name)
print(author)
print(year)


# In[19]:


x2_str = """
<books>
    <book>
        <name>혼자 공부하는 데이터 분석</name>
        <author>박해선</author>
        <year>2022</year>
    </book>
    <book>
        <name>혼자 공부하는 머신러닝+딥러닝</name>
        <author>박해선</author>
        <year>2020</year>
    </book>
</books>
"""


# In[20]:


books = et.fromstring(x2_str)

print(books.tag)


# In[21]:


for book in books.findall('book'):
    name = book.findtext('name')
    author = book.findtext('author')
    year = book.findtext('year')
    
    print(name)
    print(author)
    print(year)
    print()


# In[22]:

# 판다스에서 XML 형식 처리    

pd_xml = pd.read_xml(x2_str)


# ## 파이썬으로 API 호출하기

# In[23]:


import requests


# In[24]:


# 인증키를 발급받아 문자열 맨 끝에 추가해 주세요.
url = "http://data4library.kr/api/loanItemSrch?format=json&startDt=2021-04-01&endDt=2021-04-30&age=20&authKey=c01ec15e4574f74ee45cba2601bad15b82971e606e3b0740977ee4b363ce2fe2"


# In[25]:


r = requests.get(url)


# In[26]:


data = r.json()

print(data)


# In[27]:


data


# In[28]:


data['response']['docs']


# In[29]:


books = []
for d in data['response']['docs']:
    books.append(d['doc'])


# In[30]:


books = [d['doc'] for d in data['response']['docs']]


# In[31]:


books


# In[32]:


books_df = pd.DataFrame(books)

books_df


# In[33]:


books_df.to_json('20s_best_book.json')
