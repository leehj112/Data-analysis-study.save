# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:08:23 2024

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

#%%
# ## 파이썬에서 XML 다루기

# In[12]:

# XML 형식의 문자열
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


print(type(book)) # <class 'xml.etree.ElementTree.Element'>


# In[15]:

# 루트
print(book.tag)


# In[16]:

# book 태그의 자식 엘리먼트(요소)를 리스트 객체로 변환
book_childs = list(book)

print(book_childs)


# In[17]:

    
# etree.ElementTree.Element
name, author, year = book_childs

print(name.text)   # 혼자 공부하는 데이터 분석
print(author.text) # 박해선
print(year.text)   # 2022


# In[18]:

# findtext() : 자식 엘리먼트 찾기
name = book.findtext('name')
author = book.findtext('author')
year = book.findtext('year')

# 존재하지 않는 자식
age = book.findtext('age') 

print(name)   # 혼자 공부하는 데이터 분석
print(author) # 박해선
print(year)   # 2022
print(age)    # None


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
    print(type(book))  # <class 'xml.etree.ElementTree.Element'>
    name = book.findtext('name')
    author = book.findtext('author')
    year = book.findtext('year')
    
    print(name)
    print(author)
    print(year)
    print()


# In[22]:

# 판다스에서 XML 형식 처리    
import pandas as pd

pd_xml = pd.read_xml(x2_str)
