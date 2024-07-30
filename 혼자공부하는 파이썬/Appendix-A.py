# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:19:37 2024

@author: leehj
"""

#!/usr/bin/env python
# coding: utf-8

# # 데이터베이스에서 가져오기

# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://nbviewer.jupyter.org/github/rickiepark/hg-da/blob/main/Appendix-A.ipynb"><img src="https://jupyter.org/assets/share.png" width="61" />주피터 노트북 뷰어로 보기</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/rickiepark/hg-da/blob/main/Appendix-A.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />구글 코랩(Colab)에서 실행하기</a>
#   </td>
# </table>

# ## 파이썬에서 SQL 사용하기: SQLite

# In[1]:


# 최신 sqlalchemy는 판다스에서 에러를 일으킵니다. 1.4.* 버전을 사용해 주세요. (https://github.com/pandas-dev/pandas/issues/40686)
get_ipython().system('pip install -U sqlalchemy==1.4.46')


# In[2]:


import sqlite3


# In[3]:


conn = sqlite3.connect('ns_lib.db')


# In[4]:


import gdown

gdown.download('https://bit.ly/3RhoNho', 'ns_202104.csv', quiet=False)


# **이전에 만든 nslib_book 테이블이 있다면 먼저먼저 삭제해 주세요.**

# In[5]:


c = conn.cursor()

c.execute("CREATE TABLE nslib_book \
          (name TEXT, author TEXT, borrow_count INTEGER)")


# In[6]:


c.execute("CREATE TABLE IF NOT EXISTS nslib_book \
          (name TEXT, author TEXT, borrow_count INTEGER)")


# In[7]:


c.execute("DROP TABLE nslib_book")


# In[8]:


c.execute("CREATE TABLE nslib_book \
          (name TEXT, author TEXT, borrow_count INTEGER)")


# ## 데이터프레임 데이터를 테이블에 추가하기

# In[9]:


import pandas as pd

ns_df = pd.read_csv('ns_202104.csv', low_memory=False)
ns_df.head()


# In[10]:


for index, row in ns_df.iterrows():
    c.execute("INSERT INTO nslib_book (name,author,borrow_count) \
              VALUES (?,?,?)", (row['도서명'], row['저자'], row['대출건수']))


# In[11]:


for index, row in ns_df.iterrows():
    pass


# In[12]:


book_df = ns_df[['도서명','저자','대출건수']]
book_df.head()


# In[13]:


book_df.columns = ['name', 'author', 'borrow_count']
book_df.head()


# In[14]:


book_df.to_sql('nslib_book', conn, if_exists='replace', index=False)


# ## 파이썬으로 테이블에서 데이터 읽기

# In[15]:


c.execute("SELECT * FROM nslib_book")


# In[16]:


c.fetchone()


# In[17]:


c.fetchone()


# In[18]:


c.fetchmany(3)


# In[19]:


all_rows = c.fetchall()


# In[20]:


book_df = pd.DataFrame(all_rows)
book_df.head()


# In[21]:


book_df = pd.read_sql_query("SELECT * FROM nslib_book", conn)
book_df.head()


# In[22]:


book_df = pd.read_sql_table('nslib_book', 'sqlite:///ns_lib.db')
book_df.head()


# ## 데이터베이스에서 제공하는 함수로 통계량 구하기

# In[23]:


len(book_df)


# In[24]:


c.execute("SELECT count(*) FROM nslib_book")
c.fetchone()


# In[25]:


c.execute("SELECT sum(borrow_count) FROM nslib_book")
c.fetchone()


# In[26]:


c.execute("SELECT avg(borrow_count) FROM nslib_book")
c.fetchone()


# ## 테이블 데이터 정렬하기

# In[27]:


c.execute("SELECT * FROM nslib_book ORDER BY borrow_count DESC LIMIT 10")
c.fetchall()


# In[28]:


c.close()
conn.close()
