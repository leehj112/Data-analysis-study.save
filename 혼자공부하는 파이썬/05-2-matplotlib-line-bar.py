# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:11:28 2024

@author: leehj
"""

#!/usr/bin/env python
# coding: utf-8

# # 05-2 선, 막대 그래프 그리기

# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://nbviewer.jupyter.org/github/rickiepark/hg-da/blob/main/05-2.ipynb"><img src="https://jupyter.org/assets/share.png" width="61" />주피터 노트북 뷰어로 보기</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/rickiepark/hg-da/blob/main/05-2.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />구글 코랩(Colab)에서 실행하기</a>
#   </td>
# </table>

# ## 연도별 발행 도서 개수 구하기

# In[1]:


import gdown

gdown.download('https://bit.ly/3pK7iuu', './data/ns_book7.csv', quiet=False)


# In[2]:


import pandas as pd

ns_book7 = pd.read_csv('./data/ns_book7.csv', low_memory=False)
ns_book7.head()


# In[3]:

# value_counts()
# 고유한 값의 등장 횟수
# 지정된 컬럼의 값이 인덱스
count_by_year = ns_book7['발행년도'].value_counts()
count_by_year


# In[4]:

# value_counts()
# 정렬: 오름차순
count_by_year = count_by_year.sort_index()
count_by_year


# In[5]:

# 인덱스의 값이 2030이하인 자료만 추출
count_by_year = count_by_year[count_by_year.index <= 2030]
count_by_year


#%%
# ## 주제별 도서 개수 구하기

# In[6]:

# 컬럼('주제분류번호')의 자료형은?
print(ns_book7.info()) # 주제분류번호   359792 non-null  object
print(ns_book7.dtypes) # 전체 컬럼의 자료형 확인
print(ns_book7['주제분류번호'].dtype) # object

#%%
import numpy as np

ns_book_subject_type = ns_book7['주제분류번호'].dtype

print("ns_book_subject_type:", type(ns_book_subject_type)) # <class 'numpy.dtypes.ObjectDType'>

print("ns_book_subject_type:", ns_book_subject_type) # object
print(isinstance(ns_book_subject_type, object)) # True
print(isinstance(ns_book_subject_type, str))    # False
print(isinstance(ns_book_subject_type, np.dtypes.ObjectDType)) # True

#%%
import numpy as np

cnt = 0
def kdc_1st_char(no):
    global cnt
    cnt += 1
    print(f"[{cnt}] {no} ", end=',')
    if no is np.nan:
        print(no)
        return '-1'
    else:
        print(f"{no[0]} ")
        return no[0]

count_by_subject = ns_book7['주제분류번호'].apply(kdc_1st_char).value_counts()
count_by_subject


#%%
# ## 선 그래프 그리기

# In[7]:


import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 100


# In[8]:

# '발행년도'별 도서수
plt.plot(count_by_year.index, count_by_year.values)
plt.title('Books by year')
plt.xlabel('year')
plt.ylabel('number of books')
plt.show()


# In[9]:

# https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
# linestyle=':' : dotted
# plt.plot(count_by_year, marker='.', linestyle=':', color='red')
# plt.plot(count_by_year, marker='.', linestyle='dotted', color='red')
# plt.plot(count_by_year, marker='.', linestyle=('dotted'), color='#0000ff') # blue
plt.plot(count_by_year, marker='.', linestyle=('dotted'), color='#000000') # black
plt.title('Books by year')
plt.xlabel('year')
plt.ylabel('number of books')
plt.show()


# In[10]:

# *-g : 별모양, solid, green
plt.plot(count_by_year, '*-g')
plt.title('Books by year')
plt.xlabel('year')
plt.ylabel('number of books')
plt.show()


# In[11]:

# plt.xticks(range(1947, 2030, 10)) 
# plt.annotate(val, (idx, val))

plt.plot(count_by_year, '*-g')
plt.title('Books by year')
plt.xlabel('year')
plt.ylabel('number of books')
plt.xticks(range(1947, 2030, 10))
for idx, val in count_by_year[::5].items():
    plt.annotate(val, (idx, val))
plt.show()


# In[12]:


plt.plot(count_by_year, '*-g')
plt.title('Books by year')
plt.xlabel('year')
plt.ylabel('number of books')
plt.xticks(range(1947, 2030, 10))
for idx, val in count_by_year[::5].items():
    plt.annotate(val, (idx, val), xytext=(idx+1, val+10))
plt.show()


# In[13]:


plt.plot(count_by_year, '*-g')
plt.title('Books by year')
plt.xlabel('year')
plt.ylabel('number of books')
plt.xticks(range(1947, 2030, 10))
for idx, val in count_by_year[::5].items():
    plt.annotate(val, (idx, val), xytext=(2, 2), textcoords='offset points')
plt.show()


#%%
# ## 막대 그래프 그리기

plt.bar(count_by_subject.index, count_by_subject.values)
plt.title('Books by subject')
plt.xlabel('subject')
plt.ylabel('number of books')
for idx, val in count_by_subject.items():
    plt.annotate(val, (idx, val), xytext=(0, 2), textcoords='offset points')
plt.show()


# In[15]:


plt.bar(count_by_subject.index, count_by_subject.values, width=0.7, color='blue')
plt.title('Books by subject')
plt.xlabel('subject')
plt.ylabel('number of books')
for idx, val in count_by_subject.items():
    plt.annotate(val, (idx, val), xytext=(0, 2), textcoords='offset points', 
                 fontsize=8, ha='center', color='green')
plt.show()


# In[16]:


plt.barh(count_by_subject.index, count_by_subject.values, height=0.7, color='blue')
plt.title('Books by subject')
plt.xlabel('number of books')
plt.ylabel('subject')
for idx, val in count_by_subject.items():
    plt.annotate(val, (val, idx), xytext=(2, 0), textcoords='offset points', 
                 fontsize=8, va='center', color='green')
plt.show()


#%%
# ## 이미지 출력하고 저장하기

# In[17]:


# 노트북이 코랩에서 실행 중인지 체크합니다.
"""
import sys
if 'google.colab' in sys.modules:
    # 샘플 이미지를 다운로드합니다.
    get_ipython().system('wget https://bit.ly/3wrj4xf -O jupiter.png')
"""

# In[18]:


img = plt.imread('./data/jupiter.png')

# 높이, 너비, 채널
img.shape # (1561, 1646, 3)


# In[19]:


plt.imshow(img)
plt.show()


# In[20]:


plt.figure(figsize=(8, 6))
plt.imshow(img)
plt.axis('off')
plt.show()


# In[21]:


from PIL import Image

pil_img = Image.open('./data/jupiter.png')
plt.figure(figsize=(8, 6))
plt.imshow(pil_img)
plt.axis('off')
plt.show()


# In[22]:


import numpy as np

arr_img = np.array(pil_img)
arr_img.shape # (1561, 1646, 3)


# In[23]:


plt.imsave('./data/jupiter-save.jpg', arr_img)


# ## 그래프를 이미지로 저장하기

# In[24]:


plt.rcParams['savefig.dpi']


# In[25]:


plt.barh(count_by_subject.index, count_by_subject.values, height=0.7, color='blue')
plt.title('Books by subject')
plt.xlabel('number of books')
plt.ylabel('subject')
for idx, val in count_by_subject.items():
    plt.annotate(val, (val, idx), xytext=(2, 0), textcoords='offset points', 
                 fontsize=8, va='center', color='green')
plt.savefig('./data/books_by_subject.png')
plt.show()


# In[26]:


pil_img = Image.open('./data/books_by_subject.png')

plt.figure(figsize=(8, 6))
plt.imshow(pil_img)
plt.axis('off')
plt.show()
