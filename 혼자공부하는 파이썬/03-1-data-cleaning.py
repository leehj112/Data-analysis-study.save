# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:10:13 2024

@author: leehj
"""

#!/usr/bin/env python
# coding: utf-8

# # 03-1 불필요한 데이터 삭제하기

# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://nbviewer.jupyter.org/github/rickiepark/hg-da/blob/main/03-1.ipynb"><img src="https://jupyter.org/assets/share.png" width="61" />주피터 노트북 뷰어로 보기</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/rickiepark/hg-da/blob/main/03-1.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />구글 코랩(Colab)에서 실행하기</a>
#   </td>
# </table>

# ## 열 삭제하기

# In[40]:


import gdown

gdown.download('https://bit.ly/3RhoNho', './data/ns_202104.csv', quiet=False)


# In[41]:


import pandas as pd

ns_df = pd.read_csv('./data/ns_202104.csv', low_memory=False)
ns_df.head()


# In[42]:

# 인덱스 참조
# 행: 전체
# 열: '번호'부터 '등록일자'
# 맨 마지막에 위한 컬럼('Unnamed: 13')을 제외
ns_book = ns_df.loc[:, '번호':'등록일자']
ns_book.head()


# In[43]:

# 데이터프레임의 컬럼 목록
print(ns_df.columns)
"""
Index(['번호', '도서명', '저자', '출판사', '발행년도', 'ISBN', '세트 ISBN', '부가기호', '권',
       '주제분류번호', '도서권수', '대출건수', '등록일자', 'Unnamed: 13'],
      dtype='object')
"""

# In[44]:


print(ns_df.columns[0])  # 번호
print(ns_df.columns[-1]) # Unnamed: 13


# In[45]:

# 컬럼('Unnamed: 13')이 아닌 컬럼은 True
# 컬럼('Unnamed: 13')인 컬럼은 False
ns_df.columns != 'Unnamed: 13'
"""
array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
        True,  True,  True,  True, False])
"""
# In[46]:

# 컬럼('Unnamed: 13')이 아닌 컬럼만 선택해서 데이터프레임 생성
selected_columns = ns_df.columns != 'Unnamed: 13' # Array of bool(numpy)

# 리스트로 컬럼을 지정할 수 있다.
# selected_columns = list(selected_columns)         # list

ns_book = ns_df.loc[:, selected_columns]
ns_book.head()


# In[47]:

# 컬럼('부가기호')을 제외
# ns_book = ns_df.loc[:, ns_df.columns != '부가기호']

selected_columns = ns_df.columns != '부가기호'
ns_book = ns_df.loc[:, selected_columns]
ns_book.head()


# In[48]:

# drop() 메서드를 이용해서 열을 삭제
# axis : 0(행), 1(열)    
ns_book = ns_df.drop('Unnamed: 13', axis=1)
ns_book.head()


# In[49]:

# 여러 개의 열을 삭제 
# 리스트에 컬럼 목록을 지정
ns_book = ns_df.drop(['부가기호','Unnamed: 13'], axis=1)
ns_book.head()


# In[50]:

# 원본 데이터프레임을 변경 : inplace(True)
# 리턴 : None
ndf = ns_book.drop('주제분류번호', axis=1, inplace=True)
ns_book.head()

"""
# 이미 지워진 컬럼을 다시 지우려 하면 에러
# 또는 없는 컬럼을 지우려 하면 에러
KeyError: "['주제분류번호'] not found in axis"
"""

#%%

colname = '주제분류번호'

try:
    ndf = ns_book.drop(colname, axis=1, inplace=True)
    ns_book.head()
except KeyError as e:
    print(f"존재하지 않는 컬럼을 삭제하려 했습니다.: 컬럼({colname})")

# In[51]:

# 열에 하나라도 nan이 포함되어 있으면 칼럼을 삭제
# axis: 1(열)
ns_book = ns_df.dropna(axis=1)
ns_book.head()  # 14 -> 5

ns_df_col_len = len(ns_df.columns)
ns_book_col_len = len(ns_book.columns)
print("ns_df.columns: ", ns_df_col_len)     # 14
print("ns_book.columns: ", ns_book_col_len) # 5


# In[52]:

# 칼럼의 모든 데이터가 nan이면 칼럼을 삭제
ns_book = ns_df.dropna(axis=1, how='all')
ns_book.head()
print("ns_book.columns: ", len(ns_book.columns)) # 13


# ## 행 삭제하기

# In[53]:

# 행의 인덱스 0,1 삭제
# axis : 0(행), 기본값
ns_book2 = ns_book.drop([0,1])
ns_book2.head()


# In[54]:

# 슬라이싱
# 행선택 : 2행부터 끝까지
ns_book2 = ns_book[2:]
ns_book2.head() # 401680


# In[55]:

# 슬라이스는 끝 번호가 포함되지 않음: 0,1 행 선택
ns_book2 = ns_book[0:2]
ns_book2.head()


# In[56]:

# 컬럼('출판사')의 값이 '한빛미디어'인 행을 선택 : 불리언 배열
# 결과: Series(True, False)
selected_rows = ns_df['출판사'] == '한빛미디어'
ns_book2 = ns_book[selected_rows]
ns_book2.head()


# In[57]:

# loc[]
ns_book2A = ns_book.loc[selected_rows]
ns_book2A.head()


# In[58]:


# 컬럼('대출건수')의 값이 1000건보다 많은 행을 선택 : 불리언 배열
ns_book2_selected = ns_book['대출건수'] > 1000
ns_book2 = ns_book[ns_book2_selected]
ns_book2.head()

#%%

# ## 중복된 행 찾기
# DataFrame.duplicated() 메서드 
#   - 중복된 행 중에서 처음 행을 제외한 나머지 행을 True
#   - 그 외에 중복되지 않은 나머지 모든 행을 False
#   - keep: 'first', 'last', False
#     . 'first' : 첫 번째 데이터를 제외하고 중복 처리
#     . 'last'  : 마지막 데이터를 제외하고 중복 처리
#     . False   : 모든 중복 데이터를 중복 처리 
# In[59]:


# 전체 칼럼을 기준으로 중복되는 행
sum(ns_book.duplicated()) # 0


# In[60]:

# 칼럼이 '도서명','저자','ISBN'에서 중복되는 행
sum(ns_book.duplicated(subset=['도서명','저자','ISBN'])) # 22096


# In[61]:

# 중복된 행 중에서 처음 행을 제외한 나머지 행을 True
# 그 외에 중복되지 않은 나머지 모든 행은 False    
# dup_rowsT = ns_book.duplicated(subset=['도서명','저자','ISBN'], keep='first')
dup_rowsT = ns_book.duplicated(subset=['도서명','저자','ISBN'])


#%%
# index: 85165,89933
bklst = ['도서명','저자','ISBN']
dulst = [85165, 89933]

ns_book.loc[dulst,bklst]
dup_rowsT.loc[dulst]
"""
            도서명           저자           ISBN
85165  #진로스타그램   청년기획단 너랑 지음  9791157232703
89933  #진로스타그램   청년기획단 너랑 지음  9791157232703
#
85165    False
89933     True
dtype: bool
"""

#%%

# 중복된 행 중에서 처음 행을 제외한 나머지 행을 True
# 그 외에 중복되지 않은 나머지 모든 행은 False  
#
# keep=False: 중복된 행을 모두 True로 표시한 불리언 배열을 반환
dup_rows = ns_book.duplicated(subset=['도서명','저자','ISBN'], keep=False)
ns_book3 = ns_book[dup_rows]
ns_book3.head()

dup_rows.loc[dulst]
"""
85165    True
89933    True
dtype: bool
"""
# In[62]:


#%%

# P171 그룹별로 모으기
# DataFrame.groupby() 

#%%
count_df = ns_book[['도서명','저자','ISBN','권','대출건수']]


# In[63]:

# by : 그룹핑을 할 컬럼, 
# dropna : NaN 포함 유무, True or False
#
# by=['도서명','저자','ISBN','권']
# dropna=False : NaN이 있는 행을 삭제하지 않음
# 결과: DataFrameGroupBy 객체
# 인덱스: 그룹으로 지정된 컬럼
# 컬럼: 대출건수
group_df = count_df.groupby(by=['도서명','저자','ISBN','권'], dropna=False)

#%%

loan_count = group_df.sum()
loan_mean = group_df.mean()
print('loan_count:', loan_count.head())
print('loan_mean:', loan_mean.head())


# In[64]:


loan_count = count_df.groupby(by=['도서명','저자','ISBN','권'], dropna=False).sum()
loan_count.head()


# In[65]:


dup_rows = ns_book.duplicated(subset=['도서명','저자','ISBN','권'])
unique_rows = ~dup_rows # 반전(~)
ns_book3 = ns_book[unique_rows].copy() # 복사본, 새로운 데이터프레임 생성


# In[66]:

# 중복이 있는지 확인 
# 결과 : 0, 중복이 없음
sum(ns_book3.duplicated(subset=['도서명','저자','ISBN','권'])) # 0


# In[67]:

# 인덱스: ['도서명','저자','ISBN','권']
# 원본수정: inplace=True
ns_book3.set_index(['도서명','저자','ISBN','권'], inplace=True)
ns_book3.head()


# In[68]:

# 다른 데이터프레임을 사용해 원본 데이터프레임의 값을 업데이트
ns_book3.update(loan_count)
ns_book3.head()


# In[69]:

# 인덱스를 컬럼으로 이동
ns_book4 = ns_book3.reset_index()
ns_book4.head()


# In[70]:


sum(ns_book['대출건수']>100) # 2311


# In[71]:

# 중복된 도서의 대출 건수 포함
sum(ns_book4['대출건수']>100) # 2550


# In[72]:


ns_book4 = ns_book4[ns_book.columns]
ns_book4.head()


# In[73]:


ns_book4.to_csv('./data/ns_book4.csv', index=False)


# In[74]:


def data_cleaning(filename):
    """
    남산 도서관 장서 CSV 데이터 전처리 함수
    
    :param filename: CSV 파일이름
    """
    # 파일을 데이터프레임으로 읽습니다.
    ns_df = pd.read_csv(filename, low_memory=False)
    # NaN인 열을 삭제합니다.
    ns_book = ns_df.dropna(axis=1, how='all')

    # 대출건수를 합치기 위해 필요한 행만 추출하여 count_df 데이터프레임을 만듭니다.
    count_df = ns_book[['도서명','저자','ISBN','권','대출건수']]
    # 도서명, 저자, ISBN, 권을 기준으로 대출건수를 groupby합니다.
    loan_count = count_df.groupby(by=['도서명','저자','ISBN','권'], dropna=False).sum()
    # 원본 데이터프레임에서 중복된 행을 제외하고 고유한 행만 추출하여 복사합니다.
    dup_rows = ns_book.duplicated(subset=['도서명','저자','ISBN','권'])
    unique_rows = ~dup_rows
    ns_book3 = ns_book[unique_rows].copy()
    # 도서명, 저자, ISBN, 권을 인덱스로 설정합니다.
    ns_book3.set_index(['도서명','저자','ISBN','권'], inplace=True)
    # load_count에 저장된 누적 대출건수를 업데이트합니다.
    ns_book3.update(loan_count)
    
    # 인덱스를 재설정합니다.
    ns_book4 = ns_book3.reset_index()
    # 원본 데이터프레임의 열 순서로 변경합니다.
    ns_book4 = ns_book4[ns_book.columns]
    
    return ns_book4


# In[75]:

# 위에서 순차적으로 처리한 결과 ns_book4와
# 함수 data_cleaning()을 이용해서 ns_book4의 
# 원본 파일('ns_202104.csv')의 처리 결과가 동일한지 확인
# 결과: True, 함수 data_cleaning()은 이상 없다.
new_ns_book4 = data_cleaning('./data/ns_202104.csv')

ns_book4.equals(new_ns_book4) # True
