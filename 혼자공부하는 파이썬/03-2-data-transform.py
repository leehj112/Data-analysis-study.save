# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:10:23 2024

@author: leehj
"""

#!/usr/bin/env python
# coding: utf-8

# # 03-2 잘못된 데이터 수정하기

# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://nbviewer.jupyter.org/github/rickiepark/hg-da/blob/main/03-2.ipynb"><img src="https://jupyter.org/assets/share.png" width="61" />주피터 노트북 뷰어로 보기</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/rickiepark/hg-da/blob/main/03-2.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />구글 코랩(Colab)에서 실행하기</a>
#   </td>
# </table>

# ## 데이터프레임 정보 요약 확인하기

# In[ ]:


import gdown

#%%

gdown.download('https://bit.ly/3GisL6J', './data/ns_book4.csv', quiet=False)


# In[ ]:


import pandas as pd

ns_book4 = pd.read_csv('./data/ns_book4.csv', low_memory=False)
ns_book4.head()


# In[ ]:


# NaN이 아닌 값의 갯수
# 메모리: memory usage: 38.1+ MB
#   - 기본적으로 원소 개수와 데이터 타입을 기반으로
#     메모리 사용량을 추정
ns_book4.info()


# In[ ]:

# 정확한 메모리 사용량 표시
# memory usage: 298.2 MB
ns_book4.info(memory_usage='deep')


#%%
# ## 누락된 값 처리하기
# True  : NaN
# False : NaN 아닌 정상적인 값
ns_isna = ns_book4.isna()

#%%

# 결과 : Series
# 각 컬럼별로 NaN 건수 집계
ns_isna_sum = ns_book4.isna().sum()
"""
번호                  0
도서명              403
저자                198
출판사             4641
발행년도             14
ISBN                  0
세트 ISBN        328015
부가기호          74205
권               321213
주제분류번호      19864
도서권수              0
대출건수              0
등록일자              0
dtype: int64
"""

#%%

# notna()
# 누락되지 않은 값을 확인
ns_notna_sum = ns_book4.notna().sum()
print(ns_notna_sum)
"""
번호             384591
도서명           384188
저자             384393
출판사           379950
발행년도         384577
ISBN             384591
세트 ISBN         56576
부가기호         310386
권                63378
주제분류번호     364727
도서권수         384591
대출건수         384591
등록일자         384591
dtype: int64
"""

# In[ ]:

ns_book4.loc[0, '도서권수'] # 1
print('도서권수의 자료형:', ns_book4['도서권수'].dtype) # int64

#%%

# 자료형이 정수계열(int64)에 None을 넣으면 값은 nan
ns_book4.loc[0, '도서권수'] = None
ns_book4.loc[0, '도서권수'] # nan

ns_book4['도서권수'].isna().sum() # 1


# In[ ]:


ns_book4.head(2)


#%%
    
#%%
# 특정컬럼의 자료형 확인
# DataFrame['컬럼'].dtype

ns_book4.loc[0, '도서권수'] = 1
ns_book4.loc[0, '도서권수'] # 1
ns_book4.head(2)
ns_book4.info() #  10  도서권수     384591 non-null  float64
print('도서권수의 자료형:', ns_book4['도서권수'].dtype) # float64

#%%

# 자료형 변환 : astype()
ns_book4 = ns_book4.astype({'도서권수':'int32', '대출건수': 'int64'})
ns_book4.info() #  10  도서권수     384591 non-null  float64
print('도서권수의 자료형:', ns_book4['도서권수'].dtype) # int32
print('대출건수의 자료형:', ns_book4['대출건수'].dtype) # int64

#%%

print(ns_book4['부가기호'].dtype) # object
    
#%%    

# # 자료형이 문자열(object)에 None을 넣으면 값은 None
ns_book4.loc[0, '부가기호'] = None
print(ns_book4.loc[0, '부가기호']) # None

#%%

import numpy as np

ns_book4.loc[0, '부가기호'] = np.nan
print(ns_book4.loc[0, '부가기호']) # nan
print(ns_book4.isna().sum())


# In[ ]:

# 세트 ISBN: isna -> 328015
ns_book4['세트 ISBN'].isna().sum() # 328015

#%%    
# NaN -> 빈 문자열('') 대체

set_isbn_na_rows = ns_book4['세트 ISBN'].isna()
#%%

ns_book4.loc[set_isbn_na_rows, '세트 ISBN'] = ''
ns_book4['세트 ISBN'].isna().sum() # 0


# In[ ]:

# nan을 지정된 값으로 모두 채움
# nan -> '없음'    
ns_none = ns_book4.fillna('없음')    

#%%    

# nan을 '없음'으로 모두 바꾼 후
# isna()는 zero이므로 sum()의 결과는 모두 zero이다.
ns_book4.fillna('없음').isna().sum()
"""
번호           0
도서명         0
저자           0
출판사         0
발행년도       0
ISBN           0
세트 ISBN      0
부가기호       0
권             0
주제분류번호   0
도서권수       0
대출건수       0
등록일자       0
dtype: int64
"""


# In[ ]:

# 지정된 컬럼에서 nan을 지정된 값으로 채움
# isna().sum()을 수행하면 zero
ns_book4['부가기호'].fillna('없음').isna().sum() # 0


# In[ ]:

# 지정된 컬럼에서 nan을 지정된 값으로 채우고
# 전체 데이터프레임을 대상으로 처리
# 결과: 부가기호 0건
ns_book4.fillna({'부가기호':'없음'}).isna().sum()
"""
번호                0
도서명            403
저자              198
출판사           4641
발행년도           14
ISBN                0
세트 ISBN           0
부가기호            0
권             321213
주제분류번호    19864
도서권수            1
대출건수            0
등록일자            0
dtype: int64
"""

# In[ ]:

# replace(원래값, 새로운값)
ns_book4.replace(np.nan, '없음').isna().sum()


# In[ ]:

# 여러 값을 대상으로 처리
# np.nan -> '없음'
# '2021' -> '21'
# 리스트 형태로 지정
ndf = ns_book4.replace([np.nan, '2021'], ['없음', '21'])

#%%
ndf.replace(['없음'], [np.nan], inplace=True) # 원본변경
#%%

#%%

# 지정된 컬럼만 변경 : 리스트 형태로 지정
ndf = ns_book4['발행년도'].replace([np.nan, '2021'], ['없음', '21'])

# In[ ]:

# 딕셔너리 형태로 지정
ns_book4.replace({np.nan: '없음', '2021': '21'}).head(2)


# In[ ]:

# 열이름을 지정하고 특정한 값을 대체
ns_book4.replace({'부가기호': np.nan}, '없음').head(2)


# In[ ]:

# 다중의 열이름을 지정하고 특정한 값을 대체
ns_book4.replace({'부가기호': {np.nan: '없음'}, '발행년도': {'2021': '21'}}).head(2)


#%%
# ## 정규 표현식

# In[ ]:

ns_2021 = ns_book4.replace({'발행년도': {'2021': '21'}})[100:102]

# In[ ]:

# 괄호: 그룹
# 12(34) -> (34) : 괄로로 묶은 (34)가 첫번째(r'\1') 그룹
# r'\1' : 괄로에 묶은 첫번째 그룹
# regex=True : 정규 표현식
ns_99 = ns_book4.replace({'발행년도': {r'\d\d(\d\d)': r'\1'}}, regex=True)[100:102]


# In[ ]:

# r'\d{2}(\d{2})' : 숫자2자리, 숫자2자리(그룹1)
# r'\1' : 그룹1
ns_book4.replace({'발행년도': {r'\d{2}(\d{2})': r'\1'}}, regex=True)[100:102]



# In[ ]:

# 삭제 문자: ' (지은이)', ' (옮긴이)'
# .* : 임의의 문자가 0개 이상
# \s : 공백(space)
# \( : '('
# \) : ')'
ns_name = ns_book4.replace({'저자': {r'(.*)\s\(지은이\)(.*)\s\(옮긴이\)': r'\1\2'},
                  '발행년도': {r'\d{2}(\d{2})': r'\1'}}, regex=True)[100:102]

print(ns_book4.loc[100:101,'저자'])
"""
100    헨리 클라우드, 존 타운센드 (지은이), 김진웅 (옮긴이)
101          로런스 인그래시아 (지은이), 안기순 (옮긴이)
Name: 저자, dtype: object
"""

print(ns_name['저자'])
"""
100    헨리 클라우드, 존 타운센드, 김진웅
101          로런스 인그래시아, 안기순
Name: 저자, dtype: object
"""
#%%

# ## 잘못된 값 바꾸기

# In[ ]:

print(ns_book4['발행년도'].dtype) # object

#%%    

# 아래 코드는 오류 발생
# ns_book4.astype({'발행년도': 'int32'})
# ValueError: invalid literal for int() with base 10: '1988.'

# In[ ]:

# '발행년도'가 '1988.'인 데이터
pub_1988 = ns_book4[ns_book4['발행년도'] == '1988.']
    
#%%
ns_book4['발행년도'].str.contains('1988.').sum() # 20

#%%
ns_book4['발행년도'].str.contains('1988').sum() # 407


# In[ ]:

# 정규표현식
# \D : \d의 반대, 숫자가 아닌 것
invalid_number = ns_book4['발행년도'].str.contains('\D', na=True)
print(invalid_number.sum())     # 1777
ns_book4[invalid_number].head()

df_invalid_number = ns_book4[invalid_number]


# In[ ]:


ns_book5 = ns_book4.replace({'발행년도':r'.*(\d{4}).*'}, r'\1', regex=True)
ns_book5[invalid_number].head()


# In[ ]:


unkown_year = ns_book5['발행년도'].str.contains('\D', na=True)
print(unkown_year.sum())      # 67
ns_book5[unkown_year].head()


# In[ ]:


ns_book5.loc[unkown_year, '발행년도'] = '-1'
ns_book5 = ns_book5.astype({'발행년도': 'int32'})


# In[ ]:

# gt() : great than
ns_book5['발행년도'].gt(4000).sum()  # 131


# In[ ]:


dangun_yy_rows = ns_book5['발행년도'].gt(4000)
ns_book5.loc[dangun_yy_rows, '발행년도'] = ns_book5.loc[dangun_yy_rows, '발행년도'] - 2333


# In[ ]:


dangun_year = ns_book5['발행년도'].gt(4000)
print(dangun_year.sum())      # 13
ns_book5[dangun_year].head(2)


# In[ ]:


ns_book5.loc[dangun_year, '발행년도'] = -1


# In[ ]:

# lt : less than, 작은 값
# 0보다 크고 1900보다 작은 값
old_books = ns_book5['발행년도'].gt(0) & ns_book5['발행년도'].lt(1900)
ns_book5[old_books]


# In[ ]:


ns_book5.loc[old_books, '발행년도'] = -1


# In[ ]:


ns_book5['발행년도'].eq(-1).sum() # 86


# ## 누락된 정보 채우기

# In[ ]:


na_rows = ns_book5['도서명'].isna() | ns_book5['저자'].isna() \
          | ns_book5['출판사'].isna() | ns_book5['발행년도'].eq(-1)
print(na_rows.sum())      # 5268
ns_book5[na_rows].head(2)


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
from bs4 import BeautifulSoup


# In[ ]:


def get_book_title(isbn):
    # Yes24 도서 검색 페이지 URL
    url = 'http://www.yes24.com/Product/Search?domain=BOOK&query={}'
    # URL에 ISBN을 넣어 HTML 가져옵니다.
    r = requests.get(url.format(isbn))
    soup = BeautifulSoup(r.text, 'html.parser')   # HTML 파싱
    # 클래스 이름이 'gd_name'인 a 태그의 텍스트를 가져옵니다.
    title = soup.find('a', attrs={'class':'gd_name'}) \
            .get_text()
    return title


# In[ ]:

# 테스트 용
get_book_title(9791191266054)


# In[ ]:


import re

def get_book_info(row):
    title = row['도서명']
    author = row['저자']
    pub = row['출판사']
    year = row['발행년도']
    # Yes24 도서 검색 페이지 URL
    url = 'http://www.yes24.com/Product/Search?domain=BOOK&query={}'
    # URL에 ISBN을 넣어 HTML 가져옵니다.
    r = requests.get(url.format(row['ISBN']))
    soup = BeautifulSoup(r.text, 'html.parser')   # HTML 파싱
    try:
        if pd.isna(title):
            # 클래스 이름이 'gd_name'인 a 태그의 텍스트를 가져옵니다.
            title = soup.find('a', attrs={'class':'gd_name'}) \
                    .get_text()
    except AttributeError:
        pass

    try:
        if pd.isna(author):
            # 클래스 이름이 'info_auth'인 span 태그 아래 a 태그의 텍스트를 가져옵니다.
            authors = soup.find('span', attrs={'class':'info_auth'}) \
                          .find_all('a')
            author_list = [auth.get_text() for auth in authors]
            author = ','.join(author_list)
    except AttributeError:
        pass

    try:
        if pd.isna(pub):
            # 클래스 이름이 'info_auth'인 span 태그 아래 a 태그의 텍스트를 가져옵니다.
            pub = soup.find('span', attrs={'class':'info_pub'}) \
                      .find('a') \
                      .get_text()
    except AttributeError:
        pass

    try:
        if year == -1:
            # 클래스 이름이 'info_date'인 span 태그 아래 텍스트를 가져옵니다.
            year_str = soup.find('span', attrs={'class':'info_date'}) \
                           .get_text()
            # 정규식으로 찾은 값 중에 첫 번째 것만 사용합니다.
            year = re.findall(r'\d{4}', year_str)[0]
    except AttributeError:
        pass

    return title, author, pub, year


# In[ ]:

# 샘플 테스트
# result_type='expand' : get_book_info()로부터 반환된 리턴 값을 각기 다른 열로 만듦
updated_sample = ns_book5[na_rows].head(2).apply(get_book_info,
    axis=1, result_type ='expand')
updated_sample


#%%
# 아래 코드 셀은 실행하는데 시간이 오래 걸립니다. 
# 편의를 위해 실행한 결과를 저장해 놓은 CSV 파일을 사용하겠습니다.

# In[ ]:

print(na_rows.sum())      # 5268

ns_book5_update = ns_book5[na_rows].apply(get_book_info, axis=1, result_type ='expand')

ns_book5_update.columns = ['도서명','저자','출판사','발행년도']
ns_book5_update.head()

#%%

# 편의를 위해 실행한 결과를 저장해 놓은 CSV 파일을 사용하겠습니다.
gdown.download('http://bit.ly/3UJZiHw', './data/ns_book5_update.csv', quiet=False)

ns_book5_update = pd.read_csv('./data/ns_book5_update.csv', index_col=0)
ns_book5_update.head()


# In[ ]:


ns_book5.update(ns_book5_update)

na_rows = ns_book5['도서명'].isna() | ns_book5['저자'].isna() \
          | ns_book5['출판사'].isna() | ns_book5['발행년도'].eq(-1)
print(na_rows.sum()) # 4615


# In[ ]:


ns_book5 = ns_book5.astype({'발행년도': 'int32'})


# In[ ]:

# ns_book5에서 ['도서명','저자','출판사']이 nan인 행을 제거
ns_book6 = ns_book5.dropna(subset=['도서명','저자','출판사'])

# '발행년도'가 -1이 아닌 행만 선택
ns_book6 = ns_book6[ns_book6['발행년도'] != -1]
ns_book6.head()


# In[ ]:


ns_book6.to_csv('./data/ns_book6.csv', index=False)


# In[ ]:

# 위의 처리 절차를 하나의 함수로 정리
def data_fixing(ns_book4):
    """
    잘못된 값을 수정하거나 NaN 값을 채우는 함수

    :param ns_book4: data_cleaning() 함수에서 전처리된 데이터프레임
    """
    # 도서권수와 대출건수를 int32로 바꿉니다.
    ns_book4 = ns_book4.astype({'도서권수':'int32', '대출건수': 'int32'})
    # NaN인 세트 ISBN을 빈문자열로 바꿉니다.
    set_isbn_na_rows = ns_book4['세트 ISBN'].isna()
    ns_book4.loc[set_isbn_na_rows, '세트 ISBN'] = ''

    # 발행년도 열에서 연도 네 자리를 추출하여 대체합니다. 나머지 발행년도는 -1로 바꿉니다.
    ns_book5 = ns_book4.replace({'발행년도':'.*(\d{4}).*'}, r'\1', regex=True)
    unkown_year = ns_book5['발행년도'].str.contains('\D', na=True)
    ns_book5.loc[unkown_year, '발행년도'] = '-1'

    # 발행년도를 int32로 바꿉니다.
    ns_book5 = ns_book5.astype({'발행년도': 'int32'})
    # 4000년 이상인 경우 2333년을 뺍니다.
    dangun_yy_rows = ns_book5['발행년도'].gt(4000)
    ns_book5.loc[dangun_yy_rows, '발행년도'] = ns_book5.loc[dangun_yy_rows, '발행년도'] - 2333
    # 여전히 4000년 이상인 경우 -1로 바꿉니다.
    dangun_year = ns_book5['발행년도'].gt(4000)
    ns_book5.loc[dangun_year, '발행년도'] = -1
    # 0~1900년 사이의 발행년도는 -1로 바꿉니다.
    old_books = ns_book5['발행년도'].gt(0) & ns_book5['발행년도'].lt(1900)
    ns_book5.loc[old_books, '발행년도'] = -1

    # 도서명, 저자, 출판사가 NaN이거나 발행년도가 -1인 행을 찾습니다.
    na_rows = ns_book5['도서명'].isna() | ns_book5['저자'].isna() \
              | ns_book5['출판사'].isna() | ns_book5['발행년도'].eq(-1)
    # 교보문고 도서 상세 페이지에서 누락된 정보를 채웁니다.
    updated_sample = ns_book5[na_rows].apply(get_book_info,
        axis=1, result_type ='expand')
    updated_sample.columns = ['도서명','저자','출판사','발행년도']
    ns_book5.update(updated_sample)

    # 도서명, 저자, 출판사가 NaN이거나 발행년도가 -1인 행을 삭제합니다.
    ns_book6 = ns_book5.dropna(subset=['도서명','저자','출판사'])
    ns_book6 = ns_book6[ns_book6['발행년도'] != -1]

    return ns_book6

###############################################################################
# THE END
###############################################################################
