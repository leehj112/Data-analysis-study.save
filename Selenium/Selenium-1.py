# -*- coding: utf-8 -*-
"""
Created on Fri May  3 09:44:56 2024

@author: leehj
"""


###### ### 7.3.2 가상 화폐 거래 정보 가져오기


# In[ ]:


import requests
import pandas as pd

url = "http://coinmarketcap.com/ko/" # URL 지정
html = requests.get(url).text        # HTML 소스 가져오기
dfs = pd.read_html(html)             # HTML 소스에서 table의 내용을 DataFrame 리스트로 변환

df = dfs[0]       # 리스트의 첫 번째 요소를 선택
df.iloc[0:12,1:6] # DataFrame 데이터에서 행과 열을 선택해 출력


# In[ ]:


from selenium.webdriver import Chrome
from bs4 import BeautifulSoup

driver = Chrome()          # 크롬 드라이버 객체 생성

url = "https://coinmarketcap.com/ko/" # URL 지정
driver.get(url)            # 웹 브라우저를 실행해 지정한 URL에 접속
driver.implicitly_wait(3)  # 웹 사이트의 내용을 받아오기까지 기다림

html = driver.page_source  # 접속 후에 해당 page의 HTML 소스를 가져옴
dfs = pd.read_html(html)   # HTML 소스에서 table의 내용을 DataFrame 리스트로 변환

df = dfs[0]       # 리스트의 첫 번째 요소를 선택
df.iloc[0:16,1:6] # DataFrame 데이터에서 행과 열을 선택해 출력


# [7장: 319페이지]

# In[ ]:


from selenium.webdriver import Chrome
from bs4 import BeautifulSoup

driver = Chrome() # 크롬 드라이버 객체 생성

url = "https://coinmarketcap.com/ko/" # URL 지정
driver.get(url)  # 웹 브라우저를 실행해 지정한 URL에 접속

# 웹 사이트 문서 높이 가져오기
scroll_height = driver.execute_script("return document.body.scrollHeight")

y = 0          # Y축 좌표의 초깃값
y_step = 1000  # Y축 아래로 이동하는 단계
while (True):
    y = y + y_step
    script =  "window.scrollTo(0,{0})".format(y)
    driver.execute_script(script) # 스크립트 실행
    driver.implicitly_wait(5)     # 스크롤 수행 후 데이터를 받아올 때까지 기다림
    
    # 수식 스크롤 위치가 문서 끝보다 크거나 같으면 while 문 빠져나가기
    if (y >= scroll_height):
        break
    
html = driver.page_source # HTML 코드를 가져옴
dfs = pd.read_html(html)  # HTML 소스에서 table의 내용을 DataFrame 리스트로 변환

df = dfs[0]         # 리스트의 첫 번째 요소를 선택
df.iloc[95:100,1:6] # DataFrame 데이터에서 행과 열을 선택해 출력


# [7장: 320페이지]

# In[ ]:


df.iloc[0:5,1:6]


# In[ ]:

lst = [name.replace(str(num), " ") for num, name in zip(df['#'], df['이름'])]

#%%
[(num, name) for num, name in zip(df['#'], df['이름'])]

#%%
df['이름'] = [name.replace(str(num), " ") for num, name in zip(df['#'], df['이름'])]
df['이름'] = [name.replace("구매하기", "") for name in df['이름']]
df.iloc[0:5,1:6]


# [7장: 321페이지]

# In[ ]:

import requests
import pandas as pd

from selenium.webdriver import Chrome
from bs4 import BeautifulSoup

def get_coin_info(page_num):
    # driver = Chrome() # 크롬 드라이버 객체 생성
    
    # page 추가해 URL 지정
    url = f"https://coinmarketcap.com/ko/?page={page_num}"
    driver.get(url)  # 웹 브라우저를 실행해 지정한 URL에 접속
    
    # 웹 브라우저를 실행하고 URL에 접속할 때까지 기다림
    time.sleep(3) 
    
    # 웹 사이트 문서 높이 가져오기
    scroll_height = driver.execute_script("return document.body.scrollHeight")

    y = 0           # Y축 좌표의 초깃값
    y_step = 1000   # Y축 아래로 이동하는 단계
    while (True):
        y = y + y_step
        script =  "window.scrollTo(0,{0})".format(y)
        driver.execute_script(script) # 스크립트 실행
        driver.implicitly_wait(5) # 스크롤 수행 후 데이터를 받아올 때까지 기다림

        # 수식 스크롤 위치가 문서 끝보다 크거나 같으면 while 문 빠져나가기
        if (y >= scroll_height):
            break

    html = driver.page_source # HTML 코드를 가져옴
    dfs = pd.read_html(html)  # HTML 소스에서 table의 내용을 DataFrame 리스트로 변환

    df = dfs[0] # 리스트의 첫 번째 요소를 선택
    
    # '이름' 열의 내용 변경
    df['이름'] = [name.replace(str(num), " ") for num, name in zip(df['#'], df['이름'])]
    df['이름'] = [name.replace("구매하기", "") for name in df['이름']]
    
    # driver.quit() # 웹 브라우저를 종료함

    return df.iloc[:,1:9]


# [7장: 322페이지]

# In[ ]:


page_num = 1 # page 지정
df_coin = get_coin_info(page_num) # 함수 호출

# DataFrame 데이터에서 행과 열을 선택해 출력
with pd.option_context('display.max_rows',6):
    pd.set_option("show_dimensions", False) # DataFrame의 행과 열 개수 출력 안 하기
    # display(df_coin.iloc[:,0:6])
    print(df_coin.iloc[:,0:6])
    

#%%

driver = Chrome() # 크롬 드라이버 객체 생성

import time
page_num = 1 # page 지정
# df1 = get_coin_info(1) # 함수 호출
# df2 = get_coin_info(2) # 함수 호출

dfx = get_coin_info(page_num) # 함수 호출

for page in range(2, 4):
    df_coin = get_coin_info(page) # 함수 호출
    dfx = pd.concat([dfx, df_coin], ignore_index=True, axis=0)
    time.sleep(3)

driver.quit() # 웹 브라우저를 종료함
#%%



# ### 7.3.3 유튜브 검색 결과 가져오기

# [7장: 324페이지]

# In[ ]:


from selenium.webdriver import Chrome
from bs4 import BeautifulSoup
import time

driver = Chrome() # 크롬 드라이버 객체 생성

base_url = "https://www.youtube.com" # 유튜브의 기본 URL
search_word = '/results?search_query=' + '방탄소년단' # 검색어
url = base_url +  search_word        # 접속하고자 하는 웹 사이트

driver.get(url) # 웹 브라우저를 실행해 지정한 URL에 접속
time.sleep(3)   # 웹 브라우저를 실행하고 URL에 접속할 때까지 기다림

print("- 접속한 웹 사이트의 제목:", driver.title) # 접속한 웹 사이트의 제목 출력
print("- 접속한 웹 사이트의 URL:", driver.current_url) # 접속한 웹 사이트의 URL 출력