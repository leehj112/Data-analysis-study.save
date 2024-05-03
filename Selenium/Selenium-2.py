# -*- coding: utf-8 -*-
"""
Created on Fri May  3 09:45:50 2024

@author: leehj
"""

# ### 7.3.3 유튜브 검색 결과 가져오기

# [7장: 324페이지]

# In[ ]:


from selenium.webdriver import Chrome
from bs4 import BeautifulSoup
import time

base_url = "https://www.youtube.com" # 유튜브의 기본 URL
search_word = '/results?search_query=' + '방탄소년단' # 검색어
url = base_url +  search_word        # 접속하고자 하는 웹 사이트

print(url)

#%%

driver = Chrome() # 크롬 드라이버 객체 생성

driver.get(url) # 웹 브라우저를 실행해 지정한 URL에 접속
time.sleep(3)   # 웹 브라우저를 실행하고 URL에 접속할 때까지 기다림

print("- 접속한 웹 사이트의 제목:", driver.title) # 접속한 웹 사이트의 제목 출력
print("- 접속한 웹 사이트의 URL:", driver.current_url) # 접속한 웹 사이트의 URL 출력


# In[ ]:


driver = Chrome()

base_url = "https://www.youtube.com"
search_word = '/results?search_query=' + '방탄소년단'
search_option = "&sp=CAMSAhAB" # 조회수로 정렬

url = base_url +  search_word + search_option # 접속하고자 하는 웹 사이트
driver.get(url)
time.sleep(3) # 웹 브라우저를 실행하고 URL에 접속할 때까지 기다림


# [7장: 326페이지]

# In[ ]:


html = driver.page_source # 접속 후에 해당 page의 HTML 코드를 가져옴
# driver.quit() # 웹 브라우저를 종료함

soup = BeautifulSoup(html, 'lxml')

# 태그(a)의 id(video-title)인 요소
# <a id="video-title" title="BTS (방탄소년단) 'Dynamite' Official MV"> ... </a>
title_hrefs = soup.select('a#video-title')

title_hrefs[0] # 첫 번째 항목 출력

#%%
print(type(title_hrefs)) # <class 'bs4.element.ResultSet'>
print(len(title_hrefs))  # 20개


# In[ ]:


title_hrefs[0].get('title') # title_hrefs[0]['title'] 도 동일


# In[ ]:


title_hrefs[0]['href'] # title_hrefs[0].get('href')도 동일


# [7장: 327페이지]

# In[ ]:


base_url = "https://www.youtube.com"
titles = []
urls = []
for title_href in title_hrefs[0:5]:
    title = title_href['title']         # 태그 안에서 title 속성의 값을 가져오기
    url = base_url + title_href['href'] # href 속성의 값 가져와 기본 url과 합치기
    titles.append(title)
    urls.append(url)
    print("{0}, {1}".format(title, url))


# [7장: 328페이지]

# In[ ]:


view_uploads = soup.select('span.style-scope.ytd-video-meta-block')

view_uploads[0:6]


# [7장: 329페이지]

# In[ ]:


view_numbers = view_uploads[0::2] # 인덱스가 짝수인 요소 선택
upload_times = view_uploads[1::2] # 인덱스가 홀수인 요소 선택

[view_numbers[0:3], upload_times[0:3]]
