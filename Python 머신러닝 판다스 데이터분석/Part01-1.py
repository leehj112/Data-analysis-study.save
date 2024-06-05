# pandas 
import pandas as pd

dict_data = {'a':1,'b':2,'c':3}

sr = pd.Series(dict_data) 


print(type(sr))
print('\n')
print(sr)
"""
a    1
b    2
c    3
"""
#%%
# 시리즈 인덱스
import pandas as pd 

list_data = ['2019-01-02', 3.14,'ABC',100,True]
sr = pd.Series(list_data) # ==> 리스트를 시리즈로 변환하여 변수 sr에 저장 
print(sr) 

"""
0    2019-01-02
1          3.14
2           ABC
3           100
4          True
dtype: objec
"""

#%% 

# 인덱스 배열은 변수 idx에 저장. 데이터 값 배열은 변수 val에 저장 
idx = sr.index
val = sr.values 
print(idx)
print('\n')
print(val) 

"""
['2019-01-02' 3.14 'ABC' 100 True]
"""

#%% 
# 원소 선택 
# 시리즈 원소 선택 

import pandas as pd

# 튜플을 시리즈로 변환
tup_data = ('영인','2015-05-01','여',True)
sr = pd.Series(tup_data, index=['이름','생년월일','성별','학생여부']) 
print(sr) 
print(sr[0]) # 원소 1개 선택 
print(sr['이름']) 
"""
이름              영인
생년월일    2015-05-01
성별               여
학생여부          True
dtype: object
영인
영인
"""
#%%
# 데이터 프레임: 2차원 배열 
# 행 index==> 인덱스 값 0~m in 시리즈 값 포함 
# 딕셔너리 --> 데이터프레임 변환
import pandas as pd

# 열이름key로 하고, 리스트value로 갖는 딕셔너리 정의(2차원 배열) 
dict_data = {'c0':[1,2,3],'c1':[4,5,6],'c2':[7,8,9],'c3':[10,11,12],'c4':[13,14,15]}

# 판다스 DataFrame() 함수 딕셔너리를 데이터프레임에 변환 df에 저장
df = pd.DataFrame(dict_data)

# df의 자료형 출력 
print(type(df)) 
print('\n')
print(df) 
"""
<class 'pandas.core.frame.DataFrame'>


   c0  c1  c2  c3  c4
0   1   4   7  10  13
1   2   5   8  11  14
2   3   6   9  12  15
"""
#%% 
# 행 인덱스/ 열 이름 설정 
import pandas as pd 

df = pd.DataFrame([[15,'남','덕영중'],[17,'여','수리중']],
                  index=['준서','예은'],
                  columns=['나이','성별','학교']) 

# 행 인덱스, 열 이름 확인
print(df) 
print('\n') 
print('\n')
print(df.columns) 
"""
나이 성별   학교
준서  15  남  덕영중
예은  17  여  수리중




Index(['나이', '성별', '학교'], dtype='object')
"""
#%%
df.index = ['학생1','학생2']
df.columns=['연령','남녀','소속']

print(df) 
print('\n') 
print(df.index)
print('\n')
print(df.columns) 

"""
연령 남녀   소속
학생1  15  남  덕영중
학생2  17  여  수리중


Index(['학생1', '학생2'], dtype='object')


Index(['연령', '남녀', '소속'], dtype='object')
"""


#%%
