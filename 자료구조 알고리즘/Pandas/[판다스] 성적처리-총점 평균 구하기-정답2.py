# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:55:30 2024

@author: leehj
"""

# -*- coding: utf-8 -*-
import pandas as pd

# DataFrame() 함수로 데이터프레임 변환. 변수 df에 저장 
exam_data = {'이름' : ['서준', '우현', '인아'],
             '수학' : [ 90, 80, 70],
             '영어' : [ 98, 89, 95],
             '음악' : [ 85, 95, 100],
             '체육' : [ 100, 90, 90]}
df = pd.DataFrame(exam_data)
print(df)
print('\n')

#%%

ndf = df.set_index('이름')
print(ndf)
sr = ndf.loc[:,'수학':'체육']
print(sr)

#%%
#  학생별 총점

cnt = len(ndf.columns)
print("과목건수: ", cnt)

ndf['총점'] = 0
ndf['평균'] = 0
print(ndf)

#%%
print("# 학생별 총점 및 평균 #")
for x in range(len(ndf)):
    rows = ndf.iloc[x, :]
    tot = 0
    for val in rows:
        tot += val
    ndf.iloc[x,4] = tot
    ndf.iloc[x,5] = tot // cnt

print(ndf)    

#%%

print("# 전체 행의 갯수") # 3건
print(len(ndf))         # 행수 갯수
print(ndf.shape)        # tuple(3,6) : 3 * 6
print(ndf.shape[0])     # 3 
print(len(ndf.index))   # 3 
print(len(ndf.iloc[:])) # 3

#%%

# 칼럼 및 칼럼의 갯수
print("칼럼: ", ndf.columns)
print("칼럼의 갯수: ", len(ndf.columns))

#%%
rowcnt = len(ndf) #  전체 행의 갯수 : 3건
ndf.loc['총점'] = 0
ndf.loc['평균'] = 0

print("# 과목별 총점 및 평균 #")
for x in range(cnt):
    cols = ndf.iloc[:,x]    # 각 칼럼의 과목별 전체 학생 데이터의 시리즈(Series)
    tot = 0
    for val in cols:
        tot += val
    ndf.iloc[rowcnt, x] = tot
    ndf.iloc[rowcnt+1, x] = tot // rowcnt
    
print(ndf)    

