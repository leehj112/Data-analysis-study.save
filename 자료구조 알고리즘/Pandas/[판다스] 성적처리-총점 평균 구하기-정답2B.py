# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:56:06 2024

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
    rows = ndf.iloc[x, 0:4]
    tot = rows.sum()
    avg = int(rows.mean()) # 평균: 수학, 영어, 음악, 체육
    ndf.iloc[x,4] = tot
    ndf.iloc[x,5] = avg

print(ndf)    

#%%

rowcnt = len(ndf)
ndf.loc['총점'] = 0
ndf.loc['평균'] = 0

print("# 과목별 총점 및 평균 #")
for x in range(cnt):
    cols = ndf.iloc[0:3,x]    
    tot = cols.sum()
    avg = int(cols.mean())
    ndf.iloc[rowcnt, x] = tot
    ndf.iloc[rowcnt+1, x] = avg
    
print(ndf)    
