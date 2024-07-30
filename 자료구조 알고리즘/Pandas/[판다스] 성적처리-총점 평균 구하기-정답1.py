# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:55:00 2024

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
print(df, '\n')

#%%
# '이름'을 인덱스로 지정
ndf = df.set_index('이름')
print(ndf)
# sr = ndf.loc[:,'수학':'체육'] # 전체 행, 전체 칼럼 새로운 데이터프레임으로 만들어서 확인
sr = ndf.loc[:,:] # 전체 행, 전체 칼럼 새로운 데이터프레임으로 만들어서 확인
print(sr)

#%%
#  학생별 총점
cnt = len(ndf.columns) 
print("과목건수: ", cnt)

# 칼럼 추가
ndf['총점'] = 0
ndf['평균'] = 0
print(ndf)

#%%
print("# 학생별 총점 및 평균 #")
for x in range(len(ndf)): # 행의 갯수
    rows = ndf.iloc[x, :] # 하나의 행을 추출
    tot = 0
    for val in rows: # 칼럼 1개씩 탐색하면 누적
        tot += val
    ndf.iloc[x,4] = tot        # 총점
    ndf.iloc[x,5] = tot // cnt # 평균

print(ndf)    


