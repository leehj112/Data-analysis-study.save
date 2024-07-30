# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:56:39 2024

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

#%%
#  학생별 총점: 가로축
tot = ndf.sum(axis=1) # 총점
avg = ndf.mean(axis=1).round() # 평균
ndf['총점'] = tot
ndf['평균'] = avg
print(ndf)

#%%
# 과목별 총점: 세로축
tot = ndf.sum(axis=0) # 총점
avg = ndf.mean(axis=0).round() # 평균
ndf.loc['평균'] = tot
ndf.loc['총점'] = avg
print(ndf)
