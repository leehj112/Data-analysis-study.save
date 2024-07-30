# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:50:26 2024

@author: leehj
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:03:25 2024

@author: Solero
"""

# 점프 투 파이썬 - 라이브러리 예제 편
# 015 숫자에 이름을 붙여 사용하려면?
# enum

from enum import IntEnum

class Week(IntEnum):
    MON = 1
    TUE = 2
    WED = 3
    THU = 4
    FRI = 5
    SAT = 6
    SUN = 7
    
#%%

print(type(Week))  # <class 'enum.EnumType'>

print(Week.MON.name)  # MON
print(Week.MON.value) # 1

#%%

mon = Week.MON
print('mon:', mon) # 1

sun = Week.SUN
print('sun:', sun) # 7

#%%

for week in Week:
    print("{}:{}".format(week.name, week.value))
"""    
MON:1
TUE:2
WED:3
THU:4
FRI:5
SAT:6
SUN:7    
"""

#%%

import datetime

# 오늘 날짜 : 2024-03-18
today = datetime.date.today()
print('오늘날짜:', today)              # 2024-03-18
print('오늘요일:', today.isoweekday()) # 1:월요일, 2:화요일, ... 7:일요일

week = today.isoweekday()

if week == Week.MON:
    print('week: 월요일')
    
# Week.name 목록 -> 리스트 객체로 생성    
weeks = [w.name for w in Week]    
print(weeks) # ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']

weekno = today.weekday() # 0:월요일, 1:화요일, ... 6:일요일
print('오늘요일:', weeks[week-1])
print('오늘요일:', weeks[weekno])

