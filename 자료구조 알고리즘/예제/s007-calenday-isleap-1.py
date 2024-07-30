# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:49:43 2024

@author: leehj
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:23:53 2024

@author: Solero
"""

# 점프 투 파이썬 - 라이브러리 예제 편
# 007 2월이 29일인 해를 알려면? P33
# calendar.isleap()

# 윤년 구하기
#
# 윤년의 기준 : 2월 달이 29일 
# 1. 서력 기원 연수가 4로 나누어 떨어지는 해는 우선 윤년
# 2. 그중에 100으로 나누어 떨어지는 해는 평년으로 한다.
# 3. 400으로 나누어 떨어지는 해는 다시 윤년으로 정한다.
# 
# 2024년은 윤년

#
import calendar

year = 2024

# 윤년이면 True, 평년이면 False
isleap = calendar.isleap(year)
print(f"({year})년은 ({isleap})이다")
print("({0})년은 ({1})이다".format(year, "윤년" if isleap else "평년"))

#%%

# 윤년의 기준 : 2월 달이 29일 
# 1. 서력 기원 연수가 4로 나누어 떨어지는 해는 우선 윤년
# 2. 그중에 100으로 나누어 떨어지는 해는 평년으로 한다.
# 3. 400으로 나누어 떨어지는 해는 다시 윤년으로 정한다.
def is_leap_year(year):
    if year % 400 == 0:
        return True
    if year % 100 == 0:
        return False
    if year % 4 == 0:
        return True
    return False

isleap2 = is_leap_year(year)
print(f"({year})년은 ({isleap2})이다")
print("({0})년은 ({1})이다".format(year, "윤년" if isleap2 else "평년"))

#%%

def is_leap_year3(year):
    if year % 4 == 0: # 윤년
        if year % 100 == 0 and year % 400 != 0: # 평년
            return False
        return True # 윤년
    else: # 평년
        return False

for n in range(4, 2030, 4):
    isleap3 = is_leap_year3(n)
    print("({0})년은 ({1})이다".format(n, "윤년" if isleap3 else "<평년>"))

#%%

# if문을 한 문장으로 줄이면?
def is_leap_year4(year):
    if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0): # 윤년
        return True # 윤년
    else: # 평년
        return False

for n in range(4, 410, 4):
    isleap4 = is_leap_year4(n)
    print("({0})년은 ({1})이다".format(n, "윤년" if isleap4 else "<평년>"))
