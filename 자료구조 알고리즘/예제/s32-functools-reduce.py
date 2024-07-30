# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:50:58 2024

@author: leehj
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:08:35 2024

@author: Solero
"""

# 점프 투 파이썬 - 라이브러리 예제 편
# 032 함수를 적용하여 하나의 값으로 줄이려면?

# functools.reduce(function, iterable)
# function을 반복 가능한 객체의 요소에 차례대로(왼쪽에서 오른쪽으로)
# 누적 적용하여 이 객체를 하나의 값으로 줄이는 함수

#%%

def add(data):
    result = 0
    for i in data:
        result += i
    return result

print(add([1,2,3,4,5])) # 15

#%%

# reduce()를 이용하여 누적합을 구하기
import functools

data = [1,2,3,4,5]

#%%

# 결과: ((((1+2)+3)+4)+5)
result = functools.reduce(lambda x, y: x + y, data)
print('result:', result)

#%%

# 5!
# 결과: ((((1*2)*3)*4)*5)
print(functools.reduce(lambda x, y: x * y, data)) # 120

#%%

# reduce()를 이용하여 최대값 구하기

num_list = [3,2,8,1,6,7]
max_num = functools.reduce(lambda x, y: x if x > y else y, num_list)
print(max_num) # 8

#%%

for x in iter(num_list):
    print(x, end=', ')

#%%
'''
[Lib/functools.py]
def reduce(function, sequence, initial=_initial_missing):
    """
    reduce(function, iterable[, initial]) -> value

    Apply a function of two arguments cumulatively to the items of a sequence
    or iterable, from left to right, so as to reduce the iterable to a single
    value.  For example, reduce(lambda x, y: x+y, [1, 2, 3, 4, 5]) calculates
    ((((1+2)+3)+4)+5).  If initial is present, it is placed before the items
    of the iterable in the calculation, and serves as a default when the
    iterable is empty.
    """

    it = iter(sequence)

    if initial is _initial_missing:
        try:
            value = next(it)
        except StopIteration:
            raise TypeError(
                "reduce() of empty iterable with no initial value") from None
    else:
        value = initial

    for element in it:
        value = function(value, element)

    return value

try:
    from _functools import reduce
except ImportError:
    pass
'''
