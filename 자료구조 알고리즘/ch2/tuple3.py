# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:37:32 2024

@author: leehj
"""

# 튜플의 모든 원소를 ​​enumerate 함수로 스캔하기(1부터 카운트)

x = ('John', 'George', 'Paul', 'Ringo')

for i, name in enumerate(x, 1):
    print(f'{i} 번째 = {name}')