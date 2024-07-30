# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 09:50:37 2024

@author: leehj
"""

print('1부터 n까지의 합을 구합니다.')
n = int(input('n값을 입력하세요.: '))

sum = 0
for i in range(1, n + 1):
    sum += i  # sum에 i를 더함

print(f'1부터 {n}까지 정수의 합은 {sum}입니다.')