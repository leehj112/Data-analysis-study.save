# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:13:40 2024

@author: leehj
"""

# 1부터 n까지의 합 구하기 3(가우스 덧셈 방법)

print('1부터 n까지의 합을 구합니다.')
n = int(input('n값을 입력하세요.: '))

sum = n * (n + 1) // 2

print(f'1부터 {n}까지의 합은 {sum}입니다.')