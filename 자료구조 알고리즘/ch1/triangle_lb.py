# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:14:22 2024

@author: leehj
"""

# [Do it! 실습 1-22] 왼쪽 아래가 직각인 이등변 삼각형으로 * 출력하기

print('왼쪽 아래가 직각인 이등변 삼각형을 출력합니다.')
n = int(input('짧은 변의 길이를 입력하세요.: '))

for i in range(n):          # 행 루프
    for j in range(i + 1):  # 열 루프
        print('*', end='')
    print()