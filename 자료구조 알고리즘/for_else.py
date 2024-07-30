# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 09:44:37 2024

@author: leehj
"""


import random

n = int(input('난수의 개수를 입력하세요.: '))

for _ in range(n):
    r = random.randint(10, 99) # 난수 10부터 99까지 정수을 발생
    print(r, end=' ')
    if r == 13:
        print('\n프로그램을 중단합니다.')
        break
else : # for문에서 break로 종료하지 않으면 처리한다.
    print('\n난수 생성을 종료합니다.')