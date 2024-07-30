# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:58:04 2024

@author: leehj
"""

# [Do it! 실습 5-1] 양의 정수인 팩토리얼 구하기
# 5! = 1 * 2 * 3 * 4 * 5

# 인자: 정수
# 리턴: 정수
def factorial(n: int) -> int:
    t = 1

    while(n > 1): # 5,4,3,2
        t *= n
        print("* {} = {}".format(n, t))
        n -= 1
    print("* {}".format(1))
    return t

if __name__ == '__main__':
    n = int(input('출력할 팩토리얼 값을 입력하세요.: '))
    print(f'{n}의 팩토리얼은 {factorial(n)}입니다.')