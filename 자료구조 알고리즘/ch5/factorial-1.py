# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:58:16 2024

@author: leehj
"""

# [Do it! 실습 5-1] 양의 정수인 팩토리얼 구하기

def factorial(n: int) -> int:
    """양의 정수 n의 팩토리얼을 구하는 과정"""
    
    print("[factorial] n:{}".format(n))
    
    # n: 5,4,3,2
    if n <= 1: # n:1
        return 1
    
    return n * factorial(n - 1)

if __name__ == '__main__':
    n = int(input('출력할 팩토리얼 값을 입력하세요.: '))
    print(f'{n}의 팩토리얼은 {factorial(n)}입니다.')