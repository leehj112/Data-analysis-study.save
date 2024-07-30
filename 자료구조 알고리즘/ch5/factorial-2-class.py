# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:02:45 2024

@author: leehj
"""

# [Do it! 실습 5-1] 양의 정수인 팩토리얼 구하기
# 팩토리얼 클래스

class Factorial:
    def __init__(self, n:int):
        self.__n = n # 속성: 계산할 팩토리얼 값
    
    # 내부 메서드: 비공개
    def __factorial(self, x: int) -> int:
        if x <= 1:
            return 1
        
        return x * self.__factorial(x - 1)
    
    # 공개 메서드
    def compute(self):
        return self.__factorial(self.__n)

n = 5
factobj = Factorial(n)

result = factobj.compute()

# 객체 밖에서 객체의 내부 속성에 접근할 수 없다.
# AttributeError: 'Factorial' object has no attribute '__n'
# print(f'{n},{factobj.__n}의 팩토리얼은 {result}입니다.')

print(f'{n}의 팩토리얼은 {result}입니다.')

