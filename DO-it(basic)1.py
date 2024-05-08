# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# 정수를 입력받아 최대값 구하기 

print("세 정수의 최대값을 구한다:") 

a = int(input('정수 a값을 입력하세요:' ))
b = int(input('정수 b값을 입력하세요:'))
c = int(input('정수 c값:')) 

maximum = a   # maximum에 a에 값을 대입 

if b > maximum: maximum = b   # b의 값이 maximum보다 크면 maximum에 b에 값을 대입 
if c > maximum: maximum = c   # c의 값이 maximum보다 크면 maximum에 c에 값을 대입 

print(f'최대값은 {maximum}입니다.') 

#%% 

# 문자열과 숫자 입력 받기 
name = input('이름입력:', end='') 

print(f'안녕하세요? {name}님.') 

#%% 

# 세 정수를 입력받아 중앙값 구하기1
def med3(a,b,c) :
    if a>= b:
        if b >= c:
            return b 
        elif a <= c:
            return a 
        else: 
            return c 
    elif a > c:
        return a
    elif b > c:
        return c
    else: 
        return b

print('세 정수의 중앖을 구합나다.')

a = int(input('정수 a의 값:')) 
b = int(input('정수 b의 값'))
c = int(input('정수 c의 값:')) 

print(f'중앙값은 {med3(a, b, c)}입니다.') 


#%% 세 정수 입력받아 중앙값 구하기2

def med3(a, b, c): 
    # a, b, c 의 중앙값을 구하여 반환 
    if (b>=a and c <= a) or (b <= a and c >= a): 
        return a 
    elif (a > b and c < b) or (a < b and c > b): 
        return b 
    return c  


#%% 
# 입력받은 정수의 부호(양수, 음수, 0) 출력하기 

n = int(input('정수를 입력하세요:')) 

if n > 0:
    print('이 수는 양수') 
elif n < 0:
    print('이 수는 음수:') 

else:
    print('이 수는 0:') 

#%% 

# 3개로 분기하는 조건문 
n = int(input('정수를 입력하세요:')) 

if n == 1:
    print('A')
elif n == 2:
    print('B') 
else:
    print('C') 

# 4개로 분기하는 조건문

n = int(input('정수를 입력:')) 

if n == 1:
    print('A')
elif n == 2:
    print('B') 
elif n == 3:
    print('c') 
else: 
    pass # else 문의 pass 문이 실행 


#%% 
# 연산자나 피연산자 
# +-등의 기호를 산술 연산자 
# 연산 대상을 피연산자 
# 조건 연산자 : 참/거짓 

# 1부터 n까지 정수의 합 구하기  
print('1부터 n까지 정수의 합 구하기.') 
n = int(input('n값을 입력하세요.:')) 

sum = 0
i = 1 

while i <= n:  # i가 n보다 작거나 같은 동안 반복 
    sum += i   # sum i를 더함 
    i += 1     # i에 1을 더함 

print(f'1부터{n}까지 정수의 합은{sum}입니다.') 
#%% 
# for 문 반복 
# 1부터 n까지 정수의 합 구하기 2(for 문) 

print('1부터 n까지 정수의 합을 구합니다.') 
n = int(input('n값을 입력하세요:')) 

sum = 0
for i in range(1, n+1):
    sum += i              # sum에 i를 더함
    

print(f'1부터{n}까지 정수의 합은 {sum}입니다.') 

#%%
# a 부터 b까지 정수의 합 구하기(for문) 

print('a부터 b까지 정수의 합을 구한다.') 
a = int(input('정수 a입력:')) 
b = int(input('정수 b입력:')) 


if a > b:
    a, b = b, a 
    

sum = 0
for i in range(a, b+1): 
    sum += i 
    
print(f'{a}부터 {b}까지 정수의 합은{sum}입니다.')

#%% 
# a부터 b까지 정수의 합 구하기1

print('a부터 b까지 정수의 합.') 

a = int(input('정수 a입력:')) 
b = int(input('정수 b입력:')) 

if a > b :
    a, b = b, a
sum = 0
for i in range(a, b+1): 
    if i < b:
        print(f'{i} + ', end='') 
    else:
        print(f'{i} = ', end='') 
    sum += i 
print(sum) 

#%% 

# a부터 b까지 정수의 합 구하기2

print('a부터 b까지 정수의 합.') 
a = int(input('정수 a를 입력:')) 
b = int(input('정수 b를 입력:')) 

if a > b:
    a, b = b, a
sum = 0
for i in range(a, b):
    print(f'{i}+', end='')
    sum += i 

print(f'{b}=', end='')
sum += b


print(sum) 
#%%% 

# +/- 를 번갈아 출력
print('+ -를 번갈아 출력한다.')
n = int(input('몇 개출력:')) 

for i in range(n):
    if i % 2:
        print('-', end='')  # 홀수인 경우 -출력 
    else:
        print('+', end='')  # 짝수인 경우 +출력 
print() 

#%% 
# n이 홀수인 경우 출력 w개마다 줄바꿈하기 1

print('*를 출력합니다.') 
n = int(input('몇 개를 출력할까요)) 
w = int(input('몇 개마다 줄바꿈할까요:')) 

for i in range(n):
    print('*', end='')
    if i % w == w-1:
        print() 
        
if n % w:
    print() 
    

#%% 
# *를 n개로 출력하되 w개마다 줄바꿈하기2 

print('*를 출력합니다.') 
n = int(input('몇 개출력:')) 
w = int(input('몇 개줄바꿈:')) 

for _ in range(n // w): 
    print('*' *w)
rest = n % w
if rest:
    print('*' * rest) 
#%% 
# 1부터 n까지 정수의 합 구하기(n값은 양수만 입력) 

print('1부터 n가지 정수의 합을 구합니다.') 

while True:
    n = int(input('n값을 입력하세요.: ')) 
    if n > 0:
        break 
    
sum = 0
i = 1

for i in range(1, n+1): 
    sum += i
    i += 1 
print(f"1부터 {n}까지 정수의 합은 {sum}입니다.") 

#%%
area = int(input('직사각형의 넓이를 입력하세요:'))

for i in range(1, area + 1):
    if i * i > area: break
    if area % i: continue
    print(f'{i} * {area // i}') 
    
#%% 
# 10 ~ 99 사이의 난수 n 생성 
import random

n = int(input('난수의 개수를 입력:')) 

for _ in range(n):
    r = random.randint(10, 99) 
    print(r, end=' ') 
    if r == 13:
        print('\n프로그램을 중단.') 
        break 
else: 
    print('난수 생성을 종료.') 
    
#%%
# 1 ~ 12까지 8을 건너뛰고 출력1
for i in range(1, 13): 
    if i == 8:
        continue
    print(i, end= ' ') 
print() 
#%%

# 1부터 12까지 8을 건너뛰고 출력하기

for i in list(range(1,8)) + list(range(9, 13)):
    print(i, end=' ')
print() 

#%% 

# 2자리 양수 입력받기
print('2자리 양수를 입력:') 

while True:
    no = int(input('값을 입력:')) 
    if no >= 10 and no <= 99:
        break 
    
print(f'입력받은 양수는 {no}입니다.') 

#%%
# 구구단 곱셈표 출력

print('-' * 27)
for i in range(1, 10):
    for j in range(1, 10):
        print(f'{i*j:3}', end='') 
    print() 

print('-'*27) 

#%% 
# 직각 이등변 삼각형 출력
print('왼쪽 아래가 지각인 이등변 삼각형 출력.') 
n = int(input('잛은 변 길이 입력:')) 

for i in range(n):
    for j in range(i + 1):
        print('*', end='') 
    print() 
#%%
# 오른쪽 아래가 직각인 이등변 삼각형 출력
print('오른쪽 아래 출력:')
n = int(input('짧은 변의 길이 입력:'))

for i in range(n):
    for _ in range(n -i -1):
        print(' ', end = ' ') 
    for _ in range(i + 1): 
        print('*', end=' ') 
    print() 
#%%

# 함수 내부 외부에서 정의한 변수와 객체의 식별 번호를 출력하기 
n = 1           # 전역변수(함수 내부, 외부에서 사용) 
def put_id():
    x = 1       # 지역변수(함수 내부에서만 사용)
    print(f'id(x) = {id(x)}')
print(f'id(1) = {id(1)}') 
print(f'id(n) - {id(n)}') 
put_id() 



#%%
# 1부터 100까지 반복하여 출력하기 
for i in range(1, 101):
    print(f'i = {1,3} id(i) = {id(i)}') 
#%% 
# 자료구조와 배열 
# 배열: 흩어진 변수를 하나로 묶어서 사용할 수 있어 코드를 쉽고 효율적으로 작성할 수 있다. 

#리스트: 원소를 변경 가능 [], , 로 구성 
#튜플: 원소 변경 불가능 (), ,로 구성 
#인덱스: []안에 정수값 인덱스를 지정하는 인덱스식 리스트의 특정 원소를 정할 수 있다.

x = [11,22,33,44,55,66,77]
x[2] 
# 33
x[-3] 
# 55
x[-4]
#44


# 슬라이스 : 리스트 또는 튜플 원소 이불를 연속해서 또는 일정한 간격으로 꺼내 새로운 리스트 또는 튜플을 만드는것 
# 리스트 또는 튜플을 만드는 것은: 슬라이스 

# s[i:j]
# s[i:j:k] 

s = [11, 22, 33, 44, 55, 66, 77]

s[0:7:2] 
# 리스트 s의 0번째 원소부터 6번째 원소를 출력 

s[-4:-2] 
# 리스트 s의 뒤에서 4번째 원소부터 뒤에서 3번재 원소를 출력 

s[3:1] 
# 리스트 s의 j값(1)이 i값(3)보다 작지만 오류가 x 

#%% 
n = 5
id(n) 


n = 'ABC'
id(n) 


#%% 
x = 0
type(x + 17) 
# int 

type(x=17) 
# type() takes 1 or 3 arguments

# 자료구조: 데이터 단위와 데이터 자체 사이의 물리적 또는 논리적인 관계 

#%% 
x = [15,64,7,3.14,[32, 55],'ABC']
len(x) 

#%% 
# 배열 원소의 최대값을 구하는 함수 구현 
# 시퀀스 원소의 최대값 출력 
from typing import Any, Sequence 

def max_of(a: Sequence) -> Any:
    # 시퀀스형 a원소의 최대값을 반환
    maximum = a[0] 
    for i in range(1, len(a)):
        if a[i] > maximum:
            maximum = a[i] 
            
    return maximum 

if __name__=='__main__': 
    print('배열의 최대값을 구한다.') 
    num = int(input('원소 수 입력:')) 
    x = [None] * num # 원소 수가 num인 리스트를 생성 
    
    for i in range(num):
        x[i] = int(input(f'x[i]값을 입력하세요:')) 
        
    print(f'최대값은 {max_of(x)}입니다.')  
#%% 

# 리스트의 모든 원소를 스캔 

x = ['john', 'George', 'paul',' Ringo'] 

for i in range(len(x)):
    print(f'x[{i}] = {x[i]}') 
#%%
# 리스트의 모든 원소를 enumerate()함수로 스캔 
x = ['john', 'George', 'paul',' Ringo'] 

for i, name in enumerate(x): 
    print(f'x[{i}] = {name}')

#%% 
# 리스트의 모든 원소를 enumerate() 함수로 스캔(1부터 카운트) 
x = ['john', 'George', 'paul',' Ringo'] 

for i, name in enumerate(x, 1): 
    print(f'{i}번째 = {name}') 
    
#%%

# 리스트의 모든 원소를 스캔하기(인덱스 값을 사용하지 않음)
x = ['john', 'George', 'paul',' Ringo']


for i in x:
    print(i) 
    
    
#%% 
# 튜플의 스캔 
# 배열 원소를 역순으로 정렬하기 

from typing import Any, MutableSequence 

def reverse_array(a: MutableSequence) -> None: 
    # 뮤티블 시퀀스 a의 원소를 역순으로 정렬
    
    n = len(a) 
    for i in range(n // 2):
        a[i], a[n - i - 1] = a[n - i - 1], a[i] 

if __name__ == ' __main__': 
    print('배열 원소를 역순으로 정렬.') 
    nx = int(input('원소 수를 입력:')) 
    x = [None] * nx
    
    for i in range(nx):
        x[i] = int(input(f'x[i]값을 입력하세요:')) 
        
    reverse_array(x) 
    
    print('배열 원소를 역순 정렬') 

    for i in range(nx):
        print(f'x[{i}] = {x[i]}') 
        


