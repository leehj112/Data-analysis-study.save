# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:50:44 2024

@author: leehj
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:36:06 2024

@author: Solero
"""

# 점프 투 파이썬 - 라이브러리 예제 편
# 019 소수점을 정확하게 계산하려면?

# decimal.Decimal()
# 숫자를 10진수로 처리하여 정확한 소수점 자릿수를 표현할 때 사용

a = 0.1 * 3
b = 1.2 - 0.1
c = 0.1 * 0.1

# 예상 결과
ax = 0.3
bx = 1.1
cx = 0.01

# 아래의 결과는 모두 : False
# 이진수 기반의 float 연산은 미세한 오차가 발생할 수 있다.
# IEEE 754 규약
print(f"({a}) == ({ax}) : {a == ax}")  # (0.30000000000000004) == (0.3) : False
print(f"({b}) == ({bx}) : {b == bx}")  # (1.0999999999999999) == (1.1) : False
print(f"({c}) == ({cx}) : {c == cx}")  # (0.010000000000000002) == (0.01) : False

#%%

from decimal import Decimal

# 소숫점을 Decimal로 변환하여 연산 후 float로 변환
ad = float(Decimal('0.1') * 3)
bd = float(Decimal('1.2') - Decimal('0.1'))
cd = float(Decimal('0.1') * Decimal('0.1'))
print(f"({ax}) == ({ad}) : {ax == ad}")  # (0.3) == (0.3) : True
print(f"({bx}) == ({bd}) : {bx == bd}")  # (1.1) == (1.1) : True
print(f"({cx}) == ({cd}) : {cx == cd}")  # (0.01) == (0.01) : True

#%%

av = 0.1
sv = str(av) # 실수를 문자열로 변환
print(av, type(av)) # 0.1 <class 'float'>
print(sv, type(sv)) # 0.1 <class 'str'>
sx = float(Decimal(sv) * 3)
print(sx) # 0.3

# float를 문자열로 변환하지 않고 그대로 Decimal에 전달하면
# float가 가진 문제가 그대로 발생한다.
tx = float(Decimal(av) * 3)
print(tx) # 0.30000000000000004


