# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:20:50 2024

@author: leehj
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 22:40:40 2024

@author: SOLGITS
"""

# 시그모이드 함수(sigmoid function)
# 0~1(0~100%) 사이의 값
# z가 무한하게 큰 음수일 때 0에 가까워 짐
# z가 무한하게 큰 양수일 때 1에 가까워 짐
# 음성 : p가 0.5이하
# 양성 : p가 0.5보다 크면

import numpy as np
import matplotlib.pyplot as plt

z = np.arange(-10, 10, 0.1)
p = 1 / (1 + np.exp(-z)) # 지수함수 계산

plt.plot(z, p)
plt.xlabel('z')
plt.ylabel('p')
plt.show()

#%%

import math

print('자연상수: e=', round(math.e, 8)) # 2.71828183

x = np.array([-2, -1, 0, 1, 2])
result = np.exp(x)
print(result)  # [0.13533528 0.36787944 1.         2.71828183 7.3890561 ]

#%%

x = np.array([-2, -1, 0, 1, 2])
result = np.exp(-x)
print(result)  # [7.3890561  2.71828183 1.         0.36787944 0.13533528]

#%%

x = np.array([-2, -1, 0, 1, 2])
result = 1 / np.exp(-x)
print(result)  # [0.13533528 0.36787944 1.         2.71828183 7.3890561 ]

#%%

x = np.array([-2, -1, 0, 1, 2])
result = 1 / (1 + np.exp(-x))
print(result)  # [0.13533528 0.36787944 1.         2.71828183 7.3890561 ]