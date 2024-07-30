# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:50:12 2024

@author: leehj
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:22:42 2024

@author: Solero
"""

# 점프 투 파이썬 - 라이브러리 예제 편
# 008 앞뒤에서 자료를 넣고 빼려면? P35
# - collections.deque()

# 데크(deque): 양방향 자료형
# 스택(stack)이나 큐(queue)처럼 쓸 수 있다.
# 장점:
#   - 리스트(list)에 비해 속도가 빠르다
#   - 쓰레드(thread) 환경에서 안전하다.    

from collections import deque

a = [1,2,3,4,5] # 리스트
q = deque(a)    # 리스트를 이용해서 데크를 생성

# 회전: 시계방향
q.rotate(2) 
# [1, 2, 3, 4, 5]
# [5, 1, 2, 3, 4]
# [4, 5, 1, 2, 3]

# 데크를 리스트로
result = list(q)
print(result) # [4, 5, 1, 2, 3]

# 시계 반대 방향: 원 위치로
q.rotate(-2) 
print(q) # deque([1, 2, 3, 4, 5])

#%% 

# 맨 마직막 요소 꺼내기
print(q.pop()) # 5

#%%

# 맨 처음 요소 꺼내기
print(q.popleft()) # 1