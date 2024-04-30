# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 12:45:06 2024

@author: leehj
"""
# 텐서플로: 고수준 API인 Keras를 중심으로 딥러닝 모델을 구축하고 훈련하는 방법 
# 케라스: 여러 개의 연결하여 신경망 모델을 구성하는 도구 
# 간단한 아키택처를 가지면서도 대부분의 딥러닝 모델을 만들 수 있다는 장점 

import tensorflow as tf
print(tf.__Version__) 

#%%
# y= x+1에 변수에 각각 10개 씩 입력 
# x 변수의 숫자 배열을 (10행, 1열) 형태의 2차원 배열로 변환 
import pandas as pd
import numpy as np 

x = [-3, 31, -11, 4, 0, 22, -2, -5, -25, -14]
y = [-2, 32, -10, 5, 1, 23, -1, -4, -24, -13]

x_train = np.array(x).reshape(-1,1) 
y_train = np.array(y) 

print(x_train.shape, y_train.shape) 
"""
(10, 1) (10,)
"""
#%% 
# sequential API: 레이어는 여러 개를 연결하여 신경망 모델을 구성하는 도구 
# 간단한 아키택처를 가지면서 딥러닝 모델을 만들 수 있음 
# 설명 변수(피처) 개수 지정 
# 1개에 설명면수 1로 설정 
# 완전 연결 레이어의 출력값은 목표 레이블(Y)을 예측 
# 유닛 갯수는 1 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense 

mode1 = Sequential() 
mode1.add(Dense(units=1, activation='linear', input_dim=1))    # lincar 옵션을 지정하여 선형 함수 출력을 그대로사용 

#%% 
# summary 메소드를 이용하여 모델 아키텍처 구조 확인 
# 딥러닝 모델이 학습할 파라미터는 2개 이다. --> # 일차함수의 기울기와 절편 

mode1.summary() 

"""
Model: "sequential_3"
┌─────────────────────────────────┬────────────────────────┬───────────────┐
│ Layer (type)                    │ Output Shape           │       Param # │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 1)              │             2 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 2 (8.00 B)
 Trainable params: 2 (8.00 B)
 Non-trainable params: 0 (0.00 B)
 """

#%% 
# 모델 컴파일 
# 모델이 훈련하는데 필요한 기본 설정을 compile 함수에 지정 
# 옵티마이저와 손실(loss) 함수를 설정 
# adam 옵티마이저를 선택하고 회귀 분석의 손실 함수인 평균 제곱오차를 지정 
# mettics 옵션에 평균 보조 평가 지표를 추가 
"""
total: 입력 --> y=wx + b 에 옵티마이저를 가중치(w, b)에 갱신 그리고 예측한 값 y_hat, y 실제값과 비교 
"""
mode1.compile(optimizer='adam', loss='mse', metrics=['mae']) 

#%% 
# fit 메소드의 훈련 데이터를 입력하여 모델 학습 
# epoch: 반복 횟수 지정
# verbose 옵션: False(0) 화면에 보여주지 않는다. -> 0, 1로 구분 
# 훈련 start 

mode1.fit(x_train, y_train, epochs=3000, verbose=0) 
# <keras.src.callbacks.history.History at 0x22877581150>
#%% 
# 가중치 확인 --> weights 속성 
# 학습 확인 
mode1.weights 

"""
[<KerasVariable shape=(1, 1), dtype=float32, path=sequential_3/dense_2/kernel>,
 <KerasVariable shape=(1,), dtype=float32, path=sequential_3/dense_2/bias>]
"""



