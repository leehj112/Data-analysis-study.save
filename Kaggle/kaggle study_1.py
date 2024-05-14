# -*- coding: utf-8 -*-
"""
Created on Tue May 14 09:04:08 2024

@author: leehj
"""

## 과거 주가 테이터 시각화 
# 1980~2024 
import numpy as np 
import pandas as pd 


import os # os모듈: 지정된 디렉토리 밑에 있는 파일과 디렉토리를 검색하는 방법을 보여준다 
for dirname, _, filenames in os.walk('kaggle/input'): # 디렉토리 시작 --> 파일 경로 출력 
    for filename in filenames:
        print(os.path.join(dirname, filename)) # os.path.join() 함수는 경로를 올바르게 연결하여 플랫폼별로 호환되는 경로 
        



stock_data = pd.read_csv('./data/AAPL.csv')  
stock_data
"""
    Date        Open        High  ...       Close   Adj Close     Volume
0      1980-12-12    0.128348    0.128906  ...    0.128348    0.099192  469033600
1      1980-12-15    0.122210    0.122210  ...    0.121652    0.094017  175884800
2      1980-12-16    0.113281    0.113281  ...    0.112723    0.087117  105728000
3      1980-12-17    0.115513    0.116071  ...    0.115513    0.089273   86441600
4      1980-12-18    0.118862    0.119420  ...    0.118862    0.091861   73449600
          ...         ...         ...  ...         ...         ...        ...
10926  2024-04-17  169.610001  170.649994  ...  168.000000  168.000000   50901200
10927  2024-04-18  168.029999  168.639999  ...  167.039993  167.039993   43122900
10928  2024-04-19  166.210007  166.399994  ...  165.000000  165.000000   67772100
10929  2024-04-22  165.520004  167.259995  ...  165.839996  165.839996   48116400
10930  2024-04-23  165.350006  167.050003  ...  166.899994  166.899994   48917700
[10931 rows x 7 columns]
"""

import matplotlib.pyplot as plt 

stock_data.set_index('Date', inplace=False)

plt.figure(figsize=(10, 6)) 
plt.plot(stock_data.index, stock_data['Close'], color='blue', linestyle='-')
plt.xlabel('Date')
plt.ylabel('Close')
plt.grid(True)
plt.show()

#%%
##LSTM 주가 예측 
import numpy as np 
import pandas as pd 

import os 
for dirname, _, filenames in os.walk('/kaggle/input') :
    for filename in filenames:
        print(os.path.join(dirname, filename)) 
        
#%% 
import pandas as pd 

df = pd.read_csv('./data/AAPL.csv') 
df.set_index('Date', inplace=True, drop=False)
df = df[df.index >= '2020-01-01'].copy()
df 
"""
 Date        Open  ...   Adj Close     Volume
Date                                ...                       
2020-01-02  2020-01-02   74.059998  ...   73.059418  135480400
2020-01-03  2020-01-03   74.287498  ...   72.349129  146322800
2020-01-06  2020-01-06   73.447502  ...   72.925629  118387200
2020-01-07  2020-01-07   74.959999  ...   72.582642  108872000
2020-01-08  2020-01-08   74.290001  ...   73.750252  132079200
               ...         ...  ...         ...        ...
2024-04-17  2024-04-17  169.610001  ...  168.000000   50901200
2024-04-18  2024-04-18  168.029999  ...  167.039993   43122900
2024-04-19  2024-04-19  166.210007  ...  165.000000   67772100
2024-04-22  2024-04-22  165.520004  ...  165.839996   48116400
2024-04-23  2024-04-23  165.350006  ...  166.899994   48917700
"""
#%% 
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout 
import matplotlib.pyplot as plt 


# 시계열 데이터를 --> LSTM 신경망 모델에 적합한 형식으로 변환하는 코드 

scaler = MinMaxScaler(feature_range=(0,1)) # 정규화 --> 0, 1사이로 모델이 수렴하도록 help
data = scaler.fit_transform(df[['Close']])  

train_size = int(len(data) * 0.8)            # next: train_set, test_set 구분 
test_size = len(data) - train_size
train, test = data[0:train_size,:], data[train_size:len(data),:]

 # creat_dataset() 함수 입력 데이터를 LSTM모델에 적합한 형식으로 변환
def create_dataset(dataset, time_step=1):
    dataX, dataY = [],[]                     # 현재 시간 스탭과 이전 시간 스탭의 데이터를 사용하여 X,Y(입력, 출력)을 만든다.
    for i in range(len(dataset)-time_step-1):# add: time_step: 순차적인 데이터 포인트 
        a = dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

#%%

time_step = 100
neurons = 50
epochs= 10  # 가중치 10 

#%%
X_train, y_train = create_dataset(train, time_step)
X_test, y_test = create_dataset(test, time_step)


#%%
# Reshape input to be [samples, time steps, features] which is required for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

#%%
# LSTM 모델 생성 --> 여러 개의 LSTM층을 쌓는 형태 모델
model = Sequential([
    LSTM(units=neurons, input_shape=(time_step, 1), return_sequences=True),
    LSTM(units=neurons, return_sequences=True), 

    LSTM(units=neurons, return_sequences=True),

    LSTM(units=neurons),
    Dense(units=1)      
])
# compile() -> 모델 compile
# mean_squared_eroor: 평균 제곱 오차 손실 함수를 사용
# Adam 옵티마이저를 사용

model.compile(loss='mean_squared_error', optimizer='adam')



# Train the model
# fit -> 모델 학습 
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64, verbose=1)
#%%
train_predict = model.predict(X_train)
test_predict = model.predict(X_test) 

#%%
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
#%% 
"""
Epoch 1/10
12/12 ━━━━━━━━━━━━━━━━━━━━ 10s 181ms/step - loss: 0.1989 - val_loss: 0.0860
Epoch 2/10
12/12 ━━━━━━━━━━━━━━━━━━━━ 1s 109ms/step - loss: 0.0194 - val_loss: 0.0053
Epoch 3/10
12/12 ━━━━━━━━━━━━━━━━━━━━ 1s 108ms/step - loss: 0.0079 - val_loss: 0.0136
Epoch 4/10
12/12 ━━━━━━━━━━━━━━━━━━━━ 1s 108ms/step - loss: 0.0064 - val_loss: 0.0219
Epoch 5/10
12/12 ━━━━━━━━━━━━━━━━━━━━ 1s 107ms/step - loss: 0.0053 - val_loss: 0.0143
Epoch 6/10
12/12 ━━━━━━━━━━━━━━━━━━━━ 1s 108ms/step - loss: 0.0046 - val_loss: 0.0066
Epoch 7/10
12/12 ━━━━━━━━━━━━━━━━━━━━ 1s 107ms/step - loss: 0.0040 - val_loss: 0.0047
Epoch 8/10
12/12 ━━━━━━━━━━━━━━━━━━━━ 1s 108ms/step - loss: 0.0036 - val_loss: 0.0043
Epoch 9/10
12/12 ━━━━━━━━━━━━━━━━━━━━ 1s 107ms/step - loss: 0.0034 - val_loss: 0.0040
Epoch 10/10
12/12 ━━━━━━━━━━━━━━━━━━━━ 1s 108ms/step - loss: 0.0032 - val_loss: 0.0036
Traceback (most recent call last)
"""



#%%
# 예측 모델 시각화 
look_back = time_step
trainPredictPlot = np.empty_like(data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(data)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(data)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(data))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()





