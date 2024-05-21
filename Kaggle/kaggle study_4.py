# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:14:07 2024

@author: leehj
"""
## Consumer's Buying Behavior Analysing_91.6%
# 소비자의 구매형태 분석_91.6% 
import numpy as np
import pandas as pd 

import os 
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename)) 
    
    

import seaborn as sns 
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore') 


data = pd.read_csv('./data/social_ads.csv')


data
"""
 Age  EstimatedSalary  Purchased
0     19            19000          0
1     35            20000          0
2     26            43000          0
3     27            57000          0
4     19            76000          0
..   ...              ...        ...
395   46            41000          1
396   51            23000          1
397   50            20000          1
398   36            33000          0
399   49            36000          1
"""
#%%
data.info() 
"""
Data columns (total 3 columns):
 #   Column           Non-Null Count  Dtype
---  ------           --------------  -----
 0   Age              400 non-null    int64
 1   EstimatedSalary  400 non-null    int64
 2   Purchased        400 non-null    int64
dtypes: int64(3)
memory usage: 9.5 KB
"""
#%%
data.isnull().sum()

"""
Age                0
EstimatedSalary    0
Purchased          0
dtype: int64
"""
#%% 
plt.figure(figsize=(10,8))
sns.countplot(x='Age',data=data)  # 빈도 시각화 
plt.title('Age Distribution')
plt.show() 

#%% 
overall_distribution = "The overall distribution of ages appears" 
central_tendency_mean = data['Age'].mean()        # 평균 
central_tendency_median = data['Age'].median()    # 중앙값 
central_tendency_mode = data['Age'].mode()[0]     # 최빈값 
variability_range = data['Age'].max() - data['Age'].min()  # 변동성 
variability_std_dev = data['Age'].std()           # 표준편차 


print("Overall Distribution:", overall_distribution)
print("Mean:", central_tendency_mean)
print("Median:", central_tendency_median)
print("Mode:", central_tendency_mode)
print("Range:", variability_range)
print("Standard Deviation:", variability_std_dev)

"""
Mean: 37.655
Median: 37.0
Mode: 35
Range: 42
Standard Deviation: 10.48287659730792
"""
#%%
plt.figure(figsize=(10,6)) 
sns.histplot(data['EstimatedSalary'], bins=20, kde=True, color='green') 
plt.title(' Estimated Salary')
plt.xlabel('Estimated Salary')
plt.ylabel('Frequency')
plt.show()


#%% 

mean_salary = data['EstimatedSalary'].mean()      # 평균 
median_salary = data['EstimatedSalary'].median()  # 중앙값 
mode_salary = data['EstimatedSalary'].mode()[0]   # 최빈값 
std_dev_salary = data['EstimatedSalary'].std()    # 표춘편차 

# Print summary statistics
print("Mean Estimated Salary:", mean_salary)
print("Median Estimated Salary:", median_salary)
print("Mode Estimated Salary:", mode_salary)
print("Standard Deviation of Estimated Salary:", std_dev_salary)

# Describe insights
print("The estimated salary distribution appears to be...")
print("There is variability in estimated salaries, with a standard deviation of", std_dev_salary)
print("The most common estimated salary is around", mode_salary)
#%%
# boxplot   
sns.boxplot(x='Purchased', y='Age',data=data)
plt.title('Purchased v/s Age')
plt.show()

#%% 
sns.boxplot(x='Purchased',y='EstimatedSalary',data=data)
plt.title('Purchased v/s EstimatedSalary')
plt.show()

#%%
# 산점도 
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='EstimatedSalary', hue='Purchased', data=data, )
plt.title(' Age vs. Estimated Salary  by Purchased')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.show()

#%% 
x = data.drop(columns='Purchased', axis=1)
y = data['Purchased'] 

x

"""
Age  EstimatedSalary
0     19            19000
1     35            20000
2     26            43000
3     27            57000
4     19            76000
..   ...              ...
395   46            41000
396   51            23000
397   50            20000
398   36            33000
399   49            36000
"""
#%% 
y 
"""
0      0
1      0
2      0
3      0
4      0
      ..
395    1
396    1
397    1
398    0
399    1
"""
#%% 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 

#%%
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42) 
rf_classifier.fit(x_train,y_train) 

#%%
y_pred = rf_classifier.predict(x_test) 

#%%
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy) 

"""
Accuracy: 0.9
"""
#%% 
report = classification_report(y_test, y_pred)
print(report) 

"""
            precision    recall  f1-score   support

           0       0.96      0.88      0.92        52
           1       0.81      0.93      0.87        28

    accuracy                           0.90        80
   macro avg       0.89      0.91      0.89        80
weighted avg       0.91      0.90      0.90        80
"""
#%% 
