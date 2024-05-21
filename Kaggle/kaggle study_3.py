# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:32:21 2024

@author: leehj
"""

## salary prediction
#    급여     예측 



import numpy as np
import pandas as pd
import unicodedata

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import pickle

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

#%%
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

#%% 
data = pd.read_excel('./data/salary_dataset.xlsx') 

#%%

df = data.copy() 

#%%

df.columns

"""
Index(['level', 'position', 'tech_stack', 'experience', 'gender', 'company',
       'company_size', 'work_type', 'city', 'currency', 'salary',
       'raise_period'],
      dtype='object')
"""

#%% 
# missing value
# teach_stack 
df. isna().sum() 

#%%
df.dropna(subset=['teach_stack'], inplace=True) 

#%% 
# 중복행 찾기 
duplicate_rows = df[df.duplicated()]

all_columns_same_duplicates =  duplicate_rows[duplicate_rows.duplicated(keep=False)]

#%% 
df["position"] = df["position"].str.lower() 
df["position"].value_counts() 

"""
position
back-end developer                                    1486
full stack developer                                  1442
front-end developer                                    602
team / tech lead                                       395
mobile application developer (android)                 178
embedded software developer                            173
software development manager / engineering manager     165
mobile application developer (full stack)              161
mobile application developer (ios)                     142
software architect                                     140
devops engineer                                        112
data scientist                                         103
qa / automation                                         97
game developer                                          97
qa / manuel test                                        87
project manager                                         83
product owner                                           64
data analyst                                            63
product manager                                         61
cto                                                     53
database administrator (dba)                            42
business analyst                                        40
director of software development                        35
sap/abap developer & consultant                         31
data engineer                                           25
cyber security                                          13
ai engineer                                             12
support engineer                                        12
erp developer                                            9
bussines intelligence                                    7
system admin & engineer                                  7
machine learning engineer                                6
ui/ux designer                                           5
business analyst                                         5
cloud platform engineer                                  4
site reliability engineer                                4
rpa developer                                            4
computer vision engineer                                 3
technical analyst                                        3
blockchain developer                                     3
research & development                                   3
solution architect                                       3
crm developer                                            3
salesforce developer                                     3
agile coach                                              2
cyber security                                           1
Name: count, dtype: int64
"""
#%%
top_10_positions = df['position'].value_counts().head(10) 

#%% 
# position 
plt.figure(figsize=(6, 4))  
sns.countplot(data=df, x='position', order=top_10_positions.index, color='skyblue')
plt.title('Frequency of Top 10 Positions')
plt.xlabel('Position')
plt.ylabel('Frequency')
plt.xticks(rotation=90)  
plt.tight_layout()


plt.savefig('top_10_positions_frequency.png', bbox_inches='tight')  


plt.show()

#%%
# level 
df["level"].value_counts() 
"""
level
Senior    2989
Middle    1937
Junior    1063
Name: count, dtype: int64
"""
#%% 

mapping = {
    'Junior': 0,
    'Middle': 1,
    'Senior': 2,
}


df['level'] = df['level'].map(mapping)


#%%
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='level', order=df['level'].value_counts().index, color="skyblue")
plt.title('Frequency of Levels')
plt.xlabel('Level')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.xticks(ticks=[0, 1, 2], labels=['Junior', 'Middle', 'Senior'])
plt.tight_layout()

plt.savefig('frequency_of_levels.png')

plt.show()

#%%
df.tech_stack.value_counts().head()
"""
tech_stack
.Net                            820
Java                            556
Python                          206
.Net;JavaScript | Html | Css    176
C / C++                         125
Name: count, dtype: int64
"""
#%%
df["experience"].value_counts() 
"""
experience
1 - 3 Yıl          1474
3 - 5 Yıl          1190
5 - 7 Yıl           848
7 - 10 Yıl          779
15 Yıl ve üzeri     512
10 - 12 Yıl         475
12 - 14 Yıl         363
0 - 1 Yıl           348
Name: count, dtype: int64
"""


#%% 
mapping = {
    
  '0 - 1 Yıl': 0,
  '1 - 3 Yıl': 1,
  '3 - 5 Yıl': 2,
  '5 - 7 Yıl': 3,
  '7 - 10 Yıl': 4,
  '10 - 12 Yıl': 5,
  '12 - 14 Yıl': 6,
  '15 Yıl ve üzeri': 7   
}

df['experience'].value_counts() 
"""
experience
1 - 3 Yıl          1474
3 - 5 Yıl          1190
5 - 7 Yıl           848
7 - 10 Yıl          779
15 Yıl ve üzeri     512
10 - 12 Yıl         475
12 - 14 Yıl         363
0 - 1 Yıl           348
Name: count, dtype: int64
"""

#%%
mapping = {
  'Erkek': 0,
  'Kadın': 1,
    }

df['gender'].value_counts() 
"""
gender
Erkek    5288
Kadın     701
Name: count, dtype: int6
"""
#%% 
df["company_size"].value_counts() 

"""
company_size
250+              2940
21 - 50 Kişi       673
101 - 249 Kişi     668
51 - 100 Kişi      627
11 - 20 Kişi       501
6 - 10 Kişi        339
1 - 5 Kişi         241
Name: count, dtype: int6
"""
#%% 
mapping = {
    '1 - 5 Kişi': 0,
    '6 - 10 Kişi': 1,
    '11 - 20 Kişi': 2,
    '21 - 50 Kişi': 3,
    '51 - 100 Kişi': 4,
    '101 - 249 Kişi': 5,
    '250+': 6
}

df["company_size"] = df["company_size"].map(mapping)
df["company"] = df["company"].str.lower()

#%%
df["city"].value_counts().head(10)
"""
city
İstanbul      3175
Ankara        1050
İzmir          455
Kocaeli        199
Bursa          156
Antalya         95
* Almanya       69
Eskişehir       55
* Hollanda      44
Samsun          43
Name: count, dtype: int64
"""
#%% 

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='city', order=df['city'].value_counts().head(10).index, color='skyblue')
plt.title('Distribution of Cities - Countries')
plt.xlabel('City - Country')
plt.ylabel('Frequency')
plt.xticks(rotation=45)  
plt.tight_layout()

plt.savefig('city_distribution.png') 

plt.show()

#%%
df["salary"].nunique()

#%%
df["currency"].value_counts()
"""
₺ - Türk Lirası    5354
$ - Dolar           353
€ - Euro            232
£ - Sterlin          50
Name: count, dtype: int64
"""
#%% 

def convert_salary_range_to_numeric(salary_range):
    
    if '+' in salary_range:
        salary_range = salary_range.replace('+', '') 
        return round(float(salary_range.replace(',', '')), 3) 
    else:
        
        lower, upper = map(float, salary_range.split(' - '))
       
        return round((lower + upper) / 2, 3)


def format_salary(salary):
    return '{:,.3f}'.format(salary)


df['salary'] = df['salary'].apply(convert_salary_range_to_numeric)


currency_multipliers = {
    '₺ - Türk Lirası': 1,
    '$ - Dolar': 32,
    '€ - Euro': 35,
    '£ - Sterlin': 40
}

df['converted_salary'] = df.apply(lambda row: row['salary'] * currency_multipliers[row['currency']], axis=1)


df['converted_salary'] = df['converted_salary'].apply(format_salary)
#%% 

mapping = {
    '₺ - Türk Lirası': 0,
    '$ - Dolar': 1,
    '€ - Euro': 2,
    '£ - Sterlin': 3
}


df["currency"] = df["currency"].map(mapping)

df.drop('salary', axis=1, inplace=True)
#%% 
df.raise_period.value_counts()
"""
raise_period
2    3548
1    2220
4     129
3      92
Name: count, dtype: int64
"""
#%% 
# city 
df['converted_salary'] = pd.to_numeric(df['converted_salary'], errors='coerce')


df = df.dropna(subset=['converted_salary'])


city_salary_stats = df.groupby('city')['converted_salary'].mean()


city_salary_stats_sorted = city_salary_stats.sort_values(ascending=False)


top_10_cities = city_salary_stats_sorted.head(10)


plt.figure(figsize=(6, 4))
top_10_cities.plot(kind='bar', color='skyblue')
plt.title('Top 10 Cities with Highest Average Salary')
plt.xlabel('City - Country')
plt.ylabel('Average Salary')
plt.xticks(rotation=90)
plt.tight_layout()


plt.savefig('top_10_cities_salary.png') 

plt.show()
#%% 
#gender and salaryender 
average_salary_by_gender = df.groupby('gender')['converted_salary'].mean().reset_index()


average_salary_by_gender['gender'] = average_salary_by_gender['gender'].replace({0: 'Male', 1: 'Female'})


plt.figure(figsize=(6, 4))
sns.barplot(x='gender', y='converted_salary', data=average_salary_by_gender, color="skyblue")
plt.title('Average Salary by Gender')
plt.xlabel('Gender')
plt.ylabel('Average Salary')


plt.savefig('average_salary_by_gender.png') 

plt.show()

#%% 
# company and salary
company_salary_stats = df.groupby('company')['converted_salary'].mean()


company_salary_stats_sorted = company_salary_stats.sort_values(ascending=False)


top_10_company = company_salary_stats_sorted.head(10)

plt.figure(figsize=(6, 4))
top_10_company.plot(kind='bar', color='skyblue')
plt.title('Top 10 Fields with Highest Average Salary')
plt.xlabel('Fields')
plt.ylabel('Average Salary')
plt.xticks(rotation=90)  
plt.tight_layout()


plt.savefig('top_10_field_average_salary.png', bbox_inches='tight')

plt.show()

#%%

#Calculate z-score for 'converted_salary' column
z_scores_salary = np.abs((df['converted_salary'] - df['converted_salary'].mean()) / df['converted_salary'].std())


threshold_salary = 5

outliers_salary = df[z_scores_salary > threshold_salary]

#%% 
plt.figure(figsize=(6, 4))
plt.scatter(df.index, df['converted_salary'], label='Data Points')
plt.scatter(outliers_salary.index, outliers_salary['converted_salary'], color='skyblue', label='Outliers')
plt.title('Outlier Detection for Converted Salary using Z-score')
plt.xlabel('Index')
plt.ylabel('Converted Salary (Amount)')
plt.legend()
plt.grid(True)
plt.tight_layout()


plt.savefig('outlier_detection_scatter_plot.png')


plt.show()
