# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:45:39 2024

@author: leehj
"""

## Generalized Linear Model using statsmodels

#%% 
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
!pip install hvplot -q
import hvplot
import hvplot.pandas
from sklearn.model_selection import train_test_split
import numpy as np


df = pd.read_csv("/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")
pd.set_option('display.max_columns', 100) # Show all columns
df.head()

#%%
y_dummy = pd.get_dummies(df['Attrition'])
y = pd.DataFrame(y_dummy['Yes'])
y.head()

#%% 

pd.set_option('display.max_columns', 100) # Show all columns
df.describe()

#%% 
df.hvplot.hist(y='DistanceFromHome', by='Attrition', subplots=False, width=900, height=300, bins=30)
#%%
df.hvplot.hist(y='Education', by='Attrition', subplots=False, width=900, height=300)
#%%
df.hvplot.hist(y='EnvironmentSatisfaction', by='Attrition', subplots=False, width=900, height=300)
#%%
df.hvplot.hist(y='JobInvolvement', by='Attrition', subplots=False, width=900, height=300)
#%%
df.hvplot.hist(y='JobLevel', by='Attrition', subplots=False, width=900, height=300)
#%%
df.hvplot.hist(y='JobSatisfaction', by='Attrition', subplots=False, width=900, height=300)
#%%
df.hvplot.hist(y='MonthlyIncome', by='Attrition', subplots=False, width=900, height=300)
#%%
df.hvplot.hist(y='RelationshipSatisfaction', by='Attrition', subplots=False, width=900, height=300)
#%%
df.hvplot.hist(y='PercentSalaryHike', by='Attrition', subplots=False, width=900, height=300)
#%%
df.hvplot.hist(y='StockOptionLevel', by='Attrition', subplots=False, width=900, height=300)
#%%
X = df[['JobLevel', 'MonthlyIncome', 'StockOptionLevel', 'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']]
X.head()
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y)
#%%
data = pd.concat([X_train, y_train], axis=1)
#%%
model = smf.glm(formula=formula, data=data, family=family)
#%%
result = model.fit()
#%%
result.summary()
#%%
pred = result.predict(X_test)
#%%
x = np.arange(0,len(pred))
sns.scatterplot(x="JobLevel", y="Yes", data=data)
plt.plot(x, pred, color="red", lw=3
#%%
sns.scatterplot(x="StockOptionLevel", y="Yes", data=data)
plt.plot(x, pred, color="red", lw=3)
#%%
sns.scatterplot(x="TotalWorkingYears", y="Yes", data=data)
plt.plot(x, pred, color="red", lw=3)
#%%
sns.scatterplot(x="YearsAtCompany", y="Yes", data=data)
plt.plot(x, pred, color="red", lw=3)
#%%
sns.scatterplot(x="YearsSinceLastPromotion", y="Yes", data=data)
plt.plot(x, pred, color="red", lw=3)
#%%
sns.scatterplot(x="YearsWithCurrManager", y="Yes", data=data)
plt.plot(x, pred, color="red", lw=3)
#%%

               
