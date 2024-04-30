# -*- coding: utf-8 -*-
"""(part2)Data-study1.py

Automatically generated by Colab.

Original file is located at
 
"""

# 구글 연동 ==> colab
# 구글 드라이브 마운트
from google.colab import drive
drive.mount('/gdrive/')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

drive_path = "/gdrive/My Drive/"

train = pd.read_csv(drive_path + "titanic/train.csv")
test = pd.read_csv(drive_path + "titanic/test.csv")
submission = pd.read_csv(drive_path + "titanic/submission.csv")

print(train.shape, test.shape, submission.shape)
# titanic 불러오기

train.head(3)
# train data check

test.head(2)
# test data check

submission.head()
# submission data check

train.info()

train.describe(include='all')
# train 데이터프레임 통계 정보



#%% 
# 결측값 확인
import missingno as msno
msno.bar(train, figsize=(10,5), color=(0.7,0.2,0.2))
plt.show()

msno.matrix(test, figsize=(10, 5), color=(0.7, 0.2, 0.2))
plt.show()

#%% 



# 데이터 결합
# TrainSplit 열을 추가, Train, Test를 값으로 지정
train['TrainSplit'] = 'Train'
test['TrainSplit'] = 'Test'
data = pd.concat([train, test], axis=0)
print(data.shape)

# 숫자형 피처 추출
data_num = data.loc[:, ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]

# 결측값 대체
data_num['Age'] = data_num['Age'].fillna(data_num['Age'].mean())
data_num['Fare'] = data_num['Fare'].fillna(data_num['Fare'].mode()[0])

# 학습용 데이터와 예측 대상인 테스트 데이터 구분
selected_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

X_train = data_num.loc[data['TrainSplit']=='Train', selected_features]
y_train = data_num.loc[data['TrainSplit']=='Train', 'Survived']

X_test = data_num.loc[data['TrainSplit']=='Test', selected_features]

print("Train 데이터셋 크기: ", X_train.shape, y_train.shape)
print("Test 데이터셋 크기: ", X_test.shape)


#%% 
# 훈련 - 검증 데이터 분할
from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val =  train_test_split(X_train, y_train, test_size=0.2,
                                             shuffle=True, random_state=20)

# 로지스틱 회귀 모델
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(X_tr, y_tr)
y_val_pred = lr_model.predict(X_val)

# 실제값과 예측값과 비교하여 혼동 행열을 계산
# Confusion Matrix
from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(y_val, y_val_pred), annot=True, cbar=False, square=True)
plt.show()

#%% 


# 평가 지표
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score
print("Accuracy: %.4f" % accuracy_score(y_val, y_val_pred))
print("Precision: %.4f" % precision_score(y_val, y_val_pred))
print("Recall: %.4f" % recall_score(y_val, y_val_pred))
print("F1: %.4f" % f1_score(y_val, y_val_pred))
print("AUC: %.4f" % roc_auc_score(y_val, y_val_pred))
# 최대값은 1이며 예측력이 좋은 모델일 수록 1에 가까운값을 갖는다.

#%% 


# test 데이터에 대한 예측값 정리
y_test_pred = lr_model.predict(X_test)


submission['Survived'] = y_test_pred.astype(int)

submission_filepath = drive_path + 'baseline_num_lr_submission_001.csv'
submission.to_csv(submission_filepath, index=False)
submission.head(5)


#%% 
# 피처 엔지니어링
train['Survived'].value_counts(dropna=False)
# 생존자 342명 check --> 1

#%% 

sns.countplot(x='Survived',data=data[data['TrainSplit']=='Train'])
plt.show()
# TrainSplit 열을 이용하여 train 데이터만 따로 추출하여 적용

#%% 

# Pclass: 객실 등급
sns.countplot(x='Pclass', hue='TrainSplit', data=data)
plt.show()

#%% 

sns.countplot(x='Pclass',hue='Survived',data=data[data['TrainSplit']=='Train'])
plt.show()

#%% 

# barplot 함수를 사용하여 등급별 객실 요금위 중간값 분포
sns.barplot(x='Pclass',y='Fare', hue = 'Survived',
            data=data[data['TrainSplit']=='Train'], estimator=np.median)
plt.show()

#%% 


# histplot 함수의 multriple옵션을 조정하는 방법
sns.histplot(x='Sex', hue='Survived', multiple='dodge',
             data=data[data['TrainSplit']=='Train'])
plt.show()

#%% 

sns.histplot(x='Sex', hue='Survived', multiple='stack',
            data=data[data['TrainSplit']=='Train'])
plt.show()

#%% 

data.loc[data['Sex']=='female','Sex'] = 0
data.loc[data['Sex']=='male','Sex'] = 1
data['Sex'] = data['Sex'].astype(int)

data['Sex'].value_counts(dropna=False)
# female 0, male 1

#%% 

# Name 열 문자열 데이터 확인
data['Name'].unique()

# Name 열을 선택 str속성을 적용하여 문자열을 직접 추출
# split 메소드를 적용하면
title_name =data['Name'].str.split(", ", expand=True)[1]
title_name

#%% 
title = title_name.str.split(".", expand=True)[0]
title.value_counts(dropna=False)

#%% 


# replace 함수를 타이틀이 들어 있는 시리즈 객체에 적용하면 리스트 안의 문자열 뒤에 나오는 문자열로 모두 바꾼다.
title = title.replace(['Ms'], 'Miss')
title = title.replace(['Mlle', 'the Countess', 'Lady', 'Don', 'Dona', 'Mme', 'Sir', 'Jonkheer'], 'Noble')
title = title.replace(['Col', 'Major', 'Capt'], 'Officer')
title = title.replace(['Dr', 'Rev'], 'Priest')
data['Title'] = np.array(title)
data['Title'].value_counts(dropna=False)


#%% 
#결측값 확인 및 대체
# Age 열의 결측값을 확인
# 같은 타이틀을 갖는 승객끼리 그룹을 나누고, 그룹별 승객 나이의 중간값으로 결측값을 대처, fillna 메소드 사용
for title in data['Title'].unique():
    # 결측값 개수 확인
    print("%s 결측값 개수: " % title, data.loc[data['Title']==title, 'Age'].isnull().sum())
    # 각 Title의 중앙값으로 대체
    age_med = data.loc[data['Title']==title, 'Age'].median()
    data.loc[data['Title']==title, 'Age'] = data.loc[data['Title']==title, 'Age'].fillna(age_med)

# 결측값 처리 여부 확인
print("\n")
print("Age 열의 결측값 개수: ", data['Age'].isnull().sum())

#%% 
# displot의 kind 옵셥을 hist로 지정
# hue에 속성에 따라 생존자 구분

sns.displot(x='Age',kind='hist', hue='Survived',
            data=data[data['TrainSplit']=='Train'])

plt.show()

# Age 분포
sns.displot(x='Age', kind='hist', hue='Survived',
            data=data[data['TrainSplit']=='Train'])
plt.show()

#%% 
# 형재자매/배우자 수와 승객 나이 및 생존율 관계 
sns.boxplot(x='SibSp', y='Age', hue='Survived',
            data=data[data['TrainSplit']=='Train'])
plt.show() 

#%% 

sns.boxplot(x='Parch',y='Age',hue='Survived',
            data=data[data['TrainSplit']=='Train'])
plt.show() 

#%% 
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

sns.barplot(x='FamilySize', y='Survived', hue='Pclass', estimator=np.mean,
            data=data[data['TrainSplit']=='Train']) 

plt.show() 
# 5인가구가 상대적으로 높은 결과를 추출 
#%% 
# 결측값 확인
data.loc[data['Fare'].isnull(), :]
 
"""
PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked	TrainSplit	Title	AgeBin	FamilySize
152	1044	NaN	3	Storey, Mr. Thomas	1	60.5	0	0	3701	NaN	NaN	S	Test	Mr	Senior	1
"""
#%% 
# 3등석 요금의 평균값을 가지고 결측값 대체 
p3_fare_mean = data.loc[data['Pclass']==3, 'Fare'].mean()
print(p3_fare_mean)
data['Fare']=data['Fare'].fillna(p3_fare_mean) 
data.loc[data['PassengerId']==1044,:'Fare']

# 13.302888700564973

#%% 
sns.displot(x='Fare', kind='kde', hue='Survived',
            data=data[data['TrainSplit']=='Train'])
plt.show() 
# displot, kind 옵션을 kde로 설정 --> 밀도함수 그래프 

#%% 
data.loc[data['Embarked'].isnull(), :]
# 탑승항구 결측값 2개 확인
"""
PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked	TrainSplit	Title	AgeBin	FamilySize
61	62	1.0	1	Icard, Miss. Amelie	0	38.0	0	0	113572	80.0	B28	NaN	Train	Miss	Adult2	1
829	830	1.0	1	Stone, Mrs. George Nelson (Martha Evelyn)	0	62.0	0	0	113572	80.0	B28	NaN	Train	Mrs	Senior	1
"""
#%% 
# 최빈값을 사용하여 결측값 처리
print("Embarked 열의 최빈값", data['Embarked'].mode()[0]) 
data['Embarked']= data['Embarked'].fillna(data['Embarked'].mode()[0]) 
data['Embarked'].value_counts(dropna=False) 
# 탑승항구가 많은 S값으로 결측값을 채움 --> 데이터 최빈값은 mode 함수로 구한다 
"""
Embarked 열의 최빈값 S
Embarked
S    916
C    270
Q    123
Name: count, dtype: int64
"""
#%% 
# 시본 catplot함수의 kind옵션을 'point'로 설정 --> 각 클래스별 데이터 중심과 분산을 비교 
# 결과 C항구의 탑승자 생존율이 높은 편 
sns.catplot(x='Embarked', y='Survived', kind='point',
            data=data[data['TrainSplit']=='Train'])
plt.show()

#%% 
# 고유값 확인 
# cabin 
data['Cabin'].unique() 
"""
array([nan, 'C85', 'C123', 'E46', 'G6', 'C103', 'D56', 'A6',
       'C23 C25 C27', 'B78', 'D33', 'B30', 'C52', 'B28', 'C83', 'F33',
       'F G73', 'E31', 'A5', 'D10 D12', 'D26', 'C110', 'B58 B60', 'E101',
       'F E69', 'D47', 'B86', 'F2', 'C2', 'E33', 'B19', 'A7', 'C49', 'F4',
       'A32', 'B4', 'B80', 'A31', 'D36', 'D15', 'C93', 'C78', 'D35',
       'C87', 'B77', 'E67', 'B94', 'C125', 'C99', 'C118', 'D7', 'A19',
       'B49', 'D', 'C22 C26', 'C106', 'C65', 'E36', 'C54',
       'B57 B59 B63 B66', 'C7', 'E34', 'C32', 'B18', 'C124', 'C91', 'E40',
       'T', 'C128', 'D37', 'B35', 'E50', 'C82', 'B96 B98', 'E10', 'E44',
       'A34', 'C104', 'C111', 'C92', 'E38', 'D21', 'E12', 'E63', 'A14',
       'B37', 'C30', 'D20', 'B79', 'E25', 'D46', 'B73', 'C95', 'B38',
       'B39', 'B22', 'C86', 'C70', 'A16', 'C101', 'C68', 'A10', 'E68',
       'B41', 'A20', 'D19', 'D50', 'D9', 'A23', 'B50', 'A26', 'D48',
       'E58', 'C126', 'B71', 'B51 B53 B55', 'D49', 'B5', 'B20', 'F G63',
       'C62 C64', 'E24', 'C90', 'C45', 'E8', 'B101', 'D45', 'C46', 'D30',
       'E121', 'D11', 'E77', 'F38', 'B3', 'D6', 'B82 B84', 'D17', 'A36',
       'B102', 'B69', 'E49', 'C47', 'D28', 'E17', 'A24', 'C50', 'B42',
       'C148', 'B45', 'B36', 'A21', 'D34', 'A9', 'C31', 'B61', 'C53',
       'D43', 'C130', 'C132', 'C55 C57', 'C116', 'F', 'A29', 'C6', 'C28',
       'C51', 'C97', 'D22', 'B10', 'E45', 'E52', 'A11', 'B11', 'C80',
       'C89', 'F E46', 'B26', 'F E57', 'A18', 'E60', 'E39 E41',
       'B52 B54 B56', 'C39', 'B24', 'D40', 'D38', 'C105'], dtype=object)
"""
#%% 
data['Cabin'].str.slice(0,1).value_counts(dropna=False) 
# st속성은 문자열 추출, slice함수로 문자열의 첫 글자만 추출 
# 결측값:1014개 
"""
Cabin
NaN    1014
C        94
B        65
D        46
E        41
A        22
F        21
G         5
T         1
Name: count, dtype: int64
"""
#%% 
# 특히 객실 구역 데이터가 없어서 결측값으로 분류된'U'의 경우 생존율 가장 낮게 나타남 
# catplot함수 옵셕에 bar 입력 -> 시각화 
# 객실 구역별 생존 그래프 
data['Cabin'] = data['Cabin'].str.slice(0, 1) 
data['Cabin'] = data['Cabin'].fillna('u') 

sns.catplot(x='Cabin', y='Survived', kind='bar',
            data=data[data['TrainSplit']=='Train']) 

plt.show()  
#%% 
# 고유값 확인
# Ticket: 탑승권 
data['Ticket'].value_counts(dropna=False)  
"""
Ticket
CA. 2343        11
CA 2144          8
1601             8
PC 17608         7
S.O.C. 14879     7
                ..
113792           1
36209            1
323592           1
315089           1
359309           1
Name: count, Length: 929, dtype: int64
"""
#%% 
# 문자열 정리-> 알파벳 추출
data['Ticket'] = data['Ticket'].str.replace(".","")
data['Ticket'] = data['Ticket'].str.strip().str.split(' ').str[0] 
data['Ticket'].value_counts(dropna=False) 

"""
Ticket
PC          92
CA          68
A/5         25
SOTON/OQ    24
W/C         15
            ..
350060       1
239854       1
4134         1
11771        1
359309       1
Name: count, Length: 745, dtype: int64
"""
#%% 
from sklearn.preprocessing import LabelEncoder
for col in ['Title', 'AgeBin']:
    encoder = LabelEncoder()
    data[col] = encoder.fit_transform(data[col])

data.loc[:, ['Title', 'AgeBin']].head()
"""
	Title	AgeBin
0	2	9
1	3	1
2	1	9
3	3	0
4	2	0
"""
# 문자열 데이터 --> 숫자형 데이터
# method: LabelEncoder 객체를 만들고, fit_transform 함수를 사용하여 각 열의 데이터에 적용 
# 각 열에 속하는 범주의 개수만큼 숫자 레이블로 변환 
# ex) 3개 범주 --> 0,1,2 같이 3개의 숫자를 사용하여 데이터를 바꿔준다. 
# ### 원핫인코딩
#%% 