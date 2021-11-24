####### 실습형 2부분 1번
### https://cafe.naver.com/yjbooks?iframe_url_utf8=%2FArticleRead.nhn%253Fclubid%3D19039057%2526articleid%3D16030
### 고객의 신상정보 데이터를 통한 회사 서비스 이탈 예측 (종속변수 : Exited)
### 문제타입 : 분류유형
### f1-score

train_data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/churn/train.csv')
test_data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/churn/test.csv')
test_data.keys()

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

for i in train_data.keys():
    print(type(train_data[f'{i}'][0]))
    print(train_data[f'{i}'].describe())

train_data.keys()
data_x = train_data.iloc[:,3:-1].copy()
data_y = train_data.iloc[:,-1].copy()
data_x_sex = pd.get_dummies(data_x.Gender, drop_first= True)
data_x_nation = pd.get_dummies(data_x.Geography)
data_x.drop(['Gender'], inplace = True, axis = 1)
data_x.drop(['Geography'], inplace = True, axis = 1)
data_x['Gender'] = data_x_sex # 1 = Male
data_x = pd.concat([data_x,data_x_nation], axis = 1)
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size = 0.2, stratify = data_y)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(penalty = 'l2', random_state = 1004)
model.fit(train_x, train_y)
y_hat = model.predict(test_x) # 0.793125
model.score(test_x, test_y) 
f1_score(test_y, y_hat) # 0.12201591511936338


from sklearn.ensemble import RandomForestClassifier
r_model = RandomForestClassifier(random_state = 1004)
r_model.fit(train_x, train_y)
r_y_hat = r_model.predict(test_x) # 0.861875
r_model.score(test_x,test_y) 
f1_score(test_y, r_y_hat) # 0.5774378585086042


from sklearn.neighbors import KNeighborsClassifier
k_model = KNeighborsClassifier()
k_model.fit(train_x, train_y)
k_model.score(test_x, test_y) # 0.753125
y_hat = k_model.predict(test_x) 
f1_score(test_y, r_y_hat) # 0.5774378585086042

from sklearn.svm import SVC
s_model =SVC(gamma = 'auto')
s_model.fit(train_x, train_y)
s_model.score(test_x,test_y) # 0.79625
y_hat = s_model.predict(test_x)
f1_score(y_hat, test_y) # 0




####### 실습형 2부분 2번
### https://cafe.naver.com/yjbooks?iframe_url_utf8=%2FArticleRead.nhn%253Fclubid%3D19039057%2526articleid%3D16030
### 유방암 발생여부 예측 (종속변수 diagnosis : B(양성)  , M(악성))
### 문제타입 : 분류유형
### f1-score

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
train_data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/cancer/train.csv')

train_data.keys()
train_data.describe()
train_data.info()

train_data.diagnosis.unique()
data_x = train_data.iloc[:,2:-1]
data_y = train_data.iloc[:,1]
data_y = pd.get_dummies(data_y, drop_first = True)

train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size = 0.2, stratify = data_y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import numpy as np
r_model = RandomForestClassifier()
train_y = np.array(train_y).flatten()
r_model.fit(train_x, train_y)
r_model.score(test_x,test_y) # 0.945054945054945
y_hat = r_model.predict(test_x)
f1_score(y_hat, test_y) # 0.9206349206349206

from sklearn.neighbors import KNeighborsClassifier
k_model = KNeighborsClassifier()
k_model.fit(train_x, train_y)
k_model.score(test_x, test_y) # 0.9120879120879121
y_hat = k_model.predict(test_x)
f1_score(y_hat, test_y) # 0.8666666666666666

from sklearn.linear_model import LogisticRegression
l_model = LogisticRegression(max_iter = 100000)
l_model.fit(train_x, train_y)
l_model.score(test_x,test_y) # 0.978021978021978
y_hat = l_model.predict(test_x)
f1_score(y_hat, test_y) #0.9696969696969697

from sklearn.svm import SVC
s_model = SVC(gamma = 'auto')
s_model.fit(train_x, train_y)
s_model.score(test_x,test_y) # 0.9340659340659341
y_hat = s_model.predict(test_x)
f1_score(y_hat, test_y) # 0.9032258064516129
