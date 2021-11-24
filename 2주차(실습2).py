####### 실습형 2부분 1번
# https://cafe.naver.com/yjbooks?iframe_url_utf8=%2FArticleRead.nhn%253Fclubid%3D19039057%2526articleid%3D16030
# 2018년도 성인의 건강검진 데이터로부터 흡연상태 예측 (흡연상태 1- 흡연, 0-비흡연 )
# 문제타입 : 분류유형
# 평가지표 : f1-score
import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/smoke/train.csv')

data.info()
data[['치석','구강검진수검여부','성별코드']]

from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
lable_one = LabelEncoder()
data['치석'] = lable_one.fit_transform(data['치석'])
data['구강검진수검여부'] = lable_one.fit_transform(data['구강검진수검여부'])
data['성별코드'] = lable_one.fit_transform(data['성별코드'])

data_y = data['흡연상태'].copy()
data.drop(['흡연상태'], inplace = True, axis = 1)
data_x = data.copy()

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(data_x,data_y, test_size = 0.2, stratify = data_y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

from sklearn.ensemble import RandomForestClassifier
r_model = RandomForestClassifier(random_state = 1004)
r_model.fit(train_x,train_y)
r_model.score(test_x,test_y) # 0.7586129502861632
y_hat = r_model.predict(test_x)
f1_score(y_hat, test_y) # 0.6800535475234271

# 아래 SVC fit 시간 너무 오래 걸림
# from sklearn.svm import SVC
# s_model = SVC(gamma = 'auto')
# s_model.fit(train_x,train_y)
# s_model.score(test_x,test_y)

from sklearn.linear_model import LogisticRegression
l_model = LogisticRegression(penalty = 'l2',random_state = 1004)
l_model.fit(train_x, train_y)
l_model.score(test_x, test_y) # 0.7451464482100775
y_hat = l_model.predict(test_x)
f1_score(y_hat, test_y) # 0.6639040994524198

from sklearn.neighbors import KNeighborsClassifier
k_model = KNeighborsClassifier()
k_model.fit(train_x, train_y)
k_model.score(test_x,test_y) # 0.7118168555717652
y_hat = k_model.predict(test_x)
f1_score(y_hat, test_y) # 0.609726443768997


####### 실습형 2부분 2번
# https://cafe.naver.com/yjbooks?iframe_url_utf8=%2FArticleRead.nhn%253Fclubid%3D19039057%2526articleid%3D16030
# 자동차 보험 가입 예측
# 문제타입 : 분류유형
# 평가지표 : f1-score
import pandas as pd
from sklearn.metrics import f1_score
data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/insurance/train.csv')

data.info()
data['Vehicle_Age'].unique()
data['Vehicle_Damage'].unique()
data['Gender'].unique()

from sklearn.preprocessing import LabelEncoder
la_en = LabelEncoder()
data['Vehicle_Age'] = la_en.fit_transform(data['Vehicle_Age'])
data['Vehicle_Damage'] = la_en.fit_transform(data['Vehicle_Damage'])
data['Gender'] = la_en.fit_transform(data['Gender'])

data_y = data.iloc[:,-1].copy()
data_x = data.iloc[:,1:-1].copy()

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, train_size = 0.8, stratify = data_y)

from sklearn.ensemble import RandomForestClassifier
r_model = RandomForestClassifier(random_state = 1004)
r_model.fit(train_x,train_y)
r_model.score(test_x,test_y) # 0.8666568270523796
y_hat = r_model.predict(test_x)
f1_score(y_hat, test_y) # 0.17777328344625343

from sklearn.neighbors import KNeighborsClassifier
k_model = KNeighborsClassifier()
k_model.fit(train_x, train_y)
k_model.score(test_x,test_y) # 0.8580143658368592
y_hat = k_model.predict(test_x)
f1_score(y_hat, test_y) # 0.07618437900128042

# 아래 SVC fit 시간 너무 오해걸림
# from sklearn.svm import SVC
# s_model = SVC(gamma = 'auto')
# s_model.fit(train_x, train_y)
# s_model.score(test_x, test_y)

from sklearn.linear_model import LogisticRegression
l_model = LogisticRegression(penalty = 'l2',random_state = 1004)
l_model.fit(train_x, train_y)
l_model.score(test_x, test_y) # 0.8774312046967759
y_hat = l_model.predict(test_x)
f1_score(y_hat, test_y) # 0
