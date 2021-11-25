####### 실습형 2부분 1번
# 심장질환예측
# 문제타입 : 분류유형
# 평가지표 : f1-score
import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/heart/train.csv')
data.keys()
data_y = data['target'].copy()
data_x = data.iloc[:,:-1].copy()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
scaler = StandardScaler()
data_x = scaler.fit_transform(data_x)

train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, train_size = 0.7, stratify = data_y)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
r_model = RandomForestClassifier(random_state = 1004)
r_model.fit(train_x, train_y)
r_model.score(test_x,test_y) # 0.8082191780821918
y_hat = r_model.predict(test_x)
f1_score(y_hat, test_y) # 0.8292682926829269

from sklearn.neighbors import KNeighborsClassifier
k_model = KNeighborsClassifier()
k_model.fit(train_x, train_y)
k_model.score(test_x,test_y) # 0.6027397260273972
y_hat = k_model.predict(test_x)
f1_score(y_hat, test_y) # 0.6329113924050633

from sklearn.linear_model import LogisticRegression
r_model = LogisticRegression(max_iter = 100000)
r_model.fit(train_x, train_y)
r_model.score(test_x,test_y) # 0.8082191780821918
y_hat = r_model.predict(test_x)
f1_score(y_hat, test_y) # 0.825

from sklearn.svm import SVC
s_model = SVC(gamma = 'auto')
s_model.fit(train_x, train_y)
s_model.score(test_x,test_y) # 0.5342465753424658
y_hat = s_model.predict(test_x)
f1_score(y_hat, test_y) # 0.6964285714285714



####### 실습형 2부분 2번
# 핸드폰 가격예측 (price_range컬럼 0(저렴) ~3(매우비쌈) 범위 ) 
# 문제타입 : 분류유형
# 평가지표 : accuracy
import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/mobile/train.csv')

data.keys()
data.info()
data_y = data.iloc[:,-1].copy()
data_x = data.iloc[:,:-1].copy()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
scaler = StandardScaler()
data_x = scaler.fit_transform(data_x)

train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, train_size = 0.8, stratify = data_y)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
r_model = RandomForestClassifier(random_state = 1004)
r_model.fit(train_x, train_y)
r_model.score(test_x,test_y) # 0.8725
y_hat = r_model.predict(test_x)
accuracy_score(y_hat, test_y) # 0.8725

from sklearn.neighbors import KNeighborsClassifier
k_model = KNeighborsClassifier()
k_model.fit(train_x, train_y)
k_model.score(test_x,test_y) # 0.48
y_hat = k_model.predict(test_x)
accuracy_score(y_hat, test_y) # 0.48

from sklearn.linear_model import LogisticRegression
r_model = LogisticRegression(max_iter = 100000)
r_model.fit(train_x, train_y)
r_model.score(test_x,test_y) # 0.9525
y_hat = r_model.predict(test_x)
accuracy_score(y_hat, test_y) # 0.9525

from sklearn.svm import SVC
s_model = SVC(gamma = 'auto')
s_model.fit(train_x, train_y)
s_model.score(test_x,test_y) # 0.8625
y_hat = s_model.predict(test_x)
accuracy_score(y_hat, test_y) # 0.8625
