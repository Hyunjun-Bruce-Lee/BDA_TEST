# 센서데이터로 동작 유형 분류
# 문제타입 : 분류
# 평가지표 : f1 score
data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/muscle/train.csv')
data.info()

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data['pose'] = encoder.fit_transform(data['pose'])

data_x = data.iloc[:,:-1]
data_y = data.iloc[:,-1]

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, train_size = 0.8, stratify = data_y)

from sklearn.ensemble import RandomForestClassifier
r_model = RandomForestClassifier()
r_model.fit(train_x, train_y)
r_model.score(test_x, test_y) # 0.9946236559139785
y_hat = r_model.predict(test_x)
f1_score(test_y, y_hat) # 0.9945828819068255

from sklearn.neighbors import KNeighborsClassifier 
k_model = KNeighborsClassifier()
k_model.fit(train_x, train_y)
k_model.score(test_x, test_y) # 0.9440860215053763
f1_score(test_y, y_hat) # 0.9945828819068255

from sklearn.svm import SVC
s_model = SVC(gamma = 'auto')
s_model.fit(train_x, train_y)
s_model.score(test_x, test_y) # 0.5010752688172043
f1_score(test_y, y_hat) # 0.9945828819068255

from sklearn.linear_model import LogisticRegression
l_model = LogisticRegression(max_iter = 10000)
l_model.fit(train_x, train_y)
l_model.score(test_x, test_y) # 0.5548387096774193
f1_score(test_y, y_hat) # 0.9945828819068255


####### 실습형 2부분 2번
# 현대차 스펙에 따른 가격 예측문제
# 문제타입 : 예측
# 평가지표 : r2 score
from sklearn.metrics import r2_score
data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/hyundai/train.csv')
data.info()
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data['transmission'] = encoder.fit_transform(data['transmission'])
data['fuelType'] = encoder.fit_transform(data['fuelType'])
data['model'] = encoder.fit_transform(data['model'])

data_y = data['price'].copy()
data_x = data.drop('price', axis = 1)

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, train_size = 0.8)

from sklearn.ensemble import RandomForestRegressor
r_model = RandomForestRegressor()
r_model.fit(train_x, train_y)
r_model.score(test_x, test_y) # 0.9528431595851863
y_hat = r_model.predict(test_x)
r2_score(test_y, y_hat) # 0.9528431595851863

from sklearn.linear_model import LinearRegression
l_model = LinearRegression()
l_model.fit(train_x, train_y)
l_model.score(test_x, test_y) # 0.7454598933692882
y_hat = l_model.predict(test_x)
r2_score(test_y, y_hat) # 0.7454598933692882

from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(train_x, train_y)
lasso.score(test_x, test_y) # 0.7454532495093154
y_hat = lasso.predict(test_x)
r2_score(test_y, y_hat) # 0.7454532495093154

from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(train_x, train_y)
ridge.score(test_x, test_y) # 0.745412889035778

from sklearn.svm import SVR
svr = SVR(gamma = 'auto')
svr.fit(train_x, train_y)
svr.score(test_x,test_y) # -0.02304749988589183

