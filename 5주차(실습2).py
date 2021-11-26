####### 실습형 2부분 1번
# 의료비용 예측문제
# 문제타입 : 예측
# 평가지표 : r2 score
from sklearn.metrics import r2_score
import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/MedicalCost/train.csv')
data.info()
from sklearn.preprocessing import LabelEncoder
l_encoder = LabelEncoder()
data['smoker'] = l_encoder.fit_transform(data['smoker'])
data['region'] = l_encoder.fit_transform(data['region'])
data['sex'] = l_encoder.fit_transform(data['sex'])
data.info()

data_x = data.iloc[:,:-1].copy()
data_y = data.iloc[:,-1].copy()

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, train_size = 0.8)

from sklearn.ensemble import RandomForestRegressor
r_model = RandomForestRegressor()
r_model.fit(train_x, train_y)
r_model.score(test_x, test_y) # 0.7152311871759458
y_hat = r_model.predict(test_x)
r2_score(y_hat, test_y) # 0.6725448894127494

from sklearn.svm import SVR
s_model = SVR(gamma= 'auto')
s_model.fit(train_x, train_y)
s_model.score(test_x, test_y) # -0.09723558253417064
y_hat = s_model.predict(test_x)
r2_score(y_hat, test_y) # -5074674.673158149

from sklearn.linear_model import Lasso
lasso_model = Lasso(alpha = 0.5, normalize = False) # noramlize True False 별차이 없음
lasso_model.fit(train_x, train_y)
lasso_model.score(test_x, test_y) # 0.767786583839305
y_hat = lasso_model.predict(test_x)
r2_score(y_hat, test_y) # 0.7277584923416492

from sklearn.linear_model import Ridge
ridge_model = Ridge(alpha = 0.5, normalize = False) # normalize True시 결과 더 안좋아짐
ridge_model.fit(train_x, train_y)
ridge_model.score(test_x, test_y) # 0.7682932528269812
y_hat = ridge_model.predict(test_x)
r2_score(y_hat, test_y) # 0.7252421906637556

from sklearn.linear_model import LinearRegression
l_model = LinearRegression()
l_model.fit(train_x, train_y)
l_model.score(test_x, test_y) # 0.632180877641702
y_hat = l_model.predict(test_x)
r2_score(y_hat, test_y) # 0.5514294161545015

## 성형회귀모델 계수확인
values = train_x.columns
coef = pd.Series(l_model.coef_, values).sort_values()
coef.plot(kind = 'bar')
plt.show()
### SMOKER 기준 2개 모델 분리 후 예측(단순선형 회귀)
smoke = data.loc[data['smoker'] == 1, :].copy()
no_smoke = data.loc[data['smoker'] == 0, : ].copy()
smoke_y = smoke['charges'].copy()
smoke_x = smoke.iloc[:,:-1].copy()
no_smoke_y = no_smoke['charges'].copy()
no_smoke_x = no_smoke.iloc[:,:-1].copy()

smoke_train_x, smoke_test_x, smoke_train_y, smoke_test_y = train_test_split(smoke_x, smoke_y, train_size = 0.8)
no_smoke_train_x, no_smoke_test_x, no_smoke_train_y, no_smoke_test_y = train_test_split(no_smoke_x, no_smoke_y, train_size = 0.8)

smoke_l_model = LinearRegression()
smoke_l_model.fit(smoke_x, smoke_y)
no_smoke_l_model = LinearRegression()
no_smoke_l_model.fit(no_smoke_x, no_smoke_y)

smoke_l_model.score(smoke_test_x, smoke_test_y) # 0.7619042810664934
y_hat = smoke_l_model.predict(smoke_test_x)
r2_score(y_hat, smoke_test_y) # 0.7557605536131309

no_smoke_l_model.score(no_smoke_test_x, no_smoke_test_y) # 0.46715484668281926
y_hat = no_smoke_l_model.predict(no_smoke_test_x)
r2_score(y_hat, no_smoke_test_y) # -0.3904540193029651

temp_y = pd.concat([smoke_test_y,no_smoke_test_y], axis = 0).reset_index()
temp_y.drop(['index'], axis = 1, inplace = True)
y_hat_s = smoke_l_model.predict(smoke_test_x)
y_hat_ns = no_smoke_l_model.predict(no_smoke_test_x)
y_hats = np.concatenate((y_hat_s, y_hat_ns), axis = None)
r2_score(y_hats, temp_y) #0.8211670359910349




####### 실습형 2부분 2번
# 킹카운티 주거지 가격 예측문제
# 문제타입 : 예측
# 평가지표 : r2 score
from sklearn.metrics import r2_score
import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/kingcountyprice/train.csv')

data.info()
data.head(10)

data.columns
data_y = data['price'].copy()
data_x = data.iloc[:,3:].copy()
data_x.drop(['zipcode'], axis = 1, inplace = True)

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, train_size = 0.8)

from sklearn.ensemble import RandomForestRegressor
r_model = RandomForestRegressor()
r_model.fit(train_x, train_y)
r_model.score(test_x, test_y) # 0.8624690927484228
y_hat = r_model.predict(test_x)
r2_score(y_hat, test_y) # 0.8389813706467374
