
####### 실습형 2부분 1번
# 대학원 입학 가능성 예측문제 ("Chance of Admit')
# 문제타입 : 예측
# 평가지표 : r2 score
from sklearn.metrics import r2_score
import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/admission/train.csv')
data.info()

data_y = data['Chance of Admit'].copy()
data_x = data.drop(['Serial No.','Chance of Admit'], axis = 1)
data_x.info()

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, train_size = 0.8)

from sklearn.ensemble import RandomForestRegressor
r_model = RandomForestRegressor()
r_model.fit(train_x, train_y)
r_model.score(test_x, test_y)
y_hat = r_model.predict(test_x) # 0.8224951211134295
r2_score(test_y, y_hat) # 0.8224951211134295

####### 실습형 2부분 2번
# 레드 와인 퀄리티 예측문제
# 문제타입 : 예측
# 평가지표 : r2 score
from sklearn.metrics import r2_score
import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/redwine/train.csv')

data.info()
data.head(10)

data_y = data['quality'].copy()
data_x = data.drop(['quality'], axis = 1)


from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, train_size = 0.77, random_state = 42)

from sklearn.ensemble import RandomForestRegressor
r_model = RandomForestRegressor(random_state = 42)
r_model.fit(train_x, train_y)
r_model.score(test_x, test_y) # 0.4622900022770151
y_hat = r_model.predict(test_x) 
r2_score(test_y,y_hat) # 0.4622900022770151
