###### 단답형
### https://cafe.naver.com/yjbooks?iframe_url_utf8=%2FArticleRead.nhn%3Fclubid%3D19039057%26articleid%3D16030
# A1: SOM X
# A2: 연관분석 X
# A3: 지지도 X
# A4: 신뢰도 X
# A5: 지혜


####### 실습형 1부분
import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/admission/train.csv')
### Q1. Serial No. 컬럼을 제외하고 'Chance of Admit'을 종속변수, 나머지 변수를 독립변수라 할때, 랜덤포레스트를 통해 회귀 예측을 할 떄 변수중요도 값을 출력하라 (시드값에 따라 순서는 달라질수 있음)
from sklearn.ensemble import RandomForestRegressor
data.drop('Serial No.', axis = 1, inplace = True)
data_y = data['Chance of Admit'].copy()
data_x = data.drop('Chance of Admit', axis = 1)
r_model = RandomForestRegressor()
r_model.fit(data_x, data_y)
pd.DataFrame({'import': r_model.feature_importances_}, index = data_x.columns).sort_values(by = 'import', ascending = False)
### A1 : pd.DataFrame({'import': r_model.feature_importances_}, data_x.columns).sort_values(by = 'import', ascending = False)

data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/redwine/train.csv')
### Q1. quality 값이 3인 그룹과  8인 데이터그룹의 각 컬럼별 독립변수의 표준편차 값의 차이를 구할때 가장 큰 컬럼을 구하여라
data.info()
data_3 = data.loc[data['quality'] == 3, :].copy()
data_8 = data.loc[data['quality'] == 8, :].copy()
(abs(data_3.std() - data_8.std())).sort_values(ascending = False).index[0]
### A1 : 'total sulfur dioxide'
