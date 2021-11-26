###### 단답형
### https://cafe.naver.com/yjbooks?iframe_url_utf8=%2FArticleRead.nhn%3Fclubid%3D19039057%26articleid%3D16030
# A1: 향상도 곡선 (lift curve) X
# A2: 계층적 군집 X
# A3: 평균연결법 X
# A4: 맨하탄거리 X
# A5: K - means X


####### 실습형 1부분
import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/MedicalCost/train.csv')
### Q1. 흡연자와 비흡연자 각각 charges의 상위 10% 그룹의 평균의 차이는?
smoke = data.loc[data['smoker'] == 'yes',:].copy()
no_smoke = data.loc[data['smoker'] == 'no',:].copy()
s = smoke.loc[data['charges'] >= smoke['charges'].quantile(0.9), 'charges'].mean()
ns = no_smoke.loc[data['charges'] >= no_smoke['charges'].quantile(0.9), 'charges'].mean()
s-ns
### A1 : 29297.954548156144


data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/kingcountyprice/train.csv')
### Q1. bedrooms 의 빈도가 가장 높은 값을 가지는 데이터들의 price의 상위 10%와 하위 10%값의 차이를 구하여라
data.info()
data.bedrooms.value_counts().sort_values(ascending = False)
u = data.loc[data['bedrooms'] == 3,'price'].quantile(0.9)
d = data.loc[data['bedrooms'] == 3,'price'].quantile(0.1)
u-d
### A1 : 505500.0
