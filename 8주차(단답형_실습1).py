###### 단답형
### https://cafe.naver.com/yjbooks?iframe_url_utf8=%2FArticleRead.nhn%3Fclubid%3D19039057%26articleid%3D16030
# A1: 사회연결망 분석(SNA) X
# A2: 연결정도 중심성 X
# A3: 결측치  (결측값) 
# A4: 사분위수 기법 X
# A5: 회귀 대체법 X


####### 실습형 1부분
import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/muscle/train.csv')
### Q1. pose값에 따른 각 motion컬럼의 중간값의 가장 큰 차이를 보이는 motion컬럼은 어디이며 그값은?
data.keys()
data['pose']
temp = data.groupby('pose').median()
abs(temp.iloc[0,:] - temp.iloc[1,:]).sort_values(ascending = False)
### A1 : t= df.groupby('pose').median().T
### A1 : abs(t[0] - t[1]).sort_values()


data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/hyundai/train.csv')
### Q1. 정보(row수)가 가장 많은 상위 3차종의 price값의 각 평균값은?
data.keys()
top_3 = data.groupby('model').size().sort_values(ascending = False)[:3].index
temp = dict()
for i in top_3:
    temp[i] = data.loc[data['model'] == i, 'price'].mean()
pd.DataFrame.from_dict(temp, orient = 'index')
### A1 : df.loc[df.model.isin(df.model.value_counts().index[:3])].groupby('model').mean()['price'].to_frame()
