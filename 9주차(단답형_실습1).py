###### 단답형
### https://cafe.naver.com/yjbooks?iframe_url_utf8=%2FArticleRead.nhn%3Fclubid%3D19039057%26articleid%3D16030
# A1: 딕슨의 Q검정 X
# A2: 파싱 X 
# A3: 플럼 X 
# A4: 근접 중심성 X
# A5: 척와 X


####### 실습형 1부분
import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/diabetes/train.csv')
### Q1. Outcome 값에 따른 각 그룹의  각 컬럼의 평균 차이를 구하여라
data.info()
temp = data.groupby('Outcome').mean()
abs(temp.loc[0,:] - temp.loc[1,:])
### A1 : df.groupby('Outcome').mean().diff().iloc[1,:]

data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/nflx/NFLX.csv')
### Q1. 매년 5월달의 open가격의 평균값을 데이터 프레임으로 표현하라
data.keys()
data['Date'] = pd.to_datetime(data['Date'])
data['year'] = data['Date'].map(lambda x: x.year)
data['month'] = data['Date'].map(lambda x: x.month)
data.loc[data['month'] == 5,:].groupby('year').mean()['Open'].to_frame()
### A1 : data['Date'] = pd.to_datetime(data['Date'])
### A1 : target = data.groupby(data['Date'].dt.strftime('%Y-%m')).mean()
### A1 : target.loc[target.index.str.contains('-05')].Open
