###### 단답형
### https://cafe.naver.com/yjbooks?iframe_url_utf8=%2FArticleRead.nhn%253Fclubid%3D19039057%2526articleid%3D16030
# A1: 홀드아웃 검증 X
# A2: k-폴드 크로스 밸리데이션
# A3: 민감도 sensitivity X
# A4: 특이도 specificity X 
# A5: 민감도 X


####### 실습형 1부분
import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/airline/train.csv')
### Q1. Arrival Delay in Minutes 컬럼이 결측치인 데이터들 중 'neutral or dissatisfied' 보다 'satisfied'의 수가 더 높은 Class는 어디 인가?
temp_data = data.loc[data['Arrival Delay in Minutes'].isna()]
temp_step = temp_data.groupby(['Class','satisfaction']).size()
temp_step.unstack(level = 1)
### A1 : Business


data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/waters/train.csv')
### Q1. ph값은 상당히 많은 결측치를 포함한다. 결측치를 제외한 나머지 데이터들 중 사분위값 기준 하위 25%의 값들의 평균값은?
data.info()
temp_data = data['ph'].dropna()
temp_data.loc[temp_data <= temp_data.quantile(0.25)].mean()
### A1 : 5.057093462441731
