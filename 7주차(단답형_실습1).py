##################### 7주차
###### 단답형
### https://cafe.naver.com/yjbooks?iframe_url_utf8=%2FArticleRead.nhn%3Fclubid%3D19039057%26articleid%3D16030
# A1: 데이터베이스
# A2: 혼합분포군집 X
# A3: EM 알고리즘 X
# A4: apriori X
# A5: 스테밍 (stemming) X


####### 실습형 1부분
import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/drug/train.csv')
### Q1. 남성들의 연령대별 (10살씩 구분 0~9세 10~19세 ...) Na_to_K값의 평균값을 구해서 데이터 프레임으로 표현하여라
data.keys()
data.info()
m_data = data.loc[data['Sex'] == 'M']
m_data['new_age'] = m_data['Age'].map(lambda x: x//10)
m_data.groupby('new_age', as_index = False)['Na_to_K'].mean()
### A1 : m_data.groupby('new_age').mean()['Na_to_K'].to_frame()

data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/audit/train.csv')
### Q1. Q1. 데이터의 Risk 값에 따른 score_a와 score_b의 평균값을 구하여라
data.info()
data['Risk']
data.groupby('Risk').mean()[['Score_A', "Score_B"]]
### A1 : 'total sulfur dioxide'
