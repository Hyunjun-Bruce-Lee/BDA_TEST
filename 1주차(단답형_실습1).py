####### 단답형
### https://cafe.naver.com/yjbooks?iframe_url_utf8=%2FArticleRead.nhn%253Fclubid%3D19039057%2526articleid%3D16030
# A1 : 배깅
# A2 : 크롤링
# A3 : 다중공선성
# A4 : NoSQL X
# A5 : 감성분석

####### 실습형 1부분
import pandas as pd
### https://cafe.naver.com/yjbooks/15738
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/weather/weather2.csv')

df.keys()

### Q1. 여름철(6월,7월,8월) 이화동이 수영동보다 높은 기온을 가진 시간대는 몇개인가?
df['time'] = pd.to_datetime(df['time'])
df['month'] = df['time'].map(lambda x: x.month)

temp_df = df.loc[df['month'].isin([6,7,8]),:]
(temp_df['이화동기온'] > temp_df['수영동기온']).sum()
### A1 : 1415

### Q2. 이화동과 수영동의 최대강수량의 시간대를 각각 구하여라
df.keys()
df.loc[df['이화동강수'] == df['이화동강수'].max(), 'time']
df.loc[df['수영동강수'] == df['수영동강수'].max(), 'time']
### A2: 수영동 2020-07-23 12:00:00 | 이화동 2020-09-30 09:00:00

### https://cafe.naver.com/yjbooks/15738
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/churn/train.csv')

df.keys()
df['Gender'].describe()

### Q1. 남성 이탈(Exited)이 가장 많은 국가(Geography)는 어디이고 이탈 인원은 몇명인가?
temp_df = df.loc[df['Gender'] == 'Male',:]
temp_df = temp_df.groupby('Geography').sum()['Exited']
temp_df.loc[temp_df == temp_df.max()]
### A1 : Germany    287

### Q2. 카드를 소유(HasCrCard ==1)하고 있으면서 활성멤버(IsActiveMember ==1) 인 고객들의 평균나이는? 
df.loc[(df['HasCrCard']==1) & (df['IsActiveMember'] == 1), 'Age'].mean()
### A2 : 39.61019283746556

### Q3. Balance 값이 중간값 이상을 가지는 고객들의 CreditScore의 표준편차를 구하여라
df.loc[df['Balance'] >= df['Balance'].median(),'CreditScore'].std()
### A3 : 97.29451567120783
