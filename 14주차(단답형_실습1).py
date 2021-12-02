##################### 14주차
###### 단답형
### 
# A1: 전진선택법
# A2: 정상성
# A3: 백색잡음
# A4: 워드클라우드
# A5: arima


####### 실습형 1부분
import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/happy2/happiness.csv')

###  Q1.  데이터는 2018년도와 2019년도의 전세계 행복 지수를 표현한다. 
###       각년도의 행복랭킹 10위를 차지한 나라의 행복점수의 평균을 구하여라
data.keys()
data.loc[data['행복랭킹'] == 10]['점수'].mean()

### Q2. 데이터는 2018년도와 2019년도의 전세계 행복 지수를 표현한다. 
###     각년도의 행복랭킹 50위이내의 나라들의 각각의 행복점수 평균을 데이터프레임으로 표시하라
data.loc[data['행복랭킹'] <= 50].groupby(['년도','나라명']).mean()['점수'].to_frame().sort_values(by = '년도')
# result = df[df.행복랭킹<=50][['년도','점수']].groupby('년도').mean()
# print(result)

### Q3. 2018년도 데이터들만 추출하여 행복점수와 부패에 대한 인식에 대한 상관계수를 구하여라
temp = data.loc[data['년도'] == 2018][['점수','부패에 대한인식']]
temp.corr().iloc[0,1]

### Q4. 2018년도와 2019년도의 행복랭킹이 변화하지 않은 나라명의 수를 구하여라
(data[['행복랭킹','나라명']].duplicated()).sum()

# Q5. 2019년도 데이터들만 추출하여 각변수간 상관계수를 구하고 내림차순으로 
#     정렬한 후 상위 5개를 데이터 프레임으로 출력하라
#     컬럼명은 v1,v2,corr으로 표시하라
temp = data.loc[data['년도'] == 2019]
temp_1 = temp.corr().unstack().sort_values(ascending = False).dropna()
temp_1 = temp_1.loc[temp_1 != 1].drop_duplicates()
temp_1 = temp_1.to_frame('corr').reset_index()[:5]
temp_1.columns = ['v1','v2','corr']
temp_1

# Q6. 각 년도별 하위 행복점수의 하위 5개 국가의 평균 행복점수를 구하여라
data.sort_values(by = ['년도','점수']).groupby(['년도']).head(5)[['년도','나라명','점수']].groupby('년도').mean()

# Q7.2019년 데이터를 추출하고 
#    해당데이터의 상대 GDP 평균 이상의 나라들과 평균 이하의 나라들의 
#    행복점수 평균을 각각 구하고 그 차이값을 출력하라
temp = data.loc[data['년도'] == 2019]
temp.keys()
temp_up = temp.loc[temp['상대GDP'] >= temp['상대GDP'].mean()]['점수'].mean()
temp_dw = temp.loc[temp['상대GDP'] <= temp['상대GDP'].mean()]['점수'].mean()
temp_up - temp_dw


# Q8.  각년도의 부패에 대한인식을 내림차순 정렬했을때 상위 20개 국가의 부패에 대한인식의 평균을 구하여라
data.sort_values(['년도','부패에 대한인식'], ascending = False).groupby('년도').head(20).groupby('년도').mean()['부패에 대한인식']



# Q9. 2018년도 행복랭킹 50위 이내에 포함됐다가 2019년 50위 밖으로 밀려난 국가의 숫자를 구하여라
50 - data.loc[(data['행복랭킹'] <= 50)]['나라명'].duplicated().sum()

# Q10. 2018년,2019년 모두 기록이 있는 나라들 중 
#      년도별 행복점수가 가장 증가한 나라와 그 증가 수치는?
temp1 = data.loc[data['년도'] == 2018]
temp2 = data.loc[data['년도'] == 2019]
nation = list(set(temp1.나라명) & set(temp2.나라명))
name = str()
score = 0
for i in nation:
    x = temp2.loc[temp2['나라명'] == i,'점수'].item() - temp1.loc[temp1['나라명'] == i,'점수'].item()
    if x > score:
        name = i
        score = x
print(name, score)

# df = data
# count = df.나라명.value_counts()
# target = count[count>=2].index
# df2 =df.copy()
# multiple = df2[df2.나라명.isin(target)].reset_index(drop=True)
# multiple.loc[multiple['년도']==2018,'점수'] = multiple[multiple.년도 ==2018]['점수'].values * (-1)
# result = multiple.groupby('나라명').sum()['점수'].sort_values().to_frame().iloc[-1]
# result
