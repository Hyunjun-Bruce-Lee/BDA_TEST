###### 단답형
### https://cafe.naver.com/yjbooks?iframe_url_utf8=%2FArticleRead.nhn%3Fclubid%3D19039057%26articleid%3D16030
# A1: 중심극한정리 X
# A2: 대수의 법칙 X
# A3: EDA
# A4: ANOVA (분산분석)
# A5: 주성분분석 (PCA) X


####### 실습형 1부분
import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/bicycle/seoul_bi.csv')
### Q1.  대여일자별 데이터의 수를 데이터프레임으로 출력하고, 가장 많은 데이터가 있는 날짜를 출력하라
data.keys()
data.groupby('대여일자').size().to_frame()
data.groupby('대여일자').size().to_frame().sort_values(by = 0).index[-1]
### A1 : result = df['대여일자'].value_counts().sort_index().to_frame()
### A1 : answer = result[result.대여일자  == result.대여일자.max()].index[0]

### Q2. 각 일자의 요일을 표기하고 ('Monday' ~'Sunday') 'day_name'컬럼을 추가하고 이를 이용하여 각 요일별 이용 횟수의 총합을 데이터 프레임으로 출력하라
data.keys()
data['대여일자'] = pd.to_datetime(data['대여일자'])
data['요일'] = data['대여일자'].dt.day_name()
data['요일'].value_counts().to_frame()
data.groupby('요일').size().to_frame('size')

#개어렵네ㅅㅂ# Q3. 각 요일별 가장 많이 이용한 대여소의 이용횟수와 대여소 번호를 데이터 프레임으로 출력하라
data.keys()
data['대여소번호']
temp = data.groupby(['요일','대여소번호']).size().reset_index().sort_values(by = ['요일',0], ascending = False)
temp.drop_duplicates('요일')
### A3 : result = df.groupby(['day_name','대여소번호']).size().to_frame('size').sort_values(['day_name','size'],ascending=False).reset_index()
### A3 : answer  = result.drop_duplicates('day_name',keep='first').reset_index(drop=True)
### A3 : print(answer)


### Q4. 나이대별 대여구분 코드의 (일일권/전체횟수) 비율을 구한 후 
###     가장 높은 비율을 가지는 나이대를 확인하라. 일일권의 경우 일일권 과 일일권(비회원)을 모두 포함하라
data.keys()
data['대여구분코드'].unique()
temp = pd.concat([data,pd.get_dummies(data['대여구분코드'])], axis = 1)
temp2 = temp.groupby('연령대코드').agg({'일일권':'sum','일일권(비회원)':'sum','대여구분코드':'size'}).reset_index()
temp2['ratio'] = (temp2['일일권'] + temp2['일일권(비회원)'])/temp2['대여구분코드']
temp2[['연령대코드','ratio']]
# df = data
# daily = df[df.대여구분코드.isin(['일일권','일일권(비회원)'])].연령대코드.value_counts().sort_index()
# total = df.연령대코드.value_counts().sort_index()
# ratio = (daily /total).sort_values(ascending=False)
# print(ratio)
# print('max ratio age ',ratio.index[0])

### Q5. 연령대별 평균 이동거리를 구하여라
data.groupby('연령대코드').mean()['이동거리']
# result = df[['연령대코드','이동거리']].groupby(['연령대코드']).mean()
# print(result)

### Q6. 연령대 코드가 20대인 데이터를 추출하고, 
###     이동거리값이 추출한 데이터의 이동거리값의 평균 이상인 데이터를 추출한다
###     최종 추출된 데이터를 대여일자, 대여소 번호 순서로 내림차순 정렬 후 
###     1행부터 200행까지의 탄소량의 평균을 소숫점 3째 자리까지 구하여라
data_20 = data.loc[data['연령대코드'] == '20대']
data_20_1 = data_20.loc[data['이동거리'] >= data_20['이동거리'].mean()]
data_20_2 = data_20_1.sort_values(by = ['대여일자','대여소번호'], ascending = False)
data_20_2['탄소량'] = data_20_2['탄소량'].apply(float)
round(data_20_2.iloc[:200]['탄소량'].mean(),3)
# tw = df[df.연령대코드 =='20대'].reset_index(drop=True)
# tw_mean = tw[tw.이동거리 >= tw.이동거리.mean()].reset_index(drop=True)
# tw_mean['탄소량'] =tw_mean['탄소량'].astype('float')
# target =tw_mean.sort_values(['대여일자','대여소번호'], ascending=False).reset_index(drop=True).iloc[:200].탄소량
# result = round(target.sum()/len(target),3)
# result 

### Q7. 6월 7일 ~10대의 "이용건수"의 중앙값은?
data.loc[(data['연령대코드'] == '~10대') & (data['대여일자'] == pd.to_datetime('2021-06-07'))]['이용건수'].median()
# df['대여일자']  =pd.to_datetime(df['대여일자'])
# result = df[(df.연령대코드 =='~10대') & (df.대여일자 ==pd.to_datetime('2021-06-07'))].이용건수.median()
# print(result)

# Q8. 평일 (월~금) 출근 시간대(오전 6,7,8시)의 대여소별 
#     이용 횟수를 구해서 데이터 프레임 형태로 표현한 후
#     각 대여시간별 이용 횟수의 상위 3개 대여소와 이용횟수를 출력하라
data.keys()
temp = data.loc[~(data['대여일자'].dt.day_name().isin(['sunday','saturday'])) & (data['대여시간'].isin([6,7,8]))]
temp2 = temp.groupby(['대여시간','대여소번호']).size().to_frame().sort_values(['대여시간','대여소번호'])
temp2.groupby('대여시간').tail(3)
# target = df[(df.day_name.isin(['Tuesday', 'Wednesday', 'Thursday', 'Friday','Monday'])) & (df.대여시간.isin([6,7,8]))]
# result = target.groupby(['대여시간','대여소번호']).size().to_frame('이용 횟수')
# answer = result.sort_values(['대여시간','이용 횟수'],ascending=False).groupby('대여시간').head(3)
# print(answer)

### Q9. 이동거리의 평균 이상의 이동거리 값을 가지는 데이터를 추출하여 추출데이터의 이동거리의 표본표준편차 값을 구하여라
data.keys()
temp = data.loc[data['이동거리'] >= data['이동거리'].mean()]
temp['이동거리'].std()

### Q10. 남성('M' or 'm')과 여성('F' or 'f')의 이동거리값의 평균값을 구하여라
data.keys()
data['성별'] = data['성별'].map(lambda x: 1 if (x in ['M','m']) else 0)
data.groupby('성별').mean()['이동거리']
