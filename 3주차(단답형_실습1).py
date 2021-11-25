###### 단답형
### https://cafe.naver.com/yjbooks?iframe_url_utf8=%2FArticleRead.nhn%253Fclubid%3D19039057%2526articleid%3D16030
# A1: 시그모이드
# A2: 가지치기(Pruning)
# A3: 카이제곱 X
# A4: 앙상블
# A5: 부스팅


####### 실습형 1부분
import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/insurance/train.csv')

### Q1. Vehicle_Age 값이 2년 이상인 사람들만 필터링 하고 그중에서 Annual_Premium 값이 전체 데이터의 중간값 이상인 사람들을 찾고, 그들의 Vintage값의 평균을 구하여라
data['Vehicle_Age'].unique()
data_step_1 = data.loc[data['Vehicle_Age'] == '> 2 Years', :]
data_step_1.loc[data_step_1['Annual_Premium'] >= data['Annual_Premium'].median(), 'Vintage'].mean()
### A1 : 154.43647182359118

### Q2. vehicle_age에 따른 각 성별(gender)그룹의 Annual_Premium값의 평균을 구하여 아래 테이블과 동일하게 구현하라
data_step_1 = data.groupby(['Gender','Vehicle_Age'], as_index = False)['Annual_Premium'].mean()
data_step_1.pivot(index = 'Vehicle_Age', columns = 'Gender', values = 'Annual_Premium')
### A2-1 : data_step_1.pivot(index = 'Vehicle_Age', columns = 'Gender', values = 'Annual_Premium')
### A2-2 : data_step_1 = data.groupby(['Gender','Vehicle_Age'])['Annual_Premium'].mean()
### A2-2 : temp = pd.DataFrame(data_step_1)
### A2-2 : temp.unstack(level = 0)


import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/mobile/train.csv')
### Q1. price_range 의 각 value를 그룹핑하여 각 그룹의 n_cores 의 빈도가 가장높은 value와 그 빈도수를 구하여라
data.keys()
data['n_cores']
temp = data.groupby(['price_range','n_cores'], as_index = True).count()['blue']

max_index = list()
for i,_ in temp.keys():
    test = list()
    for _,j in temp.keys():
        test.append(temp[(i,j)])
    max_index.append((i,test.index(max(test)) + 1))
set(max_index)

### A1 : data[['price_range','n_cores']].groupby(['price_range','n_cores']).size().sort_values(ascending = False).groupby(level = 0).head(1)
### A1 : data[['price_range','n_cores']].groupby(['price_range','n_cores']).size().sort_values(0).groupby(level=0).tail(1)

### Q2. price_range 값이 3인 그룹에서 상관관계가 2번째로 높은 두 컬럼과 그 상관계수를 구하여라
temp_data = data.loc[data['price_range'] == 3,:]
temp_data_2 = temp_data.corr().unstack().sort_values(ascending = False)
temp_data_2.loc[temp_data_2 != 1].reset_index().iloc[1,:]
### A2 : pc - fc , 0.635166
