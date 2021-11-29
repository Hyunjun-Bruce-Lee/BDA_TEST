###### 단답형
### 
# A1: ETL X 
# A2: 래퍼 기법 X
# A3: 유전자 알고리즘 X
# A4: 매개중심성 X
# A5: 라쏘


####### 실습형 1부분
import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/youtube/youtube.csv')
### Q1.  인기동영상 제작횟수가 많은 채널 상위 10개명을 출력하라 (날짜기준, 중복포함)
data.keys()
ch_list = data['channelId'].value_counts().sort_values(ascending =False)[:10].index
data.loc[data['channelId'].isin(ch_list),'channelTitle'].unique()
### A1 : print(list(df.loc[df.channelId.isin(df.channelId.value_counts().head(10).index)].channelTitle.unique()))

### Q2. 논란으로 인기동영상이 된 케이스를 확인하고 싶다. dislikes수가 like 수보다 높은 동영상을 제작한 채널을 모두 출력하라
data.loc[data['dislikes'] > data['likes'],'channelTitle'].unique()
### A2 : print(list(df.loc[df.likes < df.dislikes].channelTitle.unique()))

### Q3. 채널명을 바꾼 케이스가 있는지 확인하고 싶다. channelId의 경우 고유값이므로 이를 통해 채널명을 한번이라도 바꾼 채널의 갯수를 구하여라
count = 0 
for i in data.channelId.unique():
    if len(data.loc[data['channelId'] == i,'channelTitle'].unique()) > 1:
        count += 1
### A3 : change = data[['channelTitle','channelId']].drop_duplicates().channelId.value_counts()
### A3 : target = change[change>1]
### A3 : print(len(target))


### Q4. 일요일에  인기있었던 영상들중 가장많은 영상 종류(categoryId)는 무엇인가?
data['trending_date2'] = pd.to_datetime(data['trending_date2'])
data.loc[data['trending_date2'].dt.day_name() == 'Sunday','categoryId'].value_counts().index[0]

### Q5. 각 요일별 인기 영상들의 categoryId는 각각 몇개 씩인지 하나의 데이터 프레임으로 표현하라
data['day_name'] = data['trending_date2'].dt.day_name()
temp = data.groupby(['day_name','categoryId']).size()
temp.unstack().T
### A5 : group = df.groupby([df['trending_date2'].dt.day_name(),'categoryId'],as_index=False).size()
### A5 : group.pivot(index='categoryId',columns='trending_date2')

### Q6. 댓글의 수로 (comment_count) 영상 반응에 대한 판단을 할 수 있다.
###     viewcount대비 댓글수가 가장 높은 영상을 확인하라 (view_count값이 0인 경우는 제외한다)
data.keys()
temp = data.loc[data['view_count'] != 0, :]
temp['v_c_ratio'] = temp['comment_count']/temp['view_count']
temp.loc[temp['v_c_ratio'] == temp['v_c_ratio'].max(),'title']
###A6 :  target = df.loc[df.view_count !=0]
###A6 :  most = (target['comment_count'] / target['view_count']).dropna().sort_values().index[-1]
###A6 :  target.iloc[most].title


### Q7. 댓글의 수로 (comment_count) 영상 반응에 대한 판단을 할 수 있다.
###     viewcount대비 댓글수가 가장 낮은 영상을 확인하라 (view_counts, ratio값이 0인경우는 제외한다.)
temp = data.loc[data['view_count'] != 0, :]
temp['v_c_ratio'] = temp['comment_count']/temp['view_count']
temp = temp.loc[temp['v_c_ratio'] != 0, :]
temp.loc[temp['v_c_ratio'] == temp['v_c_ratio'].min(),'title'].item()

### Q8. like 대비 dislike의 수가 가장 적은 영상은 무엇인가? (like, dislike 값이 0인경우는 제외한다)
temp = data.loc[(data['likes'] != 0) & (data['dislikes'] != 0)]
temp['ratio'] = temp['dislikes']/temp['likes']
temp.loc[temp['ratio'] == temp['ratio'].min(),'Title'].item()

### Q9. 가장많은 트렌드 영상을 제작한 채널의 이름은 무엇인가? (날짜기준, 중복포함)
temp = data.groupby('channelId').size()
data.loc[data['channelId'] == temp[temp == temp.max()].index.item(),'channelTitle'].unique()
### A9 : df.loc[df.channelId ==df.channelId.value_counts().index[0]].channelTitle.unique()[0]

### Q10. 20회(20일)이상 인기동영상 리스트에 포함된 동영상의 숫자는?
data.keys()
(data.groupby('title').size() >= 20).sum()
#틀림 A10 : (df[['title','channelId']].value_counts()>=20).sum()
