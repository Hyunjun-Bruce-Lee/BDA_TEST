###### 단답형
### 
# A1: LDA 선형 판별분석 X
# A2: t-SNE X
# A3: 파생변수 X
# A4: 로버스트 스케일링 X
# A5: TF-IDF X


####### 실습형 1부분
import pandas as pd
data1 = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/youtube/videoInfo.csv')
data2 = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/youtube/channelInfo.csv')
### Q1.  각 데이터의 'ct'컬럼을 시간으로 인식할수 있게 datatype을 변경하고 
###      video 데이터의 videoname의 각 value 마다 몇개의 데이터씩 가지고 있는지 확인하라
data1.keys()
data2.keys()
data1['ct'] = pd.to_datetime(data1['ct'])
data2['ct'] = pd.to_datetime(data2['ct'])
data1.groupby('videoname').size()
### A1 : video['ct'] = pd.to_datetime(video['ct'])
### A1 : answer = video.videoname.value_counts()
### A1 : print(answer)

#별별틀린듯## Q2. 수집된 각 video의 가장 최신화 된 날짜의 viewcount값을 출력하라
data1.sort_values(by = ['videoname', 'ct'], ascending = False).drop_duplicates('videoname')[['viewcnt','videoname']]
### A2 : video.sort_values(['videoname','ct']).drop_duplicates('videoname',keep='last')[['viewcnt','videoname']]

#별별틀린듯## Q3. Channel 데이터중 2021-10-03일 이후의 각 채널의 처음 구독자 수(subcnt)를 출력하라
data2.loc[data2['ct'] >= pd.to_datetime('2021-10-03'),:].groupby('channelname').min('ct')['subcnt']
# 위처럼 하면 10월 03일 이후의 구독자 최소값이 나옴
temp = data2.loc[data2.ct >= pd.to_datetime('2021-10-03')]
temp.sort_values(by = ['channelid','ct']).drop_duplicates('channelid')[['channelname','subcnt']]
### A3 : target = channel[channel.ct >= pd.to_datetime('2021-10-03')].sort_values(['ct','channelname']).drop_duplicates('channelname')
### A3 : answer = target[['channelname','subcnt']]

#별별감도못잡음# Q4. 각채널의 2021-10-03 03:00:00 ~ 2021-11-01 15:00:00 까지 구독자수 (subcnt) 의 증가량을 구하여라
data2.keys()
a = data2.loc[data2.ct.dt.strftime('%Y-%m-%d %H') == '2021-10-03 03'][['channelname','subcnt']]
b = data2.loc[data2.ct.dt.strftime('%Y-%m-%d %H') == '2021-11-01 15'][['channelname','subcnt']]
a.sort_values(by = 'channelname', inplace = True)
b.sort_values(by = 'channelname', inplace = True)
a.reset_index(inplace = True)
b.reset_index(inplace = True)
inc = b.subcnt - a.subcnt
a['inc'] = inc
a[['channelname','inc']]
### A4 : end = channel.loc[channel.ct.dt.strftime('%Y-%m-%d %H') =='2021-11-01 15']
### A4 : start = channel.loc[channel.ct.dt.strftime('%Y-%m-%d %H') =='2021-10-03 03']
### A4 : end_df = end[['channelname','subcnt']].reset_index(drop=True)
### A4 : start_df = start[['channelname','subcnt']].reset_index(drop=True)
### A4 : end_df.columns = ['channelname','end_sub']
### A4 : start_df.columns = ['channelname','start_sub']
### A4 : tt = pd.merge(start_df,end_df)
### A4 : tt['del'] = tt['end_sub'] - tt['start_sub']
### A4 : tt[['channelname','del']]


#별별감도못잡음# Q5. 각 비디오는 10분 간격으로 구독자수, 좋아요, 싫어요수, 댓글수가 수집된것으로 알려졌다. 
###     공범 EP1의 비디오정보 데이터중 수집간격이 5분 이하, 20분이상인 데이터 구간( 해당 시점 전,후) 의 시각을 모두 출력하라
from datetime import timedelta
data1.keys()
data1['videoname'].unique()
temp = data1.loc[data1['videoname'] == ' 공범 EP1',:].sort_values(by = 'ct')
(temp['ct'].diff() <= timedelta(minutes = 5)).sum()
### A5 : import datetime
### A5 : ep_one = video.loc[video.videoname.str.contains('1')].sort_values('ct').reset_index(drop=True)
### A5 : ep_one[(ep_one.ct.diff(1) >=datetime.timedelta(minutes=20)) | (ep_one.ct.diff(1) <=datetime.timedelta(minutes=5))]
### A5 : answer = ep_one[ep_one.index.isin([720,721,722,723,1635,1636,1637])]


#별별감도못잡음# Q6. 각 에피소드의 시작날짜(년-월-일)를 에피소드 이름과 묶어 데이터 프레임으로 만들고 출력하라
temp = data1.sort_values(by = ['videoname','ct']).drop_duplicates('videoname')[['videoname','ct']]
temp.ct = temp.ct.dt.date
temp
### A6 :  start_date = video.sort_values(['ct','videoname']).drop_duplicates('videoname')[['ct','videoname']]
### A6 :  start_date['date'] = start_date.ct.dt.date
### A6 :  answer = start_date[['date','videoname']]

### Q7. "공범" 컨텐츠의 경우 19:00시에 공개 되는것으로 알려져있다.  
###      공개된 날의 21시의 viewcnt, ct, videoname 으로 구성된 데이터 프레임을 viewcnt를 내림차순으로 정렬하여 출력하라
temp = data1.sort_values(by = ['videoname','ct'])[['videoname','ct','viewcnt']]
temp_idx = temp['ct'].map(lambda x: True if str(x.time())[:2] == '21' else False)
temp.loc[temp_idx].drop_duplicates('videoname').sort_values(by = 'viewcnt', ascending = False)[['videoname','viewcnt','ct']].reset_index().drop('index', axis = 1)
### A7
# video['time']= video.ct.dt.hour
# answer = video.loc[video['time'] ==21] \
            # .sort_values(['videoname','ct'])\
            # .drop_duplicates('videoname') \
            # .sort_values('viewcnt',ascending=False)[['videoname','viewcnt','ct']]\
            # .reset_index(drop=True)


### Q8. video 정보의 가장 최근 데이터들에서 각 에피소드의 싫어요/좋아요 비율을 ratio 컬럼으로 만들고
###     videoname, ratio로 구성된 데이터 프레임을 ratio를 오름차순으로 정렬하라
data1.keys()
data1['ratio'] = data1['dislikecnt']/data1['likecnt']
temp = data1.sort_values(by = ['videoname','ct']).drop_duplicates('videoname', keep = 'last')
temp.loc[:,['videoname', 'ratio']].sort_values(by = 'ratio')
### target = video.sort_values('ct').drop_duplicates('videoname',keep='last')
### target['ratio'] =target['dislikecnt'] / target['likecnt']
### answer = target.sort_values('ratio')[['videoname','ratio']].reset_index(drop=True)
### answer



### Q9. 2021-11-01 00:00:00 ~ 15:00:00까지 각 에피소드별 viewcnt의 증가량을 데이터 프레임으로 만드시오
d_range = pd.date_range('2021-11-01 00:00:00', '2021-11-01 15:00:00', freq = 'S')
temp_data = data1.loc[data1['ct'].map(lambda x: True if (x in d_range) else False)]0
st = temp_data.sort_values(by = ['videoname','ct']).drop_duplicates('videoname')[['videoname','ct','viewcnt']].reset_index(drop = True)
en = temp_data.sort_values(by = ['videoname','ct']).drop_duplicates('videoname', keep = 'last')[['videoname','ct','viewcnt']].reset_index(drop = True)
en['inc'] = en.viewcnt - st.viewcnt
en[['videoname', 'inc']]
# start = pd.to_datetime("2021-11-01 00:00:00")
# end = pd.to_datetime("2021-11-01 15:00:00")
# target = video.loc[(video["ct"] >= start) & (video['ct'] <= end)].reset_index(drop=True)
# def check(x):
#     result = max(x) - min(x)
#     return result
# answer = target[['videoname','viewcnt']].groupby("videoname").agg(check)
# answer

### Q10. video 데이터 중에서 중복되는 데이터가 존재한다. 중복되는 각 데이터의 시간대와  videoname 을 구하여라
data1.loc[data1.duplicated()][['ct','videoname']]
