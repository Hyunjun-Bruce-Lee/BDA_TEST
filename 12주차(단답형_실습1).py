###### 단답형
### 
# A1: 변동계수
# A2: 피어슨상관계수
# A3: 전수조사
# A4: 계통추출법
# A5: 정규분포


####### 실습형 1부분
import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/worldcup/worldcupgoals.csv')
### Q1.  주어진 전체 기간의 각 나라별 골득점수 상위 5개 국가와 그 득점수를 데이터프레임형태로 출력하라주
data.groupby('Country').sum()['Goals'].sort_values(ascending = False).to_frame()[:5]
### A1 :  result = df.groupby('Country').sum().sort_values('Goals',ascending=False).head(5)
### A1 :  print(result)

### Q2. 주어진 전체기간동안 골득점을 한 선수가 가장 많은 나라 상위 5개 국가와 그 선수 숫자를 데이터 프레임 형식으로 출력하라
data.groupby('Country').count()['Player'].sort_values(ascending = False).to_frame()[:5]
### .count() nan 미포함 .size nan 포함
### A2 : result = df.groupby('Country').size().sort_values(ascending=False).head(5)
### A2 : print(result)

### Q3. Years 컬럼은 년도 -년도 형식으로 구성되어있고, 각 년도는 4자리 숫자이다. 년도 표기가 4자리 숫자로 안된 케이스가 존재한다. 해당 건은 몇건인지 출력하라
data.Years = data.Years.str.split('-')
def custom(x):
    for i in x:
        if len(i) != 4:
            return True
    return False
data.Years.map(lambda x: custom(x)).sum()
data.Years.apply(custom).sum()


# Q4. Q3에서 발생한 예외 케이스를 제외한 데이터프레임을 
#     df2라고 정의하고 데이터의 행의 숫자를 출력하라
#     (아래 문제부터는 df2로 풀이하겠습니다) 
df2 = data.loc[data.Years.map(lambda x: not custom(x))]
len(df2)

### Q5. 월드컵 출전횟수를 나타내는 'LenCup' 컬럼을 추가하고 4회 출전한 선수의 숫자를 구하여라
df2['LenCup'] = df2['Years'].map(lambda x: len(x))
df2['LenCup'].value_counts()[4]
### A5 : df2['LenCup'] =df2['yearLst'].str.len()
### A5 : result = df2['LenCup'].value_counts()[4]
### A5 : print(result)

### Q6. Yugoslavia 국가의 월드컵 출전횟수가 2회인 선수들의 숫자를 구하여라
temp = data.loc[data['Country'] == 'Yugoslavia']
(temp['Years'].map(lambda x: len(x)) == 2).sum()
### A6 : result = len(df2[(df2.LenCup==2) & (df2.Country =='Yugoslavia')])
### A6 : print(result)


### Q7. 2002년도에 출전한 전체 선수는 몇명인가?
df2['Years'].map(lambda x: True if ('2002' in x) else False).sum()
### A7 : result =len(df2[df2.Years.str.contains('2002')])
### A7 : print(result)


### Q8. 이름에 'carlos' 단어가 들어가는 선수의 숫자는 몇 명인가? (대, 소문자 구분 x)
df2.Player = df2.Player.str.upper()
df2.Player.map(lambda x: True if ('CARLOS' in x) else False).sum()
### A8 : df2.Player.str.contains('CARLOS').sum()


### Q9. 월드컵 출전 횟수가 1회뿐인 선수들 중에서 가장 많은 득점을 올렸던 선수는 누구인가?
temp = df2.loc[df2['Years'].map(lambda x: True if (len(x) == 1) else False)]
temp.loc[temp['Goals'] == temp['Goals'].max()]
### A9 : result = df2[df2.LenCup==1].sort_values('Goals',ascending=False).Player.values[0]
### A9 : print(result)

### Q10. 월드컵 출전횟수가 1회 뿐인 선수들이 가장 많은 국가는 어디인가?
df2[df2['LenCup'] == 1].groupby('Country').size().sort_values(ascending = False).index[0]
