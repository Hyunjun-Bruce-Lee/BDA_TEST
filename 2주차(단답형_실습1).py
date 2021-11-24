###### 단답형
### https://cafe.naver.com/yjbooks?iframe_url_utf8=%2FArticleRead.nhn%253Fclubid%3D19039057%2526articleid%3D16030
# A1: 정형데이터
# A2: 반정형 데이터
# A3: 비정형 데이터

# A1: SMOTE X

# A1: 오즈 X


####### 실습형 1부분
### https://cafe.naver.com/yjbooks?iframe_url_utf8=%2FArticleRead.nhn%253Fclubid%3D19039057%2526articleid%3D16030
data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/smoke/train.csv')

### Q1. 시력(좌) 와 시력(우)의 값이 같은 남성의 허리둘레의 평균은?
data.loc[(data['시력(좌)'] == data['시력(우)']) & (data['성별코드'] == 'M') , '허리둘레'].mean()
### A1 : 84.9768266281797

### Q2. 40대(연령대코드 40,45) 여성 중 '총콜레스테롤'값이 40대 여성의 '총콜레스테롤' 중간값 이상을  가지는 그룹과
### 50대(연령대코드 50,55) 여성 중 '총콜레스테롤'값이 50대 여성의 '총콜레스테롤' 중간값 이상을 가지는 두 그룹의 '수축기혈압'이 독립성,정규성,등분산성이 만족하는것을 확인했다. 
### 두 그룹의 '수축기혈압'의 독립표본 t 검증 결과를 통계값, p-value 구분지어 구하여라.
f1 =data.loc[(data['성별코드']=='F') &(data['연령대코드(5세단위)'].isin([50,55]))]
f2 =data.loc[(data['성별코드']=='F') &(data['연령대코드(5세단위)'].isin([40,45]))]
f1f = f1.loc[f1['총콜레스테롤'] >=f1['총콜레스테롤'].median()]['수축기혈압']
f2f = f2.loc[f2['총콜레스테롤'] >=f2['총콜레스테롤'].median()]['수축기혈압']
from scipy import stats
stats.ttest_ind(f1f, f2f, equal_var=True)
### A2 : statistic=8.954384087520708, pvalue=4.399762427897212e-19

### https://cafe.naver.com/yjbooks?iframe_url_utf8=%2FArticleRead.nhn%253Fclubid%3D19039057%2526articleid%3D16030
data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/smoke/train.csv')

### Q1. 수축기혈압과 이완기 혈압기 수치의 차이를 새로운 컬럼('혈압차') 으로 생성하고, 연령대 코드별 각 그룹 중 '혈압차' 의 분산이 5번째로 큰 연령대 코드를 구하여라
data['혈압차'] = data['수축기혈압'] - data['이완기혈압']
temp = data.groupby('연령대코드(5세단위)')['혈압차'].var().sort_values(ascending = False)
temp.index[4]
### A1 : 60

### Q2. 비만도를 나타내는 지표인 WHtR는 허리둘레 / 키로 표현한다. 일반적으로 0.58이상이면 비만으로 분류한다. 데이터중 WHtR 지표상 비만인 인원의 남/여 비율을 구하여라
data['WHtR'] = data['허리둘레'] / data['신장(5Cm단위)']
temp = data.loc[data['WHtR'] >= 0.58,'성별코드']
temp.value_counts()['M'] / temp.value_counts()['F'] 
### A1 : 1.1693877551020408
