# 투약하는 약 분류
# 문제타입 : 분류
# 평가지표 : accuracy
data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/drug/train.csv')
data.info()

from sklearn.preprocessing import LabelEncoder
l_encoder = LabelEncoder()
data['Sex'] = l_encoder.fit_transform(data['Sex'])
data['BP'] = l_encoder.fit_transform(data['BP'])
data['Cholesterol'] = l_encoder.fit_transform(data['Cholesterol'])

data.info()
data.describe()

data_x = data.iloc[:,:-1].copy()
data_y = data.iloc[:,-1].copy()

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, train_size = 0.8, stratify = data_y)

from sklearn.ensemble import RandomForestClassifier
r_model = RandomForestClassifier()
r_model.fit(train_x, train_y)
r_model.score(test_x, test_y) # 1

from sklearn.neighbors import KNeighborsClassifier 
k_model = KNeighborsClassifier()
k_model.fit(train_x, train_y)
k_model.score(test_x, test_y) # 0.6875

from sklearn.svm import SVC
s_model = SVC(gamma = 'auto')
s_model.fit(train_x, train_y)
s_model.score(test_x, test_y) # 0.71875

from sklearn.linear_model import LogisticRegression
l_model = LogisticRegression(max_iter = 10000)
l_model.fit(train_x, train_y)
l_model.score(test_x, test_y) # 1.0

####### 실습형 2부분 2번
# 사기회사 분류
# 문제타입 : 분류
# 평가지표 : f1_score
data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/audit/train.csv')
from sklearn.metrics import f1_score

data.info()
data['Risk']
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data['Risk'] = encoder.fit_transform(data['Risk'])
data.drop(['LOCATION_ID'], axis = 1, inplace = True)

data_x = data.iloc[:,:-1].copy()
data_y = data.iloc[:,-1].copy()

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, train_size = 0.8, stratify = data_y)

train_x.isna().sum()
from scipy.stats import mode
train_x.loc[train_x['Money_Value'].isna(), 'Money_Value'] = mode(train_x['Money_Value'])[0].item()

from sklearn.ensemble import RandomForestClassifier
r_model = RandomForestClassifier()
r_model.fit(train_x, train_y)
r_model.score(test_x, test_y) #  1.0
y_hat = r_model.predict(test_x)
f1_score(test_y, y_hat) # 1.0

from sklearn.neighbors import KNeighborsClassifier 
k_model = KNeighborsClassifier()
k_model.fit(train_x, train_y)
k_model.score(test_x, test_y) # 0.9516129032258065
f1_score(test_y, y_hat) # 1.0

from sklearn.svm import SVC
s_model = SVC(gamma = 'auto')
s_model.fit(train_x, train_y)
s_model.score(test_x, test_y) # 0.9838709677419355
f1_score(test_y, y_hat) # 1.0

from sklearn.linear_model import LogisticRegression
l_model = LogisticRegression(max_iter = 10000)
l_model.fit(train_x, train_y)
l_model.score(test_x, test_y) # 1.0
f1_score(test_y, y_hat) # 1.0
