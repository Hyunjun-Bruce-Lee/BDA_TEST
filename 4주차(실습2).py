####### 실습형 2부분 1번
# 비행탑승 경험 만족도 (satisfaction 컬럼 : 'neutral or dissatisfied' or satisfied ) 
# 문제타입 : 분류유형
# 평가지표 : accuracy
import pandas as pd
from scipy.stats import mode
data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/airline/train.csv')
data.keys()
data_y = data.iloc[:,-1].copy()
data_x = data.iloc[:,1:-1].copy()

data_x.info()
data_x.fillna(int(mode(data_x.iloc[:,-1])[0].item()), inplace = True)

from sklearn.preprocessing import LabelEncoder
L_encoder = LabelEncoder()
data_x['Gender'] = L_encoder.fit_transform(data_x['Gender'])
data_x['Class'] = L_encoder.fit_transform(data_x['Class'])
data_x['Customer Type'] = L_encoder.fit_transform(data_x['Customer Type'])
data_x['Type of Travel'] = L_encoder.fit_transform(data_x['Type of Travel'])

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, train_size = 0.8, stratify = data_y)

from sklearn.ensemble import RandomForestClassifier
r_model = RandomForestClassifier()
r_model.fit(train_x, train_y)
r_model.score(test_x, test_y) # 0.9617443609022557

# SVC fit 시간 너무 오래 걸림
# from sklearn.svm import SVC
# s_model = SVC(gamma = 'auto')
# s_model.fit(train_x, train_y)
# s_model.score(test_x, test_y)

from sklearn.linear_model import LogisticRegression
l_model = LogisticRegression(max_iter = 10000)
l_model.fit(train_x, train_y)
l_model.score(test_x, test_y) # 0.8762105263157894 # 시간 아슬아슬

from sklearn.neighbors import KNeighborsClassifier
k_model = KNeighborsClassifier()
k_model.fit(train_x, train_y)
k_model.score(test_x, test_y) # 0.7361804511278196 # 시간 아슬아슬


####### 실습형 2부분 2번
# 수질 음용성 여부 (Potablillity 컬럼 : 0 ,1 )
# 문제타입 : 분류유형
# 평가지표 : accuracy
import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/waters/train.csv')

data.info()

import matplotlib.pyplot as plt
for i in data.keys():
    data[i].plot()
    plt.show()

temp = data.dropna()
len(temp)

data_y = temp.iloc[:,-1].copy()
data_x = temp.iloc[:,:-1].copy()

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, train_size = 0.8, stratify = data_y)

from sklearn.ensemble import RandomForestClassifier
r_model = RandomForestClassifier()
r_model.fit(train_x, train_y)
r_model.score(test_x, test_y) # 0.6934984520123839

from sklearn.neighbors import KNeighborsClassifier
k_model = KNeighborsClassifier()
k_model.fit(train_x, train_y)
k_model.score(test_x,test_y) # 0.5572755417956656

from sklearn.svm import SVC
s_model = SVC(gamma = 'auto')
s_model.fit(train_x, train_y)
s_model.score(test_x, test_y) # 0.5913312693498453

from sklearn.linear_model import LogisticRegression
l_model = LogisticRegression(penalty = 'l2')
l_model.fit(train_x, train_y)
l_model.score(test_x, test_y) # 0.5975232198142415
