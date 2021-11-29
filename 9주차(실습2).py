# 당뇨여부 판단하기
# 문제타입 : 분류
# 평가지표 : f1 score
data = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/diabetes/train.csv')
data.info()

data_x = data.iloc[:,:-1].copy()
data_y = data.iloc[:,-1].copy()

train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, train_size = 0.8, stratify = data_y)

r_model = RandomForestClassifier()
r_model.fit(train_x, train_y)
r_model.score(test_x, test_y) # 0.7723577235772358
y_hat = r_model.predict(test_x)
f1_score(test_y, y_hat) # 0.6315789473684211
 
k_model = KNeighborsClassifier()
k_model.fit(train_x, train_y)
k_model.score(test_x, test_y) # 0.7642276422764228
y_hat = k_model.predict(test_x)
f1_score(test_y, y_hat) # 0.6588235294117646

l_model = LogisticRegression(max_iter = 10000)
l_model.fit(train_x, train_y)
l_model.score(test_x, test_y) # 0.7560975609756098
y_hat = l_model.predict(test_x)
f1_score(test_y, y_hat) # 0.5945945945945946



####### 실습형 2부분 2번
# 콩 종류 분류
# 문제타입 : 분류
# 평가지표 : accuracy
data = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/bean/train.csv")
data.info()

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data['Class'] = encoder.fit_transform(data['Class'])


data_x = data.iloc[:,:-1].copy()
data_y = data.iloc[:,-1].copy()

train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, train_size = 0.8, stratify = data_y)

r_model = RandomForestClassifier()
r_model.fit(train_x, train_y)
r_model.score(test_x, test_y) # 0.9186954524575104

 
k_model = KNeighborsClassifier()
k_model.fit(train_x, train_y)
k_model.score(test_x, test_y) # 0.7083141938447405

# 준나 오래걸림
l_model = LogisticRegression(max_iter = 10000)
l_model.fit(train_x, train_y) 
l_model.score(test_x, test_y) #0.9095084979329352
