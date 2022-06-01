import pandas as pd 
from sklearn.svm import LinearSVR
from sklearn.metrics import accuracy_score


uri='data.csv'

dados = pd.read_csv(uri)
x = dados[["home",  "how_it_works", "contact"]]
y = dados["bought"]

train_x = x[:75]
train_y = y[:75]

test_x = x[75:]
test_y = y[75:]

model = LinearSVR()

print ("Treinando com %d elementos e testando com %d elementos" % (len(train_x), len(test_x)))

model.fit(train_x, train_y.values.ravel()) # training response
predict = model.predict(test_x)

corrects = (test_x == test_y).sum()
total = len(test_y)

score = corrects/total
#score = accuracy_score(test_y, predict)

print ("Acurácia de %.2f %" % (score))

