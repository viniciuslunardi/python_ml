# features (1 sim, 0 não)
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


fish1 = [0, 1, 0]
fish2 = [0, 1, 1]
fish3 = [1, 1, 0]

dog1 = [0, 1, 1]
dog2 = [1, 0, 1]
dog3 = [1, 1, 1]


train_x = [fish1, fish2, fish3, dog1, dog2, dog3]

# 1-> fish 0-> dog
train_y = [1, 1, 1, 0, 0, 0]

model = LinearSVC()
model.fit(train_x, train_y)

animal1 = [1,1,1]
animal2 = [0,1,0]
animal3 = [0,1,1]

test_x = model.predict([animal1, animal2, animal3])

test_y = [0, 1, 1]

corrects = (test_x == test_y).sum()
total = len(test_y)
tax = accuracy_score(test_y, test_x)
print("taxa de acerto: %.2f " % (tax * 100))