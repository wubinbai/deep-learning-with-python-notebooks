from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense

boston = load_boston()
data = boston.data
target = boston.target

X_train, X_test, y_train, y_test = train_test_split(data,target)

def create():
    model = Sequential()
    model.add(Dense(32,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return model

EPOCHS = 10
BATCH_SIZE = 16
model = create()
history = model.fit(X_train,y_train,epochs=EPOCHS,batch_size=BATCH_SIZE,verbose=10)

test_data = X_test
test_targets = y_test
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

