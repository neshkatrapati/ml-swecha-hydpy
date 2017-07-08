import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn import datasets
import numpy

def unison_shuffled_copies(a, b):    
    p = numpy.random.permutation(len(a))
    return a[p], b[p]

# Load the IRIS data
iris = datasets.load_iris()
X = iris.data
Y = iris.target
# Shuffle
X,Y = unison_shuffled_copies(X, Y)

TRAIN_SIZE = int(X.shape[0] * 0.7)
TEST_SIZE = int(X.shape[0] * 0.3)

X_train, Y_train = X[:TRAIN_SIZE], Y[:TRAIN_SIZE]
X_test, Y_test = X[TRAIN_SIZE:], Y[TRAIN_SIZE:]
one_hot_train = keras.utils.to_categorical(Y_train, num_classes=3)
one_hot_test = keras.utils.to_categorical(Y_test, num_classes=3)


model = Sequential()
model.add(Dense(units = 8, input_dim=4))
model.add(Activation('relu'))
model.add(Dense(units = 3))
model.add(Activation('softmax'))

model.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics = ['accuracy'])


model.summary()
# Train
model.fit(X_train, one_hot_train, epochs=30)
# # Evaluate

loss, score = model.evaluate(X_test, one_hot_test)
#loss, score = model.evaluate(X_test, Y_test)
print("Accuracy is", score * 100, "%")

# # Predict
# Y_pred = model.predict(X_test)
# print(Y_test, Y_pred)

# import matplotlib.pyplot as plt
# plt.plot(range(0,len(Y_test)), Y_test)
# plt.plot(range(0,len(Y_pred)), Y_pred)


# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X_test[:,0], X_test[:,1], Y_test)
# ax.scatter(X_test[:,0], X_test[:,1], Y_pred)
# plt.show()




