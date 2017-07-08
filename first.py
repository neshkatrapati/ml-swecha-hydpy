
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential()
model.add(Dense(1, input_dim=2))

model.compile(optimizer='sgd',
              loss='mse')

# Get some training data
import numpy as np
TRAIN_SIZE = 1000
X = np.random.random((TRAIN_SIZE, 2))
Y = X.sum(axis = 1)

# Train
model.fit(X, Y, epochs=10, batch_size=32)


# Get some testing data
TEST_SIZE = 100
X_test = np.random.random((TEST_SIZE, 2))
Y_test = X_test.sum(axis = 1)

# Evaluate

score = model.evaluate(X_test, Y_test)
print("Error is", score * 100, "%")

# Predict
Y_pred = model.predict(X_test)
print(Y_test, Y_pred)

import matplotlib.pyplot as plt
plt.plot(range(0,len(Y_test)), Y_test)
plt.plot(range(0,len(Y_pred)), Y_pred)


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test[:,0], X_test[:,1], Y_test)
ax.scatter(X_test[:,0], X_test[:,1], Y_pred)
plt.show()




