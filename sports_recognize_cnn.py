from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np

catrgories = ['golf', 'soccer', 'tennis', 'valleyball']
num_classes = len(catrgories)

image_w = 400
image_h = 400

# 데이터 불러오기
X_train, X_test, Y_train, Y_test = np.load('./results/sport.npy', allow_pickle=True)
X_train = X_train.astype('float') / 256
X_test = X_test.astype('float') / 256

model = Sequential()
model.add(Convolution2D(32, 3, 3, padding='same', input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, padding='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=32, epochs=50)

score = model.evaluate(X_test, Y_test)
print('loss=', score[0])
print('accuracy=', score[1])