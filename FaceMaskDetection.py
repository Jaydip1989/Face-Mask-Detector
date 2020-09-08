import numpy as np
import cv2
import os
import keras
from keras.applications import MobileNet
from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data_path = "dataset"
categories = os.listdir(data_path)
labels = [i for i in range(len(categories))]

labels_dict = dict(zip(categories, labels))

print(labels_dict)
print(labels)
print(categories)

img_size = 100
data = []
target = []

for category in categories:
    folder_path = os.path.join(data_path, category)
    img_names = os.listdir(folder_path)

    for img_name in img_names:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (img_size, img_size))
            data.append(resized)
            target.append(labels_dict[category])

        except Exception as e:
            print("Exception", e)

data = np.array(data) / 255.0
data = np.reshape(data, (data.shape[0], img_size, img_size, 1))
target = np.array(target)

from keras.utils import np_utils

new_target = np_utils.to_categorical(target)

np.save('data', data)
np.save('target', new_target)

data = np.load('data.npy')
target = np.load('target.npy')

model = Sequential()
model.add(Conv2D(200, (3, 3), input_shape= data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(100, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(50, activation="relu"))
model.add(Dense(2, activation="softmax"))

print(model.summary())
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

train_data, test_data, target_train, target_test = train_test_split(data, target, test_size = 0.1)

checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                             monitor='val_loss',
                             verbose = 0,
                             save_best_only= True,
                             mode='auto')
history = model.fit(train_data, target_train,
                    epochs = 20,
                    callbacks = [checkpoint],
                    validation_split=0.2)

print(model.evaluate(test_data, target_test))

plt.plot(history.history['loss'], 'r', label='training loss')
plt.plot(history.history['val_loss'], 'b', label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], 'r', label='training accuracy')
plt.plot(history.history['val_accuracy'], 'b', label='validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

