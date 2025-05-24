"""Multi Label Image Classification using CNN

# **Image classification using Convolution Neural Network**


**By: Ahmad El Jouma**

**Apr 2021**

#-----------------------------------------------

# Install Libraries
"""

!pip install tensorflow-gpu==2.0.0-rc0

import tensorflow as tf
from tensorflow.keras import Sequential

from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPool2D

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tqdm import tqdm

!git clone https://github.com/laxmimerit/Movies-Poster_Dataset.git

"""# -------------Load Data-----------------"""

data = pd.read_csv('/content/Movies-Poster_Dataset/train.csv')

data.shape

data.columns

"""# ---------Shape Of Data train------------"""

data.head(10)

from google.colab import drive
drive.mount('/content/drive')

imgWidth = 350
imgHeight = 350

X = []

imgWidth = 350
imgHeight = 350

X = []

for i in tqdm(range(data.shape[0])):
  path = '/content/Movies-Poster_Dataset/Images/' + data['Id'][i] + '.jpg'
  img = image.load_img(path, target_size=(imgWidth, imgHeight, 3))
  img = image.img_to_array(img)
  img = img/255.0
  X.append(img)

X = np.array(X)

X.shape

plt.imshow(X[5])

y = data.drop(['Id', 'Genre'], axis = 1)
y = y.to_numpy()
y.shape

X_train , X_test , y_train , y_test =  train_test_split(X,y,random_state = 0,test_size = 0.15)
X_train[0].shape

"""# ---------------**Build Model**------------------"""

ImageclassificationCNN = Sequential()
ImageclassificationCNN.add(Conv2D(16, (3,3), activation='relu', input_shape = X_train[0].shape))
ImageclassificationCNN.add(BatchNormalization())
ImageclassificationCNN.add(MaxPool2D(2,2))
ImageclassificationCNN.add(Dropout(0.3))

ImageclassificationCNN.add(Conv2D(32, (3,3), activation='relu'))
ImageclassificationCNN.add(BatchNormalization())
ImageclassificationCNN.add(MaxPool2D(2,2))
ImageclassificationCNN.add(Dropout(0.3))

ImageclassificationCNN.add(Conv2D(64, (3,3), activation='relu'))
ImageclassificationCNN.add(BatchNormalization())
ImageclassificationCNN.add(MaxPool2D(2,2))
ImageclassificationCNN.add(Dropout(0.4))

ImageclassificationCNN.add(Conv2D(128, (3,3), activation='relu'))
ImageclassificationCNN.add(BatchNormalization())
ImageclassificationCNN.add(MaxPool2D(2,2))
ImageclassificationCNN.add(Dropout(0.5))

ImageclassificationCNN.add(Flatten())

ImageclassificationCNN.add(Dense(128, activation='relu'))
ImageclassificationCNN.add(BatchNormalization())
ImageclassificationCNN.add(Dropout(0.5))


ImageclassificationCNN.add(Dense(128, activation='relu'))
ImageclassificationCNN.add(BatchNormalization())
ImageclassificationCNN.add(Dropout(0.5))


ImageclassificationCNN.add(Dense(25, activation='sigmoid'))

ImageclassificationCNN.summary()

ImageclassificationCNN.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])

history = ImageclassificationCNN.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

def plot_AccuracyLossCurve(history, epoch):
  epoch_range = range(1, epoch+1)
  plt.plot(epoch_range, history.history['accuracy'])
  plt.plot(epoch_range, history.history['val_accuracy'])
  plt.title('accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()
  plt.plot(epoch_range, history.history['loss'])
  plt.plot(epoch_range, history.history['val_loss'])
  plt.title('loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

plot_AccuracyLossCurve(history, 10)

classes = np.array(data.columns[2:])
proba = ImageclassificationCNN.predict(img.reshape(1,350,350,3))
top_3 = np.argsort(proba[0])[:-4:-1]
for i in range(3):
    print("{}".format(classes[top_3[i]])+" ({:.3})".format(proba[0][top_3[i]]))
plt.imshow(img)

img = image.load_img('/content/drive/MyDrive/Poster/GOT.jpg', target_size=(img_width, img_height, 3))
plt.imshow(img)
img = image.img_to_array(img)
img = img/255.0

img = img.reshape(1, img_width, img_height, 3)
#classes = data.columns[2:]
#print(classes)
Y_prob = ImageclassificationCNN.predict(img)
top3 = np.argsort(Y_prob[0])[:-6 :-1]
for i in range(5):
  #print(classes[top3[i]])
  print("{}".format(classes[top3[i]])+" ({:.3})".format(Y_prob[0][top3[i]]))

img = image.load_img('/content/drive/MyDrive/Poster/golmal.jpeg',target_size=(350,350,3))
img = image.img_to_array(img)
img = img/255

img = image.load_img('/content/drive/MyDrive/Poster/golmal.jpeg', target_size=(img_width, img_height, 3))
plt.imshow(img)
img = image.img_to_array(img)
img = img/255.0

img = img.reshape(1, img_width, img_height, 3)

Y_prob = ImageclassificationCNN.predict(img)
top3 = np.argsort(Y_prob[0])[:-6 :-1]
for i in range(5):
  #print(classes[top3[i]])
  print("{}".format(classes[top3[i]])+" ({:.3})".format(Y_prob[0][top3[i]]))

img = image.load_img('/content/drive/MyDrive/Poster/Avenger.jpg', target_size=(img_width, img_height, 3))
plt.imshow(img)
img = image.img_to_array(img)
img = img/255.0

img = img.reshape(1, img_width, img_height, 3)

Y_prob = ImageclassificationCNN.predict(img)
top3 = np.argsort(Y_prob[0])[:-6 :-1]
for i in range(5):
  #print(classes[top3[i]])
  print("{}".format(classes[top3[i]])+" ({:.3})".format(Y_prob[0][top3[i]]))