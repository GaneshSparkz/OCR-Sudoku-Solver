import argparse
import os

import cv2
import numpy as np
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from utils.model import SudokuNet


# argument parser to get the output folder name as an argument
ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True,
    help="path to save the trained model file")
args = vars(ap.parse_args())

# initializing the hyperparameters like learning rate, number of epochs
# and batch size
LR = 1e-3
EPOCHS = 10
BATCH_SIZE = 50
STEPS_PER_EPOCH = 2000

print("[INFO] loading dataset...")
# dataset path
DATA_PATH = "dataset/"
images = []
labels = []

for i in range(0, 10):
    dir_name = DATA_PATH + str(i)
    img_names = os.listdir(dir_name)
    for filename in img_names:
        image = cv2.imread(dir_name + "/" + filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        image = cv2.resize(image, (32, 32))
        image = image.astype('float') / 255.0
        images.append(image)
        labels.append(i)

print("[INFO] {} images loaded".format(len(images)))

# preprocess the data
print("[INFO] preprocessing the data...")
# convert images into numpy array
images = np.array(images)
labels = np.array(labels)

# split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

# add a channel (i.e., grayscale) dimension to the digits
X_train = X_train.reshape(X_train.shape[0], 32, 32, 1)
X_test = X_test.reshape(X_test.shape[0], 32, 32, 1)

# convert the labels from integers to vectors (one-hot encoding)
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

# augment the data
data_gen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=10,
    shear_range=0.1,
    zoom_range=0.2
)
data_gen.fit(X_train)

# initialize the optimizer
optimizer = Adam(learning_rate=LR)

# build and compile the SudokuNet model
print("[INFO] compiling the SudokuNet model...")
model = SudokuNet.build(width=32, height=32, depth=1, n_classes=10)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# train the SudokuNet model
print("[INFO] training the SudokuNet model...")
history = model.fit_generator(
    data_gen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    shuffle=True
)

# evaluate the model
print("[INFO] evaluating the model...")
y_pred = model.predict(X_test)
print(classification_report(
    y_test.argmax(axis=1),
    y_pred.argmax(axis=1),
    target_names=[str(x) for x in lb.classes_]
))

plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Training', 'Validation'])
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epochs')
plt.show()

# save the trained model to disk
print("[INFO] saving the trained model to {}...".format(args['output']))
model.save(args['output'])
