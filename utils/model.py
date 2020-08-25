from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout


class SudokuNet:
    @staticmethod
    def build(width, height, depth, n_classes):
        """Builds the SudokuNet model to OCR the digits"""
        # initialize the model
        model = Sequential()

        # build the layers
        # first set of CNN
        model.add(Conv2D(64, (5, 5), input_shape=(height, width, depth)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (5, 5)))
        model.add(Activation('relu'))
        model.add(MaxPool2D((2, 2)))

        # second set of CNN
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPool2D((2, 2)))
        model.add(Dropout(0.5))

        # first set of FC -> Relu layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        # second set of FC -> Relu layers
        # model.add(Dense(64))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(n_classes))
        model.add(Activation('softmax'))

        # return the built model
        return model
