'''
Created on 3 Oct 2015

@author: timdettmers
'''
from __future__ import absolute_import
from __future__ import print_function
from keras.layers import containers
from keras.datasets import cifar10, mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad, RMSprop
from keras.utils import np_utils, generic_utils
from keras.layers.core import AutoEncoder
from six.moves import range
import cPickle as pickle
from scipy.spatial.distance import cdist
import numpy as np

'''
    Train a (fairly simple) deep CNN on the CIFAR10 small images dataset.

    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

    It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
    (it's still underfitting at that point, though).

    Note: the data was pickled with Python 2, and some encoding issues might prevent you
    from loading it in Python 3. You might have to load it in Python 2,
    save it in a different format, load it in Python 3 and repickle it.
'''

batch_size = 128
nb_classes = 10
nb_epoch = 50
data_augmentation = False

# shape of the image (SHAPE x SHAPE)
shapex, shapey = 32, 32
# number of convolutional filters to use at each layer
nb_filters = [32, 64]
# level of pooling to perform at each layer (POOL x POOL)
nb_pool = [2, 2]
# level of convolution to perform at each layer (CONV x CONV)
nb_conv = [3, 3]
# the CIFAR10 images are RGB
image_dimensions = 3

# the data, shuffled and split between tran and test sets

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.reshape(X_train.shape[0],X_train.shape[2]**2*3)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[2]**2*3)
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

X_train = np.vstack([X_train, X_test])

pickle.dump(y_train, open('/home/tim/codesy.p','wb'))
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


encoder = containers.Sequential([ Dense(X_train.shape[1], 256), Dense(256, 128),])
decoder = containers.Sequential([ Dense(128, 256), Dense(256, X_train.shape[1])])
#encoder = containers.Sequential([ Dense(X_train.shape[1], 2048)])
#decoder = containers.Sequential([ Dense(2048, X_train.shape[1])])

model = Sequential()
model.add(AutoEncoder(encoder=encoder, decoder=decoder,output_reconstruction=True))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)

model.compile(loss='mean_squared_error', optimizer=sgd)
print([ np.sum(weight) for weight in encoder.get_weights()])

model2 = Sequential()
model2.add(encoder)
model2.compile(loss = "mean_squared_error", optimizer = rmsprop)


if not data_augmentation:
    print("Not using data augmentation or normalization")

    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train /= 255
    X_test /= 255
    model.fit(X_train, X_train, batch_size=batch_size, nb_epoch=nb_epoch,validation_split=0.2)
    #score = model.evaluate(X_test, X_test, batch_size=batch_size)
    #print('Test score:', score)
    
    #print(encoder.predict(X_train[0:10]))
    
    
    print([ np.sum(weight) for weight in encoder.get_weights()])
    print(X_train.shape)
    codes = model2.predict(X_train) 
    print(codes.shape)
    
    
    
    pickle.dump(codes,open('/home/tim/codes.p','wb'))
    
  
        
        #print(correct/float(sample_size))
        #print(Y_train[idx])
    
    
    #print(idx)
    #print(Y_train[idx])
    
    

else:
    print("Using real time data augmentation")

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=True,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    for e in range(nb_epoch):
        print('-'*40)
        print('Epoch', e)
        print('-'*40)
        print("Training...")
        # batch train with realtime data augmentation
        progbar = generic_utils.Progbar(X_train.shape[0])
        for X_batch, Y_batch in datagen.flow(X_train, Y_train):
            loss = model.train_on_batch(X_batch, Y_batch)
            progbar.add(X_batch.shape[0], values=[("train loss", loss)])

        print("Testing...")
        # test time!
        progbar = generic_utils.Progbar(X_test.shape[0])
        for X_batch, Y_batch in datagen.flow(X_test, Y_test):
            score = model.test_on_batch(X_batch, Y_batch)
            progbar.add(X_batch.shape[0], values=[("test loss", score)])

# input shape: (nb_samples, 32)
