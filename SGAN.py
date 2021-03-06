# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 09:07:53 2021

@author: huangdezhen
"""

#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

from keras import backend as K

from keras.datasets import mnist
from keras.layers import (Activation, BatchNormalization, Concatenate, Dense,
                          Dropout, Flatten, Input, Lambda, Reshape)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical

img_rows = 28
img_cols = 28
channels = 1

img_shape = (img_rows, img_cols, channels)

z_dim = 100

num_classes = 10

class Dataset:
    def __init__(self, num_labeled):
        
        self.num_labeled = num_labeled
        
        (self.x_train, self.y_train), (self.x_test,
                                       self.y_test) = mnist.load_data()
        
        def preprocess_imgs(x):
            x = (x.astype(np.float32) - 127.5) / 127.5
            x = np.expand_dims(x, axis=3)
            return x
        
        def preprocess_labels(y):
            return y.reshape(-1, 1)
        
        self.x_train = preprocess_imgs(self.x_train)
        self.y_train = preprocess_labels(self.y_train)
        
        self.x_test = preprocess_imgs(self.x_test)
        self.y_test = preprocess_labels(self.y_test)
        
        def batch_labeled(self, batch_size):
            idx = np.random.randint(0, self.num_labeled, batch_size)
            imgs = self.x_train[idx]
            labels = self.y_train[idx]
            return imgs, labels
            
            
            
            
            
        def batch_unlabeled(self, batch_size):
            idx = np.random.randint(self.num_labeled, self.x_train.shape[0], batch_size)
            imgs = self.x_train[idx]
            return imgs
        
        def training_set(self):
            x_train = self.x_train[range(self.num_labeled)]
            y_train = self.y_train[range(self.num_labeled)]
            return x_train, y_train
        
        def test_set(self):
            return self.x_test, self.y_test
        num_labeled = 100
        
        dataset = Dataset(num_labeled)
        
        
def build_generator(z_dim):
    model = Sequential()
    
    model.add(Dense(256 * 7 * 7, input_dim = z_dim))
    model.add(Reshape((7, 7, 256)))
    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same'))
    model.add(Activation('tanh'))
    
    return model

def build_discriminator_net(img_shape):
    model = Sequential()
    
    model.add(Conv2DTranspose(32, kernel_size=3, strides=1,input_shape = img_shape, padding='same'))
    
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2DTranspose(64, kernel_size=3, strides=2, input_shape = img_shape,padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, input_shape = img_shape,padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(num_classes))
    
    return model

def build_discriminator_supervised(discriminator_net):
    model = Sequential()
    model.add(discriminator_net)
    
    model.add(Activation('softmax'))
    
    return model


def build_discriminator_supervised(discriminator_net):
    model = Sequential()
    model.add(discriminator_net)
    
    def predict(x):
        prediction = 1.0 - (1.0 /
                            (K.sum(K.exp(x), axis=-1, keepdims=True) + 1.0))
        return prediction
    model.add(Lambda(predict))
    
    return model

def build_gan(generator, discriminator):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    
    return model
discriminator_net = build_discriminator_net(img_shape)
discriminator_supervised = build_discriminator_supervised(discriminator_net)
discriminator_supervised.compile(loss='categorical_crossentropy',
                                 metric=['accuracy'],
                                 optimizer=Adam())
discriminator_unsupervised = build_discriminator_unsupervised(
    discriminator_net)
discriminator_unsupervised.compile(loss='binary_crossentropy',
                                   optimizer=Adam())
generator = build_generator(z_dim)
discriminator_unsupervised.trainable = False
gan = build_gan(generator, discriminator_unsupervised)
gan.compile(loss='binary_crossentropy', optimizer=Adam())

supervised_lossed = []
iteration_checkpoints = []

def train(iterations, batch_size, sample_interval):
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    for iteration in range(iterations):
        
        imgs, labels = dataset.batch_labeled(batch_size)
        
        labels = to_categorical(labels, num_classes=num_classes)
        
        imgs_unlabeled = dataset.batch_unlabeled(batch_size)
        
        z = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict(z)
        
        d_loss_supervised,
        accuracy = discriminator_supervised.train_on_batch(imgs, labels)
        
        d_loss_real = discriminator_unsupervised.train_on_batch(gen_imgs, fake)
        d_loss_supervised = 0.5* np.add(d_loss_real, d_loss_fake)
        
        z = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict(z)
        
        g_loss = gan.train_on_batch(z, np.ones((batch_size, 1)))
        
        if (iteration + 1) % sample_interval == 0:
            
            supervised_lossesa.append(d_loss_supervised)
            iteration_checkpoints.append(iteration + 1)
            
            print(
                "%d [D loss supervised: %.4f, acc.: %.2f%%] [D loss" +
                "unsupervised: %.4f] [G loss: %f]"
                %(iteration + 1, d_loss_supervised, 100 * accuracy,
                  (d_loss_unsupervised, g_loss)))
            
iterations = 8000
batch_size = 32
sample_interval = 800

train(iteraions, batch_size, sample_interval)

x, y = dataset.test_set()
y = to_categorical(y, num_classes= num_classes)
_, accuracy = discriminator_unsupervised.evaluate(x, y)
print("Test Accuracy: %.2f%%" % (100 * accuracy))

mnist_classifier = build_discriminator_unspervised(
    build_discriminator_net(img_shape))
mnist_classifier.compile(loss='categorical_crossentropy',
                         metrics=['accuracy'],
                         optimizer=Adam())


            


    
    