# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 14:34:06 2021

@author: huangdezhen
"""
#import statement
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
import numpy as np

#hyperparameter
batch_size = 100
original_dim = 28*28
latent_dim = 2
intermediate_dim = 256
nb_epoch = 5
epsilon_std = 1.0
#创建采样辅助函数
def sampling(args):
    z_mean, z_log_val = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
    return z_mean + K.exp(z_log_val / 2) * epsilon
#encoder
x = Input(shape=(original_dim,), name="input")
h = Dense(intermediate_dim, activation='relu', name="encoding")(x)
z_mean = Dense(latent_dim, name="mean")(h)
z_log_val = Dense(latent_dim, name="log-variance")(h)
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_val])
encoder = Model(x, [z_mean, z_log_val, z], name="encoder")

#decoder
input_decoder = Input(shape=(latent_dim,), name="decoder_input")
decoder_h = Dense(intermediate_dim, activation='relu',
                  name="decoder_h")(input_decoder)
x_decoded = Dense(original_dim, activation='sigmoid',
                  name="flat_decoded")(decoder_h)
decoder = Model(input_decoder, x_decoded, name="decoder")
#6.combine model
output_combined = decoder(encoder(x)[2])
vae = Model(x, output_combined)
vae.summary()
#7. loss function
def vae_loss(x, x_decoded_mean, z_log_var, z_mean,
             original_dim=original_dim):
    xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = -0.5 * K.sum(
        1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis = -1)
    return xent_loss + kl_loss
vae.compile(optimizer='rmsprop', loss=vae_loss)
#8. 拆分训练集/测试集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape(len(x_train), np.prod(x_train.shape[1:]))
x_test = x_test.reshape(len(x_test), np.prod(x_test.shape[1:]))


vae.fit(x_train, x_train,
        shuffle = True,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        validation_data=(x_test, x_test),verbose=1)

