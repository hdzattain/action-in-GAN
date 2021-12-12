# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 10:57:11 2021

@author: huangdezhen
"""

#1.导入
from __future__ import print_function, division
import scipy
from keras.datasets import mnist
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os

#2.启动cycleGAN类
class CycleGAN():
    def __init__(self):
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        self.dataset_name = 'apple2orange'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))
        
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)
        self.gf = 32
        self.df = 64
        
        self.lambda_cycle = 10.0
        self.lambda_id = 0.9 * self.lambda_cycle
        optimizer = Adam(0.0002, 0.5)
        
#3. 构建网络
self.d_A = self.build_discriminator()
self.d_B = self.build_discriminator()
self.d_A.compile(loss='mse',
                 optimizer=optimizer,
                 metrics=['accuracy'])
self.d_B.compile(loss='mse',
                 optimizer=optimizer,
                 metrics=['accuracy'])
self.g_AB = self.build_generator()
self.g_BA = self.build_generator()

img_A = Input(shape=self.img_shape)
img_B = Input(shape=self.img_shape)

fake_B = self.g_AB(img_A)
fake_A = self.g_BA(img_B)

reconstr_A = self.g_BA(fake_B)
reconstr_B = self.g_AB(fake_A)
img_A_id = self.g_BA(img_A)
img_B_id = self.g_AB(img_B)

self.d_A.trainable = False
self.d_B.trainable = False

valid_A = self.d_A(fake_A)
valid_B = self.d_B(fake_B)

self.combined = Model(inputs=[img_A, img_B],
                      outputs=[valid_A, valid_B,
                               reconstr_A, reconstr_B,
                               img_A_id, img_B_id])
self.combined.compile(loss=['mse','mse',
                            'mse','mse',
                            'mse','mse'],
                      loss_weights=[1, 1,
                                    self.lambda_cycle, self.lambda_cycle,
                                    self.lambda_id, self.lambda_id],
                      optimizer=optimizer)

#4. 构建生成器
def builder_generator(self):
    
    def conv2d(layer_input, filters, f_size=4):
        d = Conv2D(filters, kernel_size=f_size,
                   strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        d = InstanceNormalization()(d)
        return d
    
    def deconv2d(layer_input, skip_input, filters, f_size=4,
                 dropout_rate=0):
        
        u = Upsampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = InstanceNormalization()(u)
        u = Concatenate()([u, skip_input])
        return u
    
    d0 = Input(shape=self.img_shape)
    
    d1 = conv2d(d0, self.gf)
    d2 = conv2d(d1, self.gf * 2)
    d3 = conv2d(d2, self.gf * 4)
    d4 = conv2d(d3, self.gf * 8)
    
    u1 = deconv2d(d4, d3, self.gf * 4)
    u2 = deconv2d(u1, d2, self.gf * 2)
    u3 = deconv2d(u2, d1, self.gf)
    
    u4 = UpSampling2D(size=2)(u3)
    
    output_img = Conv2D(self.channels, kernel_size=4,
                        strides=1, padding='same', activation='tanh')(u4)
    
    return Model(d0, output_img)
#5. 构建鉴别器

def build_discriminator(self):
    def d_layer(layer_input, filters, f_size=4, normalization=True):
        d = Conv2D(filters, kernel_size=f_size,
                   strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if normalization:
            d = InstanceNormalization()(d)
        return d
    img = Input(shape=self.img_shape)
    
    d1 = d_layer(img, self.df, normalization=False)
    d2 = d_layer(d1, self.df * 2)
    d3 = d_layer(d2, self.df * 4)
    d4 = d_layer(d3, self.df * 8)
    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
    
    return Model(img, validity)

#6. CycleGAN的训练算法
def train(self, epochs, batch_size=1, sample_interval=50):
    
    start_time = datetime.datetime.now()
    
    valid = np.ones((batch_size,) + self.disc_patch)
    fake = np.zeros((batch_size,) + self.disc_patch)
    
    for epoch in range(epochs):
        for batch_i, (imgs_A, imgs_B) in enumerate(
                self.data_loader.load_batch(batch_size)):
            
            fake_B = self.g_AB.predict(imgs_A)
            fake_A = self.g_BA.predict(imgs_B)
            
            dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
            dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
            dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)
            
            dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
            dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
            dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)
            
            d_loss = 0.5 * np.add(dA_loss, dB_loss)
            
            g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
                                                  [valid, valid,
                                                   imgs_A, imgs_B,
                                                   imgs_A, imgs_B])
            if batch_i % sample_interval == 0:
                self.sample_images(epoch, batch_i)
                
#7.运行CycleGAN
gan = CycleGAN()
gan.train(epochs=100, batch_size=64, sample_interval=10)

            