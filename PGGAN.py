# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 21:04:14 2021

@author: huangdezhen
"""

import tensorflow as tf 
import keras as K

def upscale_layer(layer, upscale_factor):
    height = layer.get_shape()[1]
    width = layer.get_shape()[2]
    size = (upscale_factor * height, upscale_factor * width)
    upscaled_layer = tf.image.resize_nearest_neighbor(layer, size)
    return upscaled_layer
def smoothly_merge_last_layer(list_of_layers, alpha):
    
    last_fully_trained_layer = list_of_layers[-2]
    last_layer_upscaled = upscale_layer(last_fully_trained_layer, 2)
    larger_native_layer = list_of_layers[-1]
    assert larger_native_layer.get_shape() == last_layer_upscaled.get_shape()
    new_layer = (1 - alpha) *last_layer_upscaled + larger_native_layer * alpha
    return new_layer

def minibatch_std_layer(layer, group_size = 4):
    group_size = K.backend.minimum(group_size, tf.shape(layer)[0])
    shape = list(K.int_shape(input))
    shape[0] = tf.shape(input)[0]
    minibatch = K.backend.reshape(layer, (group_size, -1, shape[1], shape[2], shape[3]))
    minibatch -= tf.reduce_mean(minibatch, axis = 0, keepdims = True)
    minibatch = tf.reduce_mean(K.backend.square(minibatch), axis=0)
    minibatch = K.backend.square(minibatch + 1e8)
    minibatch = tf.reduce_mean(minibatch, axis=[1,2,4], keepdims=True)
    minibatch = K.backend.tile(minibatch, (group_size, 1, shape[2], shape[3]))
    return K.backend.concatenate([layer, minibatch], axis=1)
def equalize_learning_rate(shape, gain, fan_in=None):
    if fan_in is None: fan_in = np.prod(shape[:-1])
    std = gain / K.sqrt(fan_in)
    wscale = K.constant(std, name='wscale', dtype=np.float32)
    adjusted_weights = K.get_value('layer', shape=shape,
                                   initializer=tf.initializer.random_normal()) * wscale
    return adjusted_weights

def pixelwise_feat_norm(inputs, **kwargs):
    normalization_constant = K.backend.sqrt(K.backend.mean(
        inputs**2, axis=-1, keepdims = True) + 1.0e-8)
    return inputs / normalization_constant

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub

with tf.Graph().as_default():
    module = hub.Module("https://tfhub.dev/google/progan-128/1")
    latent_dim = 512
    
    latent_vector = tf.random_normal([1, latent_dim], seed=1337)
    interpolated_images = module(latent_vector)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        image_out = session.run(interpolated_images)
        
    plt.imshow(image_out.reshape(128, 128, 3))
    plt.show()
    

    