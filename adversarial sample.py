# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 11:39:58 2021

@author: huangdezhen
"""

#1.导入声明
import numpy as np
from keras.applications.resnet50 import ResNet50
from foolbox.criteria import Misclassification, ConfidentMisclassification
from keras.preprocssing import image as img
from keras.application.resnet50 import preprocessing_input, decode_predictions
import matplotlib.pyplot as plt 
import foolbox
import pprint as pp
import keras
#%matplotlib inline

#2.辅助函数
def load_image(img_path: str):
    image = img.load_img(img_path, target_size=(224, 224))
    plt.imshow(image)
    x = img.img_to_array(image)
    return x

image = load_image('DSC_0897.jpg')

#3.创建表10.1和10.2
keras.backend.set_learning_phase(0)
kmodel = ResNet50(weights='ImageNet')
preprocessing = (np.array([104, 116, 123]), 1)

fmodel = foolbox.models.KerasModel(kmodel, bounds=(0,255),
                                   preprocessing=preprocessing)

to_classify = np.expand_dims(image, axis=0)
preds = kmodel.predict(to_classify)
print('Predicted:', pp.pprint(decode_predictions(preds, top=20)[0]))
label = np.argmax(preds)

image = image[:, :, ::-1]
attack = foolbox.attacks.FGSM(fmodel, threshold=.9,
criterion=ConfidentMisclassification(.9))
adversarial = attack(image, label)

new_preds = kmodel.predict(np.expand_dims(adversarial, axis=0))
print('Predicted:', pp.pprint(decode_predictions(new_preds, top=20)[0]))

#4.高斯噪声
fig = plt.figure(figsize=(20,20))
sigma_list = list(max_vals.sigma)
mu_list = list(max_vals.mu)
conf_list = []

def make_subplot(x, y, z, new_row=False):
    rand_noise = np.random.normal(loc=mu, scale=sigma, size=(224,224,3))
    rand_noise = np.clip(rand_noise, 0, 255.)
    noise_preds = kmodel.predict(np.expand_dims(rand_noise, axis=0))
    predictions, num = decode_predictions(noise_preds, top=20)[0][0][1:3]
    num = round(num * 100， 2)
    conf_list.append(num)
    ax = fig.add_subplot(x,y,z)
    ax.annotate(prediction, xy=(0.1, 0.6),
                xycoords=ax.transAxes, fontsize=16, color='yellow')
    ax.annotate(f'{num}%' , xy=(0.1,0.4),
                xycoords=ax.transAxes, fontsize=20, color='orange')
    if new_row:
        ax.annotate(f'$\mu$:{mu}, $\sigma$:{sigma}',
                    xy=(-.2, 0.8), xycoords=ax.transAxes,
                    rotation=90, fontsize=16, color='black')
        ax.imshow(rand_noise / 255)
        ax.axis('off')
        
    for i in range(1,101):
        if (i-1) % 10 == 0:
            mu = mu_list.pop(0)
            sigma = sigma_list.pop(0)
            make_subplot(10,10,i,new_row=True)
        else:
            make_subplot(10,10,i)
    
    plt.show()