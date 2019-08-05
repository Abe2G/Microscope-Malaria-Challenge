# -*- coding: utf-8 -*-

# import the necessary packages
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Activation, Input, Flatten, Dense,Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import add, average #maximum,dot,concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

import numpy as np
from sklearn.base import BaseEstimator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import tensorflow as tf


class model(BaseEstimator):
    """Main class for Classification problem."""
    def __init__(self):
        """Init method. """
        self.num_train_samples = 0
        self.num_feat = 1
        self.num_labels = 2
        self.is_trained = False
        self.filters=(64, 32, 64, 128)
        self.conv_layer=(1,2)
        self.epoch=6
        self.batch_size=48
        self.img_width=40
        self.img_height=40
        self.img_depth=3
        self.channels_dimension=-1
        
        self.model = self.build_model()
        self.model.compile(loss='binary_crossentropy', optimizer='adam',
                           metrics=['accuracy'])
        self.model.summary()
        
    def build_model(self):
        def conv_layers(data, fltr, stride):
    		# the first block of 1x1 CONVs
            nomr1 = BatchNormalization(axis=self.channels_dimension, epsilon=2e-5, momentum=0.9)(data)
            act_relu1 = Activation("relu")(nomr1)
            conv1 = Conv2D(int(fltr * 0.25), (1, 1), use_bias=False,kernel_regularizer=l2(0.0001))(act_relu1)
            
    		# the second block 3x3 CONVs
            norm2 = BatchNormalization(axis=self.channels_dimension, epsilon=2e-5, momentum=0.9)(conv1)
            act_relu2 = Activation("relu")(norm2)
            conv2 = Conv2D(int(fltr * 0.25), (3, 3), strides=stride, padding="same", use_bias=False,kernel_regularizer=l2(0.0001))(act_relu2)
    		
            # the third block of 1x1 CONVs
            norm3 = BatchNormalization(axis=self.channels_dimension, epsilon=2e-5, momentum=0.9)(conv2)
            act_relu3 = Activation("relu")(norm3)
            conv3 = Conv2D(fltr, (1, 1), use_bias=False,kernel_regularizer=l2(0.0001))(act_relu3)
            
            reduction = Conv2D(fltr, (1, 1), strides=stride,use_bias=False, kernel_regularizer=l2(0.0001))(act_relu1)
    		# return the addition as the output reduction and conv3
            return add([conv3, reduction])
     
        input_shape=(self.img_width, self.img_height, self.img_depth)
        inputs = Input(shape=input_shape)
        conv_x = Conv2D(16, (1, 1))(inputs)
        conv_x=BatchNormalization(axis=self.channels_dimension, epsilon=2e-5, momentum=0.9)(conv_x)
        # convs = []
        for i in range(0, len(self.conv_layer)):
            # conv_x=inputs
            stride = (1, 1) if i == 0 else (2, 2)
            conv_x=conv_layers(conv_x,self.filters[i + 1],stride)

            for j in range(0, self.conv_layer[i] - 1):
                conv_x=conv_layers(conv_x,self.filters[i + 1],(1, 1))
                # convs.append(conv_x)
        # conv_x=Add()(convs)
        last_normalizer=BatchNormalization(axis=self.channels_dimension, epsilon=2e-5, momentum=0.9)(conv_x)
        last_activation=Activation("relu")(last_normalizer)
        max_pool=MaxPooling2D((8,8))(last_activation)
        glbal_max_pool=GlobalMaxPooling2D()(max_pool)
        prediction=Dense(1, kernel_regularizer=l2(0.0001))(glbal_max_pool)    
        act_last=Activation("sigmoid")(prediction)
        model = Model(inputs, act_last, name="malaria")
        return model
    
    def poly_decay(self,epoch):
        INIT_LR = 1e-1
        maxEpochs = self.epoch
        baseLR = INIT_LR
        power = 1.0
        # compute the new learning rate based on polynomial decay
        alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
        # return the new learning rate
        return alpha
    
    def fit(self, X, y):
        """Fit data"""
        
        self.num_train_samples = X.shape[0]
        X = X.reshape((self.num_train_samples, self.img_width, self.img_height, self.img_depth))
        # initialize the training data augmentation object
        trainAug = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1 / 255.0,
            rotation_range=20,
            zoom_range=0.05,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.05,
            horizontal_flip=False,
            fill_mode="nearest")
        # initialize the training generator
        train_generator = trainAug.flow(
            X,
            y,
            shuffle=True,
            batch_size=self.batch_size)
        #checkpoint
        callbacks = [LearningRateScheduler(self.poly_decay)]#
        
        self.model.fit_generator(train_generator,
            steps_per_epoch=X.shape[0] // self.batch_size,
            epochs=self.epoch, callbacks=callbacks)
        self.save(self.model)
        self.is_trained = True

    def predict(self, X):
        """Predict method"""
        num_test_samples = X.shape[0]
        X = X.reshape((num_test_samples, self.img_width, self.img_height, self.img_depth))
        # initialize the validation (and testing) data augmentation object
        testAug = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.0)
        # initialize the testing generator
        testGen = testAug.flow(
            X,
            shuffle=False,
            batch_size=self.batch_size)
        return self.model.predict_generator(testGen)
    def save(self,model):
        import time
        import os
        model_dir='../model/'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_name=os.path.join(model_dir,'malaria_patch_classifier_model.h5')
        model.save(model_name)
        print('INFO: model saved to {}'.format(model_name))
    
    def loadModel(self):
        from tf.keras.models import load_model
        print('INFO: Loading trained model from ../model/malaria_patch_classifier_model.h5')
        return load_model('../model/malaria_patch_classifier_model.h5')
        
    
# M_clf = model()
# -*- coding: utf-8 -*-

