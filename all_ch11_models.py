

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import os
import cv2
import sys
import matplotlib.image as mpimg
from PIL import Image
import scipy
from scipy import ndimage
from scipy.stats import reciprocal
import io
import uvicorn
import numpy as np
import nest_asyncio
from enum import Enum
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import time
import pickle
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

# Keras APIs
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras import models
from tensorflow import keras
import tensorflow as tf

from configs import *




############# Sequential Models #############

def no_batch_seqential_model_create(input_shape, n_hidden, n_neurons, kernel_initializer, activation):
	
	model = keras.models.Sequential()
	model.add(keras.layers.Input(shape=input_shape))
	for i in range(n_hidden):
	    model.add(keras.layers.Dense(n_neurons, activation=activation,
	     kernel_initializer=kernel_initializer))
	# Output layer
	model.add(keras.layers.Dense(1, activation="sigmoid", kernel_initializer="glorot_uniform"))
	return model

def with_batch_after_seqential_model_create(input_shape, n_hidden, n_neurons, kernel_initializer, activation):

    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    
    for i in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation=activation, kernel_initializer=kernel_initializer))
        model.add(keras.layers.BatchNormalization())
#         model.add(batch_norm)
    # Output layer
    model.add(keras.layers.Dense(1, activation="sigmoid", kernel_initializer="glorot_uniform"))
    return model

def with_batch_before_seqential_model_create(input_shape, n_hidden, n_neurons, kernel_initializer, activation):

    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    
    for i in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, kernel_initializer=kernel_initializer))
        model.add(keras.layers.BatchNormalization())
        model.add(activation)
#         model.add(batch_norm)
    # Output layer
    model.add(keras.layers.Dense(1, activation="sigmoid", kernel_initializer="glorot_uniform"))
    return model


def no_batch_seqential_model_compile(model, optimizer):
	model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=optimizer,
	              metrics=['accuracy'])
	return model


def ch11_model_hyper_parameters(n_hidden, api_type, img_size):
    
    num_of_hidden_layers = "_num of hidden lr=" + str(n_hidden)
    params               = img_size + api_type + num_of_hidden_layers
    
    return params

def ch11_call_backs(run_log_dir, patience=10, monitor="val_loss"):
    #restore_best_weights reverts to the weights that obtained the highest monitored score value
    cb_early_stop   = keras.callbacks.EarlyStopping(monitor=monitor, min_delta=0, patience=patience,
                                                    verbose=2, mode='auto', restore_best_weights=True)
    cb_tensor_board = keras.callbacks.TensorBoard(run_log_dir)
    callbacks_list = [cb_early_stop, cb_tensor_board]
    return callbacks_list



# loss, metrics, optimizer, callbacks, epochs,  batch_size, X_train, y_train, X_val, y_val
# def momentum_relu_he_normal_no_batch():


# def momentum_nesterov_relu_he_normal_no_batch():

# def rmsprop_relu_he_normal_no_batch():

# def adam_relu_he_normal_no_batch():


# def sgd_relu_he_normal_with_batch(batch_after=Treu):

# def momentum_relu_he_normal_with_batch(batch_after=Treu):


# def momentum_nesterov_relu_he_normal_with_batch(batch_after=Treu):

# def rmsprop_relu_he_normal_with_batch(batch_after=Treu):

# def adam_relu_he_normal_with_batch(batch_after=Treu):


def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * .1 **(epoch / s)
    return exponential_decay_fn


# def performance_lr():
# 	lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=.5, patience=5)
# 	return lr_scheduler