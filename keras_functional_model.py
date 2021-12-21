
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
from keras_sequential_model import *

def keras_functional_model_one_input_create(input_shape, n_hidden, n_neurons):
    '''
    The function used to build the architecture of functional model.
    once we get the best hyperparameters, build the model with these hyperparameters.

    Argument:
        n_hidden      : How many hidden layers we need
        n_neurons     : For each hidden layer which number of neurons we need (fixed number for all hidden layers)
        input_shape   : The number of features we have defined by the image (width * height * 3 for rgb)

    return:
        model: The architecture of the model we have built
    '''
    input_   = keras.layers.Input(shape=input_shape)
    temp = input_

    for hidden in range(n_hidden):
        hidden_ = keras.layers.Dense(n_neurons, activation='relu')(temp)
        temp = hidden_

    concat   = keras.layers.concatenate([input_, hidden_])
    output   = keras.layers.Dense(1, activation='sigmoid')(concat)
    model    = keras.Model(inputs=[input_], outputs=[output])

    return model


def keras_functional_model_one_input_compile(model, learning_rate):
    '''
    The architecture of the model we have built need to be compiled to define which loss function the model will use,
    beside of which optimization algorithm to update the weights to minimize the loss function. 
    
    Argument:
        model         : The model we have built
        learning_rate : the best learning rate we have got after searching
    return:
        model : The architecture of the model we have built and compiled 
    '''
    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.SGD(lr=learning_rate),
                  metrics=['accuracy'])
    return model


def train_functional_model_with_best_params(rnd_search_cv,  X_train, y_train, X_val, y_val,
                        api_type="one_input_functional_api",  img_size="img_size_100*100*3_"):
    # Which size of images we train on and which keras api we use

    img_size      = img_size
    api_type      = api_type
    input_shape   = X_train.shape[1]
    
    # Retrieve the best hyper parameter
    learning_rate  =  rnd_search_cv.best_params_['learning_rate']
    n_hidden       =  rnd_search_cv.best_params_['n_hidden']
    n_neurons      =  rnd_search_cv.best_params_['n_neurons']
    
    
    epochs   = 100
    patience = 10
    model_hyper_params  = model_hyper_parameters(img_size, api_type, n_hidden, learning_rate, epochs)

    run_log_dir    = tensor_logs_dir(TENSOR_DIR, model_hyper_params)
    callbacks_     = call_backs(run_log_dir, patience)


    if api_type == "one_input_functional_api":
        model   = keras_functional_model_one_input_create(input_shape, n_hidden, n_neurons)
    else:
        pass


    model   = keras_functional_model_one_input_compile(model, learning_rate)
    
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val),
                   callbacks=callbacks_)
    return model, history, model_hyper_params