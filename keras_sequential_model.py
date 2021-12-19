

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



def keras_sequential_model_create(input_shape, n_hidden, n_neurons):
    '''
    The function used to build the architecture of sequential model but based only on keras,
    once we get the best hyperparameters, build the model with these hyperparameters.

    Argument:
        n_hidden      : How many hidden layers we need
        n_neurons     : For each hidden layer which number of neurons we need (fixed number for all hidden layers)
        input_shape   : The number of features we have defined by the image (width * height * 3 for rgb)
    
    return:
        model: The architecture of the model we have built
    '''
    
    # Create the Sequential model
    model = keras.models.Sequential()
    
    # define the shape of the input layer from the features we have for each image
    model.add(keras.layers.Input(shape=input_shape))
    
    # loop over hidden layers
    for i in range(n_hidden):
        # for each hidden layer pass the number of neurons for this layer
        model.add(keras.layers.Dense(n_neurons, activation='relu'))
    
    # at the end handle the output layer as we just need to predict image belong to me or not so its just one unit
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    return model




def keras_sequential_model_compile(model, learning_rate):
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
                  metrics=['acc'])
    return model


def call_backs(run_log_dir,file_path, patience=10, monitor="val_loss", restore_best_weights=True, ):
    
    
    cb_early_stop   = keras.callbacks.EarlyStopping(patience=patience,monitor=monitor,
                                                    restore_best_weights=True)
    cb_tensor_board = keras.callbacks.TensorBoard(run_log_dir)
    model_save_dir = MODELS_DIR + file_path
    cb_check_point = keras.callbacks.ModelCheckpoint(model_save_dir)
    callbacks_list = [cb_early_stop, cb_check_point, cb_tensor_board]
    return callbacks_list
