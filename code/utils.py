import os

import keras
import numpy as np
from keras import backend as K
from keras.models import model_from_json


def save_model(model, directory, save_weights=True):
    architectureFile = os.path.join(directory, 'architecture.json')
    weightsFile = os.path.join(directory, 'weights.h5')

    # Initialize directory
    if(not os.path.isdir(directory)):
        os.makedirs(directory)

    # Save the model architecture
    with open(architectureFile, 'w') as f:
        f.write(model.to_json())

    # Save the weights
    if(save_weights):
        model.save_weights(weightsFile)

def load_model(directory, load_weights=True):
    architectureFile = os.path.join(directory, 'architecture.json')
    weightsFile = os.path.join(directory, 'weights.h5')

    # Model reconstruction from JSON file
    with open(architectureFile, 'r') as f:
        model = model_from_json(f.read())

    # Load weights into the new model
    if(load_weights):
        model.load_weights(weightsFile)

    return model

def get_model_memory_usage(batch_size, model):
    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes