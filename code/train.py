import argparse
import datetime
import json
import multiprocessing
import os
import sys

import keras
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import functions
from dataset import generator
from functions import load_model, value_or_default

def load_filenames(directory):
    return []

def save_history(history, outputdir):
    return

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    if len(sys.argv) > -1:
        ap.add_argument('--modeldir', required=True, help="Path to the model")
        ap.add_argument('--datasetdir', required=False, help="Path to the dataset")
        ap.add_argument('--nepochs', required=False, help="Number of training epochs")
        ap.add_argument('--lr', required=False, help="Learning rate")
    args = vars(ap.parse_args())
    model_directory = args['modeldir']
    dataset_directory = value_or_default(args['datasetdir'], os.path.join('..', 'dataset'))
    nepochs = value_or_default(args['nepochs'], 100)
    lr = value_or_default(float(args['lr']), 1e-3) 

    architecture_file = os.path.join(model_directory, 'architecture.json')
    parameters_file = os.path.join(model_directory, 'parameters.json')
    assert(os.path.isfile(architecture_file) and os.path.isfile(parameters_file)), 'No architecture or parameters found in the specified directory.'
    
    model = load_model(model_directory)
    parameters = json.load(parameters_file)

    date = datetime.datetime.now().strftime("%Y%m%dT_%H%M%S")
    session_directory = os.path.join(model_directory, f"session_{date}_epochs_{nepochs}")
    if(not os.path.isdir(session_directory)):
        os.makedirs(session_directory)

    training_generator = generator(filenames=load_filenames(os.path.join(dataset_directory, 'training')), batch_size=parameters['batchsize'], dim=[*parameters['shape']])
    validation_generator = generator(filenames=load_filenames(os.path.join(dataset_directory, 'validation')), batch_size=parameters['batchsize'], dim=[*parameters['shape']])

    
    checkpoint = ModelCheckpoint(os.path.join(session_directory, "weights.h5"), monitor='val_acc', verbose=1, save_best_only=True, mode='min')
    earlystop = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, mode='auto', min_delta=0.01, cooldown=0, min_lr=1e-9)
    callbacks_list = [checkpoint, earlystop, reduce_lr]

    model.compile(optimizer=keras.optimizers.Adam(lr=lr), loss='categorical_crossentropy')

    history = model.fit_generator(generator=training_generator, 
                                    validation_data=validation_generator, 
                                    epochs=nepochs, 
                                    callbacks=callbacks_list, 
                                    verbose=1, 
                                    use_multiprocessing=True, 
                                    workers=int(multiprocessing.cpu_count()-1)
                                )

    save_history(history, session_directory)
