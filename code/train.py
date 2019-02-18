import argparse
import json
import os
import sys

import keras
import numpy as np

import functions
from functions import load_model, value_or_default
from dataset import generator

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

