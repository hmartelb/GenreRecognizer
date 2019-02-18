import keras
import numpy as np
import functions
import argparse
import os
import sys
from functions import load_model

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    if len(sys.argv) > -1:
        ap.add_argument('--modeldir', required=True, help="Path to the model")
        
    args = vars(ap.parse_args())

