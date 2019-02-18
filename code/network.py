import argparse
import os
import sys

import kapre
import keras
import keras.backend as K
from kapre.time_frequency import Spectrogram
from keras import Model
from keras.layers import (Add, BatchNormalization, Dense, Input, LeakyReLU,
                          MaxPooling2D, SeparableConv2D)

import functions
from functions import get_model_memory_usage, save_model, value_or_default


def ResidualConvBlock(n_conv=1, n_filters=64, kernel_size=3):
    def f(x):
        h = x
        for _ in range(n_conv):
            h = SeparableConv2D(filters=n_filters, kernel_size=kernel_size, padding='same', use_bias=False)(h)
            h = BatchNormalization()(h)
            h = LeakyReLU(alpha=0.3)(h)

        f = SeparableConv2D(filters=n_filters, kernel_size=1, padding='same', use_bias=False)(x)
        f = BatchNormalization()(f)
        f = LeakyReLU(alpha=0.3)(f)
        x = Add()([f, h])
        return x
    return f

def get_network(args):
    x_in = Input(shape=args['shape']) # Expected 2D array: (audio_channel, audio_length), TODO flip the dimensions!
    x = Spectrogram(n_dft=args['n_dft'], n_hop=int(args['n_dft']/2))(x_in) 
    
    for _ in range(args['conv']['n_blocks']):
        x = ResidualConvBlock(args['conv']['n_layers'], args['conv']['n_filters'], args['conv']['kernel_size'])(x)
        curr_shape = K.int_shape(x)
        if(curr_shape[1] > args['pool_size'][0] and curr_shape[2] > args['pool_size'][1]):
            x = MaxPooling2D(pool_size=args['pool_size'])(x)
    
    for _ in range(args['dense']['n_layers']):
        x = Dense(units=args['dense']['n_units'], use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.3)(x)
    
    x_out = Dense(units=args['n_genres'], activation='softmax')(x)

    model = Model(inputs=x_in, outputs=x_out)
    return model

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    if len(sys.argv) > -1:
        ap.add_argument('--name', required=False, help="Name of the model")
        ap.add_argument('--directory', required=False, help="Path to save the model")
        ap.add_argument('--save', required=True, help="Save the model? [0/1]")
        ap.add_argument('--batchsize', required=False, help="Batch size")
        ap.add_argument('--audiolen', required=False, help="Lenght of the input audios (in seconds)")
        ap.add_argument('--audioch', required=False, help="Channels of the input audio (1: Mono, 2: Stereo ...)")
        ap.add_argument('--samplingrate', required=False, help="Preferred audio sampling rate (default: 44100)")
        ap.add_argument('--ndft', required=False, help="Compute N point Spectrogram")
        ap.add_argument('--convblocks', required=False, help="Number of convolutional blocks")
        ap.add_argument('--convlayers', required=False, help="Number of convolutional layers per block")
        ap.add_argument('--convfilters', required=False, help="Number of convolutional filters")
        ap.add_argument('--convkernel', required=False, help="Size of convolutional kernel")
        ap.add_argument('--denselayers', required=False, help="Number of dense layers")
        ap.add_argument('--denseunits', required=False, help="Number of hidden units per dense layer")
    args = vars(ap.parse_args())

    save = int(args['save']) > 0
    assert(save and args['name'] is not None and args['directory'] is not None), "Save specified, but no model name or directory."

    network_args = {
        'shape': (value_or_default(args['audioch'], 1), value_or_default(args['audiolen']*args['samplingrate'], 44100)),
        'n_dft': value_or_default(args['ndft'], 512),
        'conv': {
            'n_blocks': value_or_default(args['convblocks'], 1),
            'n_layers': value_or_default(args['convlayers'], 1),
            'n_filters': value_or_default(args['convfilters'], 64),
            'kernel_size': value_or_default(args['convkernel'], 3) 
        },
        'dense': {
            'n_layers': value_or_default(args['denselayers'], 1),
            'n_units': value_or_default(args['denseunits'], 1024)
        }
    }

    model = get_network(network_args)
    model.summary()
    print("Memory usage:", get_model_memory_usage(model, int(args['batchsize'])), " GB")

    if(save):
        output_directory = os.path.join(args['directory'], args['name'])
        save_model(model, output_directory)
        print('Model saved to ', output_directory)
