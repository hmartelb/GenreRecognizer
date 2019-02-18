import kapre
import keras
import keras.backend as K
from kapre.time_frequency import Spectrogram
from keras import Model
from keras.layers import (Add, BatchNormalization, Dense, Input, LeakyReLU,
                          MaxPooling2D, SeparableConv2D)


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
    x_in = Input(shape=args['shape'])
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
    args = {
        'shape': (44100, 1),
        'n_dft': 512,
        'conv': {
            'n_blocks': 1,
            'n_layers': 1,
            'n_filters': 64,
            'kernel_size': 3
        },
        'dense': {
            'n_layers': 1,
            'n_units': 1024
        }
    }

    model = get_network(args)
    model.summary()