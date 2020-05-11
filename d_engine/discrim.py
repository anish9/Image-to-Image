import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def conv_bn(inp,i):
    x = Conv2D(32*i,3,strides=2)(inp)
    x = LeakyReLU()(x)
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D()(x)
    return x

def discriminator(height,width):
    inp = Input(shape=(height,width,3))
    c1  = Conv2D(32,3)(inp)
    c1  = LeakyReLU()(c1)
    for apply in [1,2,4,8,16,32,32]:
        depth = conv_bn(c1,apply)
    d1 = GlobalAveragePooling2D()(depth)
    d1 = Dense(300)(d1)
    d1 = LeakyReLU()(d1)
    d1 = Dense(1)(d1)
    d1 = Activation("sigmoid")(d1)
    mods = Model(inp,d1)
    return mods
