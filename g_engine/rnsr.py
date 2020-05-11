"""build : fail"""

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


def conv_lrelu(x,filters,tag):
    fw1 = Conv2D(filters,(3,3),padding="same",kernel_initializer="he_normal",name="he_norm_conv_"+str(tag))(x)
    fw1 = LeakyReLU(alpha=0.2,name="he_norm_lrelu_"+str(tag))(fw1)
    return fw1

def convT_lrelu(x,filters,stride_=(1,1),tag=None):
    fw1 = Conv2DTranspose(filters,(2,2),padding="same",strides=stride_,kernel_initializer="he_normal",name="he_norm_convT_"+str(tag))(x)
    return fw1

def convT_apply(inps,filters,stp,tag,gtag=None):
    x = convT_lrelu(inps,filters,stp,tag="Transpose_apply_"+str(tag)+str(gtag))
    x = Activation("relu")(x)
    return x

def conv_apply(inps,filters,units,k_uint=None,gtag=None):
    Features = []
    for k in range(1,units+1):
        extracts = conv_lrelu(inps,filters,tag=str(k_uint)+str(k))
        Features.append(extracts)
    pix = Concatenate(axis=-1)(Features)
    pix = LeakyReLU(alpha=0.2,name = "units_glob_carry_acti1"+str(gtag))(pix)
    pix = Conv2D(filters,(3,3),padding="same",name="unit_glob_"+str(gtag))(pix)
    pix_inp = Conv2D(filters,(3,3),padding="same",name="units_glob_carry_"+str(gtag))(inps)
    pix_map = Add()([pix_inp,pix])
    return pix_map

def RITCHE():
    inp_fe = Input(shape=(None,None,3))
    filter_ = 2
    out_channel = 3
    a1  = conv_apply(inp_fe,filter_*2,3,1,1)
    a1d = convT_apply(a1,filter_*2,stp=(2,2),tag=1,gtag=1)
    a1u = convT_apply(inp_fe,filter_*2,stp=(2,2),tag=2,gtag=1)
    a1_ud = Add()([a1u,a1d])
    a2  = conv_apply(a1_ud,filter_,3,2,2)
    a2d = convT_apply(a2,filter_,stp=(1,1),tag=1,gtag=2)
    a2u = convT_apply(a1_ud,filter_,stp=(1,1),tag=2,gtag=2)
    a2_ud = Add()([a2u,a2d])
    a3  = conv_apply(a2_ud,filter_//2,3,3,3)
    a3d = convT_apply(a3,filter_//2,stp=(1,1),tag=1,gtag=3)
    a3u = convT_apply(a2_ud,filter_//2,stp=(1,1),tag=2,gtag=3)
    a3_ud = Add()([a3u,a3d])
    feature_interact = Concatenate(axis=-1)([a1_ud,a2_ud,a3_ud])
    feature_interact = Conv2D(filter_,3,padding="same")(feature_interact)
    feature_interact = Activation("relu")(feature_interact)
    OUTPUT_2X = Conv2D(out_channel,7,padding="same",activation="linear")(feature_interact)
    n = Model(inp_fe,OUTPUT_2X)
    return n

