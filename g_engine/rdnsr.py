
"""build : pass"""
  
import os
import cv2
import numpy as np
from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras.layers import add
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.optimizers import Nadam,Adam
from tensorflow.keras.initializers import Initializer



def conv_global(x,t,stride=False):
    xin = Conv2D(64,3,padding="same",name="convg_"+str(t))(x)
    xin = BatchNormalization(axis=-1)(xin)
    xin = Activation("relu")(xin)
    
    if stride:
        xin = Conv2D(64,3,padding="same",strides=stride,name="convg_"+str(t))(x)
        xin = BatchNormalization(axis=-1)(xin)
        xin = Activation("relu")(xin)
    
    return xin


def RDBlocks(x,name , count = 6 , g=32):
    li = [x]
    pas = Conv2D(filters=g, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu' , name = name+'_conv1')(x)

    for i in range(2 , count+1):

        pas = Conv2D(filters=g, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu', name = name+'_conv'+str(i))(pas)
        li.append(pas)
    # feature extractor from the dense net
  
    out = Concatenate(axis = -1)(li)
    feat = Conv2D(filters=64, kernel_size=(1,1), strides=(1, 1), padding='same',activation='relu' , name = name+'_Local_Conv')(out)

    feat = Add()([feat , x])
    return feat

def tensor_depth_to_space(imag,block_size,names):
    x = tf.nn.depth_to_space(imag,block_size,name=names)
    return x

def tf_subpixel_conv(tensor,block_size,filters):
    x = Conv2D(filters,3,strides=(1,1),padding="same")(tensor)
    x = Lambda(lambda x : tensor_depth_to_space(x,block_size,names="subpixel_conv"))(x)
    x  = PReLU(shared_axes=[1, 2])(x)
    return x




def RRDNSR(upsample=4,rdb_depth=12):
    
    inpu = Input(shape=(None,None,3))
    xin1 = conv_global(inpu,1)
    xin2 = conv_global(xin1,2,stride=2)
    
    global_list = [xin2]
    for e in range(1,rdb_depth+1):
        r1 = RDBlocks(xin2,"rdbs"+str(e))
        global_list.append(r1)
        
        
    concs = Concatenate(axis=-1)(global_list)    
    concs = Conv2D(64,1,padding="same",name="glist_ups_conv1")(concs)
    
    concs = Activation("relu")(concs)
    concs = Conv2D(64,3,padding="same",name="glist_ups_conv3")(concs)
    
    concs = tf_subpixel_conv(concs,2,256)
    concs = add([concs,xin1])
    
    global_merge = concs
    upsample_seg = tf_subpixel_conv(global_merge,2,128)
    if upsample == 2: 
        s2 = upsample_seg
        fout = Conv2D(3,9,padding="same",activation="linear")(s2)
    if upsample == 4:
        upsample_seg = tf_subpixel_conv(upsample_seg,2,32)
        fout = Conv2D(3,9,padding="same",activation="linear")(upsample_seg)
    network = Model(inputs=inpu,outputs=fout,name="RDNSR_"+str(upsample)+str(rdb_depth))
    return network



