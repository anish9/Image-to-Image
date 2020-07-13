import tensorflow as tf
from config import *
from glob import glob

temp = data_template
BATCH_SIZE = batch_size
BUFFER = 1024

def normalize_image(lq,hq,norm):
    simage = tf.cast(lq,tf.float32)/norm
    simage = simage - 1
    timage = tf.cast(hq,tf.float32)/norm
    timage = timage - 1
    return simage,timage


def paired(lq,hq,lrh,lrw):
    simage = tf.io.read_file(lq)
    simage = tf.io.decode_png(simage,channels=3)
    simage = tf.image.resize(simage,(lrh,lrw))
    timage = tf.io.read_file(hq)
    timage = tf.io.decode_png(timage,channels=3)
    timage = tf.image.resize(timage,(lrh*2,lrw*2))
    simage,timage = normalize_image(simage,timage,NORMALIZE)  
    return simage,timage

def loader_paired(lq,hq):
    simage,timage = paired(lq,hq,temp["low_h"],temp["low_w"])
    return simage,timage


def auto(hqp,clip,hqh,hqw):
    simage  = tf.io.read_file(hqp)
    simage  = tf.io.decode_jpeg(simage,channels=3)
    simage  = tf.image.resize(simage,size=[clip,clip])
    crop_hq = tf.image.random_crop(simage, [hqh,hqw,3], seed=None, name=None)
    crop_hq = tf.cast(crop_hq,tf.float32)
    crop_lq = tf.image.resize(crop_hq,size =[hqh//UPSCALE,hqw//UPSCALE])
    simage,timage = normalize_image(crop_lq,crop_hq,NORMALIZE) 
    return simage,timage

def loader_auto(hq):
    simage,timage = auto(hq,temp["clip_dim"],temp["patch_size"],temp["patch_size"])
    return simage,timage


if temp == PAIRED:
    train_lq = sorted(glob(temp["train_lq"])) 
    train_hq = sorted(glob(temp["train_hq"]))
    val_lq   = sorted(glob(temp["val_lq"]))
    val_hq   = sorted(glob(temp["val_hq"]))
    TRAIN = tf.data.Dataset.from_tensor_slices((train_lq,train_hq))
    VAL   = tf.data.Dataset.from_tensor_slices((val_lq,val_hq))
    TRAIN = TRAIN.map(loader_paired,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    VAL   = VAL.map(loader_paired,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    TRAIN = TRAIN.batch(BATCH_SIZE,drop_remainder=True).shuffle(BUFFER)
    VAL   = VAL.batch(BATCH_SIZE,drop_remainder=True).shuffle(BUFFER)


if temp == AUTO:
    train_hq  = sorted(glob(temp["train_hq"]))
    val_hq    = sorted(glob(temp["val_hq"]))
    TRAIN = tf.data.Dataset.from_tensor_slices(train_hq)
    VAL   = tf.data.Dataset.from_tensor_slices(val_hq)
    TRAIN = TRAIN.map(loader_auto,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    VAL   = VAL.map(loader_auto,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    TRAIN = TRAIN.batch(BATCH_SIZE,drop_remainder=True).shuffle(BUFFER)
    VAL   = VAL.batch(BATCH_SIZE,drop_remainder=True).shuffle(BUFFER)

    
def get_spec():
    x,y  = TRAIN.element_spec
    height_shape,width_shape = y.shape[1],y.shape[2]
    return height_shape,width_shape






	







        
