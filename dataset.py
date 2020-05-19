"""DATA SET UTILS"""

import tensorflow as tf
from config import *

def normalize_image(lq,hq,norm):
	"""default norm -1 to 1"""
	simage = tf.cast(lq,tf.float32)/norm
	simage = simage - 1
	timage = tf.cast(hq,tf.float32)/norm
	timage = timage - 1
	return simage,timage

def reader(lq,hq,j):
	simage = tf.io.read_file(lq)
	simage = tf.io.decode_jpeg(simage,channels=3)
	simage = tf.image.resize(simage,(LOW_RESOLUTION,LOW_RESOLUTION))
	timage = tf.io.read_file(hq)
	timage = tf.io.decode_jpeg(timage,channels=3)
	timage = tf.image.resize(timage,(LOW_RESOLUTION*UPSCALE,LOW_RESOLUTION*UPSCALE))
	simage,timage = normalize_image(simage,timage,NORMALIZE)
	return simage,timage

def loader(lq,hq):
	simage,timage = reader(lq,hq,LOW_RESOLUTION)
	return simage,timage

