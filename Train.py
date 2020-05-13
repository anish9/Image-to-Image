"""BUILD : Beta"""
"""PROP  : 2.0 <"""
"""lib : TF2.1 """
"""SOLVER : DISTRIBUTED"""

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import cv2
import numpy as np
import tensorflow as tf
from glob import glob
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard

"""import your Model blocks here"""
from dataset import *
from utils import *
from g_engine.rdnsr import *
from d_engine.discrim import *

strategy = tf.distribute.MirroredStrategy()

train_lq = sorted(glob(train_source_path))
train_hq = sorted(glob(train_target_path))
val_lq   = sorted(glob(val_source_path))
val_hq   = sorted(glob(val_target_path))


BATCH_SIZE = GLOBAL_BATCH_SIZE
BUFFER = 15
BUFFER_FETCH  = 1
TRAIN = tf.data.Dataset.from_tensor_slices((train_lq,train_hq))
VAL   = tf.data.Dataset.from_tensor_slices((val_lq,val_hq))
TRAIN = TRAIN.map(loader,num_parallel_calls=tf.data.experimental.AUTOTUNE)
VAL   = VAL.map(loader,num_parallel_calls=tf.data.experimental.AUTOTUNE)
TRAIN = TRAIN.batch(BATCH_SIZE,drop_remainder=True).shuffle(BUFFER)
VAL   = VAL.batch(BATCH_SIZE,drop_remainder=True).shuffle(BUFFER)
DISTRIBUTED_train_DATASET = strategy.experimental_distribute_dataset(TRAIN)




    
with strategy.scope():
    metric_op     = tf.keras.losses.Reduction.NONE
    disc_optim    = tf.keras.optimizers.Adam(1e-4)
    gen_optim     = tf.keras.optimizers.Adam(1e-5)
    generator     = RRDNSR(upsample=UPSCALE,rdb_depth=6)
#     generator.load_weights("CHECKPOINT1.h5")
    discriminator = discriminator(HIGH_RESOLUTION,HIGH_RESOLUTION)
    PSNR_         = PSNR_metric()
    loss_object1  = Custom_Loss(reduction=metric_op)
    loss_object2  = tf.keras.losses.BinaryCrossentropy(reduction=metric_op)
    loss_object3  = tf.keras.losses.BinaryCrossentropy(reduction=metric_op)
    loss_object4  = tf.keras.losses.MeanSquaredError(reduction=metric_op)
    def compute_loss1(labels, predictions):
        loss_func1  = loss_object1(labels, predictions)
        return tf.nn.compute_average_loss(loss_func1, global_batch_size=GLOBAL_BATCH_SIZE)
    def compute_loss2(labels, predictions):
        loss_func2  = loss_object2(labels, predictions)
        return tf.nn.compute_average_loss(loss_func2, global_batch_size=GLOBAL_BATCH_SIZE)
    def compute_loss3(labels, predictions):
        loss_func3  = loss_object3(labels, predictions)
        return tf.nn.compute_average_loss(loss_func3, global_batch_size=GLOBAL_BATCH_SIZE)
    def compute_loss4(labels, predictions):
        loss_func4  = loss_object4(labels, predictions)
        return tf.nn.compute_average_loss(loss_func4, global_batch_size=GLOBAL_BATCH_SIZE)
    
def train_step(low,high,add_disc):
    batch_size  = tf.shape(low)[0]
    if add_disc:
        print("TRAINING MODE : DISCRIMATOR ADDED")
        generate_sr = generator(low)
        true_labels = tf.ones((batch_size,1))
        fake_labels = tf.zeros((batch_size,1))

        with tf.GradientTape() as tape:
            predictions_true = discriminator(high)
            predictions_fake = discriminator(generate_sr)
            d_true_loss = compute_loss3(true_labels, predictions_true)
            d_fake_loss = compute_loss3(fake_labels, predictions_fake)
            d_loss = d_true_loss+d_fake_loss
        grads = tape.gradient(d_loss, discriminator.trainable_weights)
        disc_optim.apply_gradients(zip(grads, discriminator.trainable_weights))

    
    print("TRAINING MODE : END-TO-END")
    with tf.GradientTape() as tape:
        generate     = generator(low)
        predictions2 = discriminator(generate)
        lables2      = tf.ones((batch_size,1))
#         loss_op      = LOSSES(high,generate,predictions2,lables2)
        loss_op1 = compute_loss1(high,generate)
        loss_op2 = compute_loss2(lables2,predictions2)
        loss_op4 = compute_loss4(high,generate)
        loss_op = 0.5*loss_op1+0.5*loss_op2+0.5*loss_op4
    grads = tape.gradient(loss_op, generator.trainable_weights)
    gen_optim.apply_gradients(zip(grads, generator.trainable_weights))
    
    PSNR_state = PSNR_.update_state(high,generate)
    
    if add_disc:
        return d_loss,loss_op,PSNR_.result()
    else:
        return 0.0,loss_op,PSNR_.result()
    
@tf.function
def distributed_train_step(low,high):
    per_replica_losses = strategy.experimental_run_v2(train_step, args=(low,high,True))
    return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses,
                         axis=None)


PSNR_val = PSNR_metric()


def Trainer(epochs):
    for epoch in range(30):
        for x,y in DISTRIBUTED_train_DATASET:
            dlos,glos,psn = distributed_train_step(x,y)

            PSNR_.reset_states()
            psnr_est = psn.numpy()
            print(f"Train steps : {epoch} TRAIN-PSNR : {psnr_est}")
            generator.save_weights("CHECKPOINT.h5")
        print("Validation Results ....")
        for b,z in VAL:
            out = generator.predict_on_batch(b)
            val_metric = PSNR_val.update_state(z,out)
            print(f"PSNR : {PSNR_val.result().numpy()}")
            PSNR_val.reset_states()



if __name__ = "__main__":
	Trainer(EPOCH,"ckpt001.h5")
