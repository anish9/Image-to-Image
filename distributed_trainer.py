"""utilizes all visible GPU devices"""

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from utils import *
from glob import glob
import tensorflow as tf
from dataset import TRAIN,VAL,get_spec
from config import batch_size,epochs,save_freq,UPSCALE
from g_engine.rdnsr import *
from d_engine.discrim import *
spec = get_spec()


strategy = tf.distribute.MirroredStrategy()
DISTRIBUTED_train_DATASET = strategy.experimental_distribute_dataset(TRAIN)

with strategy.scope():
    metric_op     = tf.keras.losses.Reduction.NONE
    disc_optim    = tf.keras.optimizers.Adam(1e-4)
    gen_optim     = tf.keras.optimizers.Adam(1e-4)
    generator     = RRDNSR(upsample=UPSCALE,rdb_depth=8)
    h_,w_ = spec[0],spec[1]
    discriminator = discriminator(h_,w_)
    PSNR_         = PSNR_metric()
    loss_object1  = Custom_Loss(reduction=metric_op)
    loss_object2  = tf.keras.losses.BinaryCrossentropy(reduction=metric_op)
    loss_object3  = tf.keras.losses.BinaryCrossentropy(reduction=metric_op)
    loss_object4  = tf.keras.losses.MeanSquaredError(reduction=metric_op)
    def compute_loss1(labels, predictions):
        loss_func1  = loss_object1(labels, predictions)
        return tf.nn.compute_average_loss(loss_func1, global_batch_size=batch_size)
    def compute_loss2(labels, predictions):
        loss_func2  = loss_object2(labels, predictions)
        return tf.nn.compute_average_loss(loss_func2, global_batch_size=batch_size)
    def compute_loss3(labels, predictions):
        loss_func3  = loss_object3(labels, predictions)
        return tf.nn.compute_average_loss(loss_func3, global_batch_size=batch_size)
    def compute_loss4(labels, predictions):
        loss_func4  = loss_object4(labels, predictions)
        return tf.nn.compute_average_loss(loss_func4, global_batch_size=batch_size)
        
        
def train_step(low,high,add_disc):
    batch_size  = tf.shape(low)[0]
    if add_disc:
        print("TRAINING MODE : d+g")
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

    
    print("TRAINING MODE : g")
    with tf.GradientTape() as tape:
        generate     = generator(low)
        predictions2 = discriminator(generate)
        lables2      = tf.ones((batch_size,1))
        loss_op1 = compute_loss1(high,generate)
        loss_op2 = compute_loss2(lables2,predictions2)
        loss_op4 = compute_loss4(high,generate)
        loss_op = (0.5*loss_op1)+(0.5*loss_op4)+(0.5*loss_op2)
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


def trainer(epochs):
    PSNRS= []
    for epoch in range(epochs):
        for x,y in DISTRIBUTED_train_DATASET:
            dlos,glos,psn = distributed_train_step(x,y)

            PSNR_.reset_states()
            psnr_est = psn.numpy()/batch_size
            print(f"-----------Train steps : {epoch} TRAIN-PSNR : {round(psnr_est,2)}-----------")

            
#             
        print("Validation Results ....")
        for b,z in VAL:
            out = generator.predict_on_batch(b)
            val_metric = PSNR_val.update_state(z,out)
            val = PSNR_val.result().numpy()/batch_size
            print(f"PSNR : {round(val,2)}")
            PSNR_val.reset_states()
            print(f"saving weights at: >>>>> {round(val,4)}")
            generator.save_weights("P_g_text.h5")
            discriminator.save_weights("P_d_text.h5")
        write_viz_dist(VAL,generator,batch_size,4)
            

            
if __name__ == "__main__":
    trainer(epochs)
 
