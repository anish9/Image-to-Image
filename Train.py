from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from utils import *
from glob import glob
import tensorflow as tf
from dataset import TRAIN,VAL,get_spec
from config import batch_size,epochs,save_freq
from g_engine.rdnsr import *
from d_engine.discrim import *
spec = get_spec()



generator     = RRDNSR(upsample=2,rdb_depth=8) #subjected to builded and imported models
h_,w_         = spec[0],spec[1]
discriminator = discriminator(h_,w_)

disc_optim    = tf.keras.optimizers.Adam(1e-4)
gen_optim     = tf.keras.optimizers.Adam(1e-5)
generator.load_weights("div2k_gen.h5")
loss_object1  = Custom_Loss(reduction=tf.keras.losses.Reduction.AUTO)
loss_object2  = tf.keras.losses.BinaryCrossentropy()
loss_object4  = tf.keras.losses.MeanSquaredError()
PSNR_         = PSNR_metric()

def train_step(low,high,add_disc,mse_only=False,vgg_only=False):
    """use custom loss if needed,(represent if customized loss objects and supply the loss func in required computation
       modify traning strategy simply by flagging (add_disc) argument
        ARGUMENTS: 
        low  - low_res image
        high - high_res image
        add_disc -  adds GAN method to train the model(adds a supplied discriminator to compute GAN loss)
                    ("Note" : this is simple gan and not computes as PATCH GAN)
        mse_only - trains generator only with mse only
        vgg_only - trains generator only with perceptual loss only
    """
    batch_size  = tf.shape(low)[0]
    if add_disc:
        generate_sr = generator(low)
        true_labels = tf.ones((batch_size,1))
        fake_labels = tf.zeros((batch_size,1))
        with tf.GradientTape() as tape:
            predictions_true = discriminator(high)
            predictions_fake = discriminator(generate_sr)
            d_true_loss = loss_object2(true_labels, predictions_true)
            d_fake_loss = loss_object2(fake_labels, predictions_fake)
            d_loss = d_true_loss+d_fake_loss
        grads = tape.gradient(d_loss, discriminator.trainable_weights)
        disc_optim.apply_gradients(zip(grads, discriminator.trainable_weights))

    with tf.GradientTape() as tape:
        generate     = generator(low)
        if add_disc:
            predictions2 = discriminator(generate)
            lables2      = tf.ones((batch_size,1)) 
            loss_op1 = loss_object1(high,generate)
            loss_op2 = loss_object2(lables2,predictions2)
            loss_op4 = loss_object4(high,generate)
            loss_op = loss_op1+loss_op4+loss_op2
            if mse_only:
                loss_op = loss_op4+loss_op2
            if vgg_only:
                loss_op = loss_op1+loss_op2
        else:
            loss_op1 = loss_object1(high,generate)
            loss_op4 = loss_object4(high,generate)
            loss_op  = loss_op1+loss_op4
            if mse_only:
                loss_op = loss_op4
            if vgg_only:
                loss_op = loss_op1
    grads = tape.gradient(loss_op, generator.trainable_weights)
    gen_optim.apply_gradients(zip(grads, generator.trainable_weights))
    PSNR_state = PSNR_.update_state(high,generate)
    if add_disc:
        return d_loss,loss_op,PSNR_.result()
    else:
        return 0.0,loss_op,PSNR_.result()

    
def TRAIN_SR(epochs,discriminator_=True,vgg_only=False,mse_only=False,save_freq=2,viz_count=3):
    for e in range(epochs):
        for x,y in TRAIN:
            disc_loss,gen_loss,metric = train_step(x,y,discriminator_,vgg_only,mse_only)
            print(f"Training step : {e} | disc_loss : {disc_loss} gen_loss : {gen_loss} PSNR_LOSS : {metric}")
            PSNR_.reset_states()
            
        for b,z in VAL:
            out = generator.predict_on_batch(b)
            val_metric = PSNR_.update_state(z,out)
            val = PSNR_.result().numpy()
            print(f"VALIDATION PSNR : {round(val,2)}")
            PSNR_.reset_states()
            
        if e%save_freq == 0:
            print(f"saving weights")
            generator.save_weights("div2k_gen.h5")
            discriminator.save_weights("div2k_dis.h5")
            write_viz(VAL,generator,batch_size,viz_count)



            
if __name__ == "__main__":            
    TRAIN_SR(epochs,discriminator_=True,save_freq,viz_count=5)
