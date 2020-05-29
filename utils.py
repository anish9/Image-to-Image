from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
import cv2



image_shape = (None,None,3)
vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)


class PSNR_metric(tf.keras.metrics.Metric):
    def __init__(self, name='psnr',**kwargs):
            super(PSNR_metric, self).__init__(name=name, **kwargs)
            self.psnr_hub = self.add_weight(name='psnr', initializer='zeros')
    def update_state(self, y_true, y_pred, sample_weight=None):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            op = tf.image.psnr(y_true,y_pred,max_val=1.0)
            self.psnr_hub.assign_add(tf.reduce_mean(op))
    def result(self):
        return self.psnr_hub



class Custom_Loss(tf.keras.losses.Loss):
    def __init__(self,reduction):
        super(Custom_Loss,self).__init__(reduction=reduction)
    
    def call(self,y_true,y_pred):
        loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
        loss_model.trainable = False
        func = tf.keras.losses.mean_squared_error(loss_model(y_true),loss_model(y_pred))
        return func
    
    
    
def write_viz(method,network,batchsize,image_count):
    for x,y in method.take(1):
        preds       = network.predict_on_batch(x)
        image_index = np.random.randint(1,batchsize,image_count)
        for q in range(len(image_index)):
            vis_image   = preds[q,:,:,:]
            vis_image   = vis_image+1
            vis_image   = vis_image*127.5
            vis_image   = cv2.cvtColor(vis_image,cv2.COLOR_BGR2RGB)
            vis_image   = vis_image.astype(np.int16)
            cv2.imwrite(str(q)+"_vis.png",vis_image)
