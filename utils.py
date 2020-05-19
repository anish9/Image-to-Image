"""custom metric losses"""

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model


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
            self.psnr_hub.assign_add(tf.reduce_sum(op))
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
