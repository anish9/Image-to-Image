# Image_2_Image
Baseline framework to train any image to image model

## Requirements
* Tensorflow 2.1
* opencv

#### Example to train a custom image super resolution (2x) RDNSR generator with added discriminator (GAN loss)
* In config.py set your configs
* In Train.py line 28,29 import your models
```
from d_engine.discrim import *
from g_engine.rdnsr import *
```
* In default the model gets trained with entropy, perceptual, mse
* But, any loss can be activated or deactivated based on the task (more flexible functionality to be added)

#### Trained SR output with (RDNSR+GANloss)
![alt text](https://github.com/anish9/Image_2_Image/blob/master/asset/i1.png)

[] to do
