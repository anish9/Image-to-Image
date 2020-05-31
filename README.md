# GENCED
your_config = (GAN+encoder+cedoder{decoder})+(GAN-encoder+cedoder{decoder})
* Train discriminator if you need, detach if you don't need.
* Train with content loss, pixel wise loss
* Intended for rapid research experiments

## Requirements
* Tensorflow 2.1 <
* opencv

## Train custom
### Select the dataformat in config file, the available formats are "AUTO" and "PAIRED".
#### AUTO:
* #### Just provide high resolution images path so it automatically downscales and trains, if needed add custom degradations
#### PAIRED:
* #### provide with high-resolution and low-resolution images which to be in paired format to train. 
* ####  set the format in config file and also adjust given parameters like epochs, batchsize as per your task.

``` 
data_template = AUTO
``` 
#### Define your model architectures in g_engine and d_engine model
``` 
#EXAMPLE CODE OF CALLING YOUR CUSTOM MODELS

from g_engine.rdnsr import *
from d_engine.discrim import *
generator     = RRDNSR(upsample=2,rdb_depth=8) #CUTSOM BUILT GENERATOR
discriminator = discriminator(h_,w_) #CUSTOM BUILT DISCRIMINATOR
``` 

#### Task Examples
##### * Image super-resolution
![alt text](https://github.com/anish9/Image_2_Image/blob/master/asset/i1.png)

##### * Image-colorization
![alt text](https://github.com/anish9/GENCED/blob/master/asset/pci.jpg)

> Todo
- [ ] add distributed data and training pipeline
- [ ] add more research papers in the pipeline
