# Image_2_Image
Baseline framework to train any image to image model

## Requirements
* Tensorflow 2.1 <
* opencv

## Train custom
* Select the dataformat in config file, the available formats are "AUTO" and "PAIRED".
#### AUTO:
* #### Just provide high resolution images path so automatically downscales and train the network,if needed added custom degradations
#### PAIRED:
* #### provide with high-resolution and low-resolution images which to be in paired format to train. 


#### Trained SR output with (RDNSR+GANloss)
![alt text](https://github.com/anish9/Image_2_Image/blob/master/asset/i1.png)

