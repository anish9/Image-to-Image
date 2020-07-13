

PAIRED = { "train_lq":"../textdataset/Dataset/train/LR/*png",
           "train_hq":"../textdataset/Dataset/train/HR/*png",
           "val_lq"  :"../textdataset/Dataset/val/LR/*png",
           "val_hq"  :"../textdataset/Dataset/val/HR/*png",
           "low_w":150,"low_h":100,
          }

AUTO =    {"train_hq":"../DIV2K/TRAIN_HR/*png",
           "val_hq"  :"../DIV2K/VAL_HR/*png",
           "clip_dim":512,
           "patch_size":352
          }

batch_size                           = 8
save_freq                            = 10
epochs                               = 1011
UPSCALE                              = 2
NORMALIZE                            = 127.5
DISTRIBUTED_FLAG                     = False
data_template                        = AUTO


