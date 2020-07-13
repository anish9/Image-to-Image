
PAIRED = { "train_lq":"/home/mia/anish/experiments/image_tasks/datasets/colo_3/t_gray/*jpg",
           "train_hq":"/home/mia/anish/experiments/image_tasks/datasets/colo_3/t_rgb/*jpg",
           "val_lq"  :"/home/mia/anish/experiments/image_tasks/datasets/colo_3/v_gray/*jpg",
           "val_hq"  :"/home/mia/anish/experiments/image_tasks/datasets/colo_3/v_rgb/*jpg",
           "low_w":64,"low_h":64,
          }

AUTO =    {"train_hq":"/home/mia/anish/experiments/image_tasks/datasets/colo_3/t_rgb/*jpg",
           "val_hq"  :"/home/mia/anish/experiments/image_tasks/datasets/colo_3/v_rgb/*jpg",
           "clip_dim":512,
           "patch_size":352}


batch_size                           = 4
save_freq                            = 10
epochs                               = 30
UPSCALE                              = 8
NORMALIZE                            = 127.5
DISTRIBUTED_FLAG                     = False
data_template                        = PAIRED


