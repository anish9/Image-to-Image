""" MAIN CONFIGS """

"""SET YOUR CONFIGS HERE"""


LOW_RESOLUTION         = 140
UPSCALE                = 2
HIGH_RESOLUTION        = LOW_RESOLUTION*UPSCALE
NORMALIZE              = 127.5 
BATCH                  = 6
GLOBAL_BATCH_SIZE      = BATCH
EPOCH                  = 500

train_source_path      = "../DUMMY_DATA/TRAIN_LR/*png"
train_target_path      = "../DUMMY_DATA/TRAIN_HR/*png"
val_source_path        = "../DUMMY_DATA/VAL_LR/*png"
val_target_path        = "../DUMMY_DATA/VAL_HR/*png"


# BATCH_SIZE_PER_REPLICA = BATCH/strategy.num_replicas_in_sync
