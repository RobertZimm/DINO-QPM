import math


EPOCHS = 150 # changed from 150
STEP_LR = 30
START_LR = 0.005
STEP_LR_DECAY = 0.4
WEIGHT_DECAY = 0.0005

EPOCHS_FT = 40
STEP_LR_FT = 10
START_LR_FT = 1e-4
STEP_LR_DECAY_FT = 0.4
WEIGHT_DECAY_FT = 0.0005


class EvaluatedDict:
    def __init__(self, d, func):
        self.dict = d
        self.func = func

    def __getitem__(self, key):
        return self.dict[self.func(key)]

# Change training params here
# order: lr, weight_decay, step_lr, step_lr_gamma, epochs
dense_params = EvaluatedDict({False: [START_LR, WEIGHT_DECAY, STEP_LR, STEP_LR_DECAY, EPOCHS],
                              True: [None, None, None, None, None],}, 
                              lambda x: x == "ImageNet")


# Only needed for training on imagenet
def calculate_lr_from_args(epochs: int, 
                           step_lr: int, 
                           start_lr: float, 
                           step_lr_decay: float):
    # Gets the final learning rate 
    # after dense training with step_lr_schedule.
    n_steps = math.floor((epochs - step_lr) / step_lr)
    final_lr = start_lr * step_lr_decay ** n_steps
    
    return final_lr

ft_params = EvaluatedDict({False: [START_LR_FT, WEIGHT_DECAY_FT, STEP_LR_FT, STEP_LR_DECAY_FT, EPOCHS_FT],
                           True:[[calculate_lr_from_args(epochs=EPOCHS, 
                                                         step_lr=STEP_LR, 
                                                         start_lr=START_LR, 
                                                         step_lr_decay=STEP_LR_DECAY),
                                  WEIGHT_DECAY_FT, STEP_LR_FT, STEP_LR_DECAY, EPOCHS_FT]]}, 
                                  lambda x: x == "ImageNet")
