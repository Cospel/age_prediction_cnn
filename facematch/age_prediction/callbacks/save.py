import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import Callback


class SaveCallback(Callback):
    def __init__(self, save_path="save_cl.h5"):
        super(SaveCallback, self).__init__()
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.save_path, include_optimizer=True)
