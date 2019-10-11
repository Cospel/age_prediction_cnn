import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import Callback


class SaveCallback(Callback):
    def __init__(self, save_path="save_cl.h5"):
        super(SaveCallback, self).__init__()
        self.save_path = save_path

    def on_epoch_start(self, epoch, logs):
        try:
            print(f"Saving model before epoch {epoch} ...")
            self.model.save(self.save_path, include_optimizer=False)
        except:
            print(f"Error saving model before epoch {epoch}")

    def on_epoch_end(self, epoch, logs=None):
        try:
            print(f"Saving model after epoch {epoch} ...")
            self.model.save(self.save_path, include_optimizer=False)
        except:
            print(f"Error saving model after epoch {epoch}")
