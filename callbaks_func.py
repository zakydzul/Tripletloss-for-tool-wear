from datetime import datetime
import tensorflow as tf


class call_clbcks:
    def __init__(self, log_name):
        self.logdir = log_name + '/fit/'

    def tensorbrd(self):
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.logdir +
                                                              datetime.now().strftime("%Y%m%d-%H%M%S"),
                                                              profile_batch='5,10')
        return tensorboard_callback

    def patience(self):
        patience_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        return patience_callback


class epoch_end(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_loss') <= 0.004:
            self.model.stop_training = True