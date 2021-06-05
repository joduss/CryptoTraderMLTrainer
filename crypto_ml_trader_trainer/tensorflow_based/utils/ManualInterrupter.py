from tensorflow.python.keras.callbacks import LambdaCallback


class ManualInterrupter(LambdaCallback):
    """
    A way to interrupt manually the model training in debug mode.
    """

    should_stop: bool = False

    def __init__(self):
        super(LambdaCallback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
            if self.should_stop:
                self.model.stop_training = True
