# source1: https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%204%20-%20Lesson%202%20-%20Notebook.ipynb
# source2: https://www.coursera.org/learn/introduction-tensorflow/supplement/XFWSt/get-hands-on-with-computer-vision
# Coursera->Introduction to Tensorflow->week2
# --------------------------------------------------
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch ,logs={}):
        if (logs.get('acc')>0.99):
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training=True

callbacks=MyCallback()