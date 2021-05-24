#Source 1: https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction/lecture/KcQGM/deep-neural-network
#Source 2: https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%202%20Lesson%203.ipynb#scrollTo=tnCe_nBKu7RB
# ----------------------------------------------------------------------------
#! Adam, Adagrad and RMSprop use adaptive learning rates.
#!! This technique can be applied on SGD and many other loss functions...
# ----------------------------------------------------------------------------

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))

lrs = 1e-8 * (10 ** (np.arange(100) / 20))
plt.semilogx(lrs, history.history["loss"])
plt.axis([1e-8, 1e-3, 0, 300])

#after plotting, choose last learning late where loss is still stable.