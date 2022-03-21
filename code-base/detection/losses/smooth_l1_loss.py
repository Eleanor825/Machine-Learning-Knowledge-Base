import tensorflow as tf


class SmoothL1Loss(tf.losses.Loss):

    def __init__(self, delta):
        super(SmoothL1Loss, self).__init__(reduction="none")
        self.delta = delta

    def call(self, y_true, y_pred):
        diff = y_true - y_pred
        absolute = tf.abs(diff)
        squared = diff ** 2
        loss = tf.where(tf.less(absolute, self.delta), 0.5 * squared, absolute - 0.5)
        return tf.reduce_sum(loss, axis=-1)
