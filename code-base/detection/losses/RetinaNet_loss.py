import tensorflow as tf
from detection.losses.focal_loss import *
from detection.losses.smooth_l1_loss import *


class RetinaNetLoss(tf.losses.Loss):

    def __init__(self, num_classes=80, alpha=0.25, gamma=2.0, delta=1.0):
        super(RetinaNetLoss, self).__init__(reduction="auto")
        self.cls_loss = FocalLoss(alpha, gamma)
        self.box_loss = SmoothL1Loss(delta)
        self.num_classes = num_classes  

    def call(self, y_true, y_pred):
        assert len(y_true.shape) == 3 and y_true.shape[2] == 5
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        ### Compute classification loss
        cls_pred = y_pred[:, :, 4:]
        cls_true = tf.one_hot(tf.cast(y_true[:, :, 4], dtype=tf.int32), depth=self.num_classes, dtype=tf.float32)
        cls_loss = self.cls_loss(cls_pred, cls_true)
        ignore_mask = tf.cast(tf.equal(y_true[:, :, 4], -2.0), dtype=tf.float32)
        cls_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, cls_loss)
        cls_loss = tf.reduce_sum(cls_loss, axis=-1)
        ### Compute regression loss
        box_pred = y_pred[:, :, :4]
        box_true = y_true[:, :, :4]
        box_loss = self.box_loss(box_pred, box_true)
        positive_mask = tf.cast(tf.greater(y_true[:, :, 4], -1.0), dtype=tf.float32)
        box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)
        box_loss = tf.reduce_sum(box_loss, axis=-1)
        ### Normalize losses
        normalizer = tf.reduce_sum(positive_mask, axis=-1)
        cls_loss = tf.math.divide_no_nan(cls_loss, normalizer)
        box_loss = tf.math.divide_no_nan(box_loss, normalizer)
        ### Return loss
        print(f"cls_loss = {cls_loss}")
        print(f"box_loss = {box_loss}")
        return cls_loss + box_loss
