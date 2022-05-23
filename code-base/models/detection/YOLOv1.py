import tensorflow as tf


class YOLOv1(tf.keras.Model):

    def __init__(self, num_boxes, num_grid_sqrt, num_classes=20, **kwargs):
        """
        Arguments:
            num_boxes (int): number of bounding boxes per grid cell.
            num_grid_sqrt (int): number of cells per row (col).
            num_classes (int): number of classes in the dataset.
        """
        super(YOLOv1, self).__init__(name="YOLOv1", **kwargs)
        self._S = num_grid_sqrt
        self._B = num_boxes
        self._C = num_classes

    def _conv_block_1(self, x):
        x = tf.keras.layers.Conv2D(
            filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same",
        )(x)
        x = tf.keras.activations.relu(x, alpha=0.1)
        x = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2), strides=(2, 2), padding="same",
        )(x)
        x = tf.keras.activations.relu(x, alpha=0.1)
        return x

    def _conv_block_2(self, x):
        x = tf.keras.layers.Conv2D(
            filters=192, kernel_size=(3, 3), padding="same",
        )(x)
        x = tf.keras.activations.relu(x, alpha=0.1)
        x = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2), strides=(2, 2), padding="same",
        )(x)
        x = tf.keras.activations.relu(x, alpha=0.1)
        return x

    def _conv_block_3(self, x):
        x = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(1, 1), padding="same",
        )(x)
        x = tf.keras.activations.relu(x, alpha=0.1)
        x = tf.keras.layers.Conv2D(
            filters=256, kernel_size=(3, 3), padding="same",
        )(x)
        x = tf.keras.activations.relu(x, alpha=0.1)
        x = tf.keras.layers.Conv2D(
            filters=256, kernel_size=(1, 1), padding="same",
        )(x)
        x = tf.keras.activations.relu(x, alpha=0.1)
        x = tf.keras.layers.Conv2D(
            filters=512, kernel_size=(3, 3), padding="same",
        )(x)
        x = tf.keras.activations.relu(x, alpha=0.1)
        x = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2), strides=(2, 2), padding="same",
        )(x)
        x = tf.keras.activations.relu(x, alpha=0.1)
        return x

    def _conv_block_4(self, x):
        for _ in range(4):
            x = tf.keras.layers.Conv2D(
                filters=256, kernel_size=(1, 1), padding="same",
            )(x)
            x = tf.keras.activations.relu(x, alpha=0.1)
            x = tf.keras.layers.Conv2D(
                filters=512, kernel_size=(3, 3), padding="same",
            )(x)
            x = tf.keras.activations.relu(x, alpha=0.1)
        x = tf.keras.layers.Conv2D(
            filters=512, kernel_size=(1, 1), padding="same",
        )(x)
        x = tf.keras.activations.relu(x, alpha=0.1)
        x = tf.keras.layers.Conv2D(
            filters=1024, kernel_size=(3, 3), padding="same",
        )(x)
        x = tf.keras.activations.relu(x, alpha=0.1)
        x = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2), strides=(2, 2), padding="same",
        )(x)
        x = tf.keras.activations.relu(x, alpha=0.1)
        return x

    def _conv_block_5(self, x):
        for _ in range(2):
            x = tf.keras.layers.Conv2D(
                filters=512, kernel_size=(1, 1), padding="same",
            )(x)
            x = tf.keras.activations.relu(x, alpha=0.1)
            x = tf.keras.layers.Conv2D(
                filters=1024, kernel_size=(3, 3), padding="same",
            )(x)
            x = tf.keras.activations.relu(x, alpha=0.1)
        x = tf.keras.layers.Conv2D(
            filters=1024, kernel_size=(3, 3), padding="same",
        )(x)
        x = tf.keras.activations.relu(x, alpha=0.1)
        x = tf.keras.layers.Conv2D(
            filters=1024, kernel_size=(3, 3), strides=(2, 2), padding="same",
        )(x)
        x = tf.keras.activations.relu(x, alpha=0.1)
        return x

    def _conv_block_6(self, x):
        x = tf.keras.layers.Conv2D(
            filters=1024, kernel_size=(3, 3), padding="same",
        )(x)
        x = tf.keras.activations.relu(x, alpha=0.1)
        x = tf.keras.layers.Conv2D(
            filters=1024, kernel_size=(3, 3), padding="same",
        )(x)
        x = tf.keras.activations.relu(x, alpha=0.1)
        return x

    def _dense_layers(self, x):
        x = tf.keras.layers.Dense(units=4096)(x)
        x = tf.keras.activations.relu(x, alpha=0.1)
        x = tf.keras.layers.Dropout(rate=0.5)(x)
        output_length = self._S * self._S * (self._B * 5 + self._C)
        x = tf.keras.layers.Dense(units=output_length)(x)
        return x

    def call(self, x):
        x = self._conv_block_1(x)
        x = self._conv_block_2(x)
        x = self._conv_block_3(x)
        x = self._conv_block_4(x)
        x = self._conv_block_5(x)
        x = self._conv_block_6(x)
        x = self._dense_layers(x)
        return x
