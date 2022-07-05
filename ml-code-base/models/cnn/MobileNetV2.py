import tensorflow as tf


class MobileNetV2(tf.keras.Model):

    def __init__(self, output_dim, **kwargs):
        super(MobileNetV2, self).__init__(name="MobileNetV2", **kwargs)
        self._output_dim = output_dim

    def _bottleneck_block(self, x, input_channels, output_channels,
                          strides, expansion_factor):
        """
        Arguments:
            x (tensor): input to the bottleneck block.
            input_channels (int): number of input channels.
            output_channels (int): number of output channels.
            strides (int): strides for the depthwise convolutional layer.
            expansion_factor (float): expansion factor from num input channels to num intermediate channels.
        Returns:
            Processed input (x).
        """
        shortcut = x
        intermediate_channels = input_channels * expansion_factor
        relu6_conv = tf.keras.layers.Conv2D(
            filters=intermediate_channels, kernel_size=1, strides=1, padding="same",
        )
        depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=3, strides=strides, padding="same",
        )
        linear_conv = tf.keras.layers.Conv2D(
            filters=output_channels, kernel_size=1, strides=1, padding="same",
        )
        relu6 = tf.keras.layers.ReLU(
            max_value=6, negative_slope=0, threshold=0,
        )
        x = relu6_conv(x)
        x = relu6(x)
        x = depthwise_conv(x)
        x = relu6(x)
        x = linear_conv(x)
        return tf.keras.layers.Add()([x, shortcut])

    def _bottleneck_sequence(self, x, input_channels, output_channels, s, t, n):
        """
        Arguments:
            x (tensor): input to the bottleneck sequence.
            input_channels (int): number of input channels.
            output_channels (int): number of output channels.
            s (int): strides for the first bottleneck block in the sequence.
            t (float): expansion factor for each bottleneck block in the sequence.
            n (int): number of bottleneck blocks in the sequence.
        Returns:
            Processed input (x).
        """
        for i in range(n):
            strides = s if i == 0 else 1
            x = self._bottleneck_block(
                x, input_channels=input_channels, output_channels=output_channels,
                strides=strides, expansion_factor=t,
            )
        return x

    def call(self, x):
        x = tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, strides=2, padding="same",
        )(x)
        x = tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, strides=2, padding="same",
        )(x)
        x = self._bottleneck_sequence(
            x, input_channels=32, output_channels=16, s=1, t=1, n=1,
        )
        x = self._bottleneck_sequence(
            x, input_channels=16, output_channels=24, s=2, t=6, n=2,
        )
        x = self._bottleneck_sequence(
            x, input_channels=24, output_channels=32, s=2, t=6, n=3,
        )
        x = self._bottleneck_sequence(
            x, input_channels=32, output_channels=64, s=2, t=6, n=4,
        )
        x = self._bottleneck_sequence(
            x, input_channels=64, output_channels=96, s=1, t=6, n=3,
        )
        x = self._bottleneck_sequence(
            x, input_channels=96, output_channels=160, s=2, t=6, n=3,
        )
        x = self._bottleneck_sequence(
            x, input_channels=160, output_channels=320, s=1, t=6, n=1,
        )
        x = tf.keras.layers.Conv2D(
            filters=1280, kernel_size=1, strides=1, padding="same",
        )(x)
        x = tf.keras.layers.AveragePooling2D(
            pool_size=7, strides=1, padding="valid",
        )(x)
        x = tf.keras.layers.Conv2D(
            filters=self._output_dim, kernel_size=1, strides=1, padding="valid",
        )(x)
        return x
