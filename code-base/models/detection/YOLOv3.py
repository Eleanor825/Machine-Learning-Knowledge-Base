import tensorflow as tf


class YOLOv3(tf.keras.Model):

    def __init__(self, num_classes, **kwargs):
        super(YOLOv3, self).__init__(name="YOLOv3", **kwargs)
        self._num_classes = num_classes

    def _conv_block(self, x, convs, skip=True):
        count = 0
        for conv in convs:
            if count == (len(convs) - 2) and skip:
                skip_connection = x
            count += 1

            if conv['stride'] > 1:
                # peculiar padding as darknet prefer left and top
                x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
            x = tf.keras.layers.Conv2D(
                filters=conv['filters'],
                kernel_size=conv['kernel'],
                strides=conv['stride'],
                # peculiar padding as darknet prefer left and top
                padding='valid' if conv['stride'] > 1 else 'same',
                name='conv_' + str(conv['layer_idx']),
                use_bias=False if conv['bnorm'] else True)(x)
            if conv['bnorm']:
                x = tf.keras.layers.BatchNormalization(
                    epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)
            if conv['leaky']:
                x = tf.keras.layers.LeakyReLU(
                    alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)

        return tf.keras.layers.add([skip_connection, x]) if skip else x

    def call(self, x):
        ##################################################
        x = self._conv_block(x, [
            {'filters': 32, 'kernel': 3, 'stride': 1,
            'bnorm': True, 'leaky': True, 'layer_idx': 0},
            {'filters': 64, 'kernel': 3, 'stride': 2,
            'bnorm': True, 'leaky': True, 'layer_idx': 1},
            {'filters': 32, 'kernel': 1, 'stride': 1,
            'bnorm': True, 'leaky': True, 'layer_idx': 2},
            {'filters': 64, 'kernel': 3, 'stride': 1,
            'bnorm': True, 'leaky': True, 'layer_idx': 3}
        ])
        ##################################################
        x = self._conv_block(x, [
            {'filters': 128, 'kernel': 3, 'stride': 2,
            'bnorm': True, 'leaky': True, 'layer_idx': 5},
            {'filters':  64, 'kernel': 1, 'stride': 1,
            'bnorm': True, 'leaky': True, 'layer_idx': 6},
            {'filters': 128, 'kernel': 3, 'stride': 1,
            'bnorm': True, 'leaky': True, 'layer_idx': 7}
        ])
        ##################################################
        x = self._conv_block(x, [
            {'filters':  64, 'kernel': 1, 'stride': 1,
            'bnorm': True, 'leaky': True, 'layer_idx': 9},
            {'filters': 128, 'kernel': 3, 'stride': 1,
            'bnorm': True, 'leaky': True, 'layer_idx': 10}
        ])
        ##################################################
        x = self._conv_block(x, [
            {'filters': 256, 'kernel': 3, 'stride': 2,
            'bnorm': True, 'leaky': True, 'layer_idx': 12},
            {'filters': 128, 'kernel': 1, 'stride': 1,
            'bnorm': True, 'leaky': True, 'layer_idx': 13},
            {'filters': 256, 'kernel': 3, 'stride': 1,
            'bnorm': True, 'leaky': True, 'layer_idx': 14}
        ])
        ##################################################
        for i in range(7):
            x = self._conv_block(x, [
                {'filters': 128, 'kernel': 1, 'stride': 1,
                'bnorm': True, 'leaky': True, 'layer_idx': 16+i*3},
                {'filters': 256, 'kernel': 3, 'stride': 1,
                'bnorm': True, 'leaky': True, 'layer_idx': 17+i*3}
            ])
        ##################################################
        skip_36 = x
        ##################################################
        x = self._conv_block(x, [
            {'filters': 512, 'kernel': 3, 'stride': 2,
            'bnorm': True, 'leaky': True, 'layer_idx': 37},
            {'filters': 256, 'kernel': 1, 'stride': 1,
            'bnorm': True, 'leaky': True, 'layer_idx': 38},
            {'filters': 512, 'kernel': 3, 'stride': 1,
            'bnorm': True, 'leaky': True, 'layer_idx': 39}
        ])
        ##################################################
        for i in range(7):
            x = self._conv_block(x, [
                {'filters': 256, 'kernel': 1, 'stride': 1,
                'bnorm': True, 'leaky': True, 'layer_idx': 41+i*3},
                {'filters': 512, 'kernel': 3, 'stride': 1,
                'bnorm': True, 'leaky': True, 'layer_idx': 42+i*3}
            ])
        ##################################################
        skip_61 = x
        ##################################################
        x = self._conv_block(x, [
            {'filters': 1024, 'kernel': 3, 'stride': 2,
            'bnorm': True, 'leaky': True, 'layer_idx': 62},
            {'filters':  512, 'kernel': 1, 'stride': 1,
            'bnorm': True, 'leaky': True, 'layer_idx': 63},
            {'filters': 1024, 'kernel': 3, 'stride': 1,
            'bnorm': True, 'leaky': True, 'layer_idx': 64},
        ])
        ##################################################
        for i in range(3):
            x = self._conv_block(x, [
                {'filters':  512, 'kernel': 1, 'stride': 1,
                'bnorm': True, 'leaky': True, 'layer_idx': 66+i*3},
                {'filters': 1024, 'kernel': 3, 'stride': 1,
                'bnorm': True, 'leaky': True, 'layer_idx': 67+i*3},
            ])
        ##################################################
        x = self._conv_block(x, [
            {'filters':  512, 'kernel': 1, 'stride': 1,
            'bnorm': True, 'leaky': True, 'layer_idx': 75},
            {'filters': 1024, 'kernel': 3, 'stride': 1,
            'bnorm': True, 'leaky': True, 'layer_idx': 76},
            {'filters':  512, 'kernel': 1, 'stride': 1,
            'bnorm': True, 'leaky': True, 'layer_idx': 77},
            {'filters': 1024, 'kernel': 3, 'stride': 1,
            'bnorm': True, 'leaky': True, 'layer_idx': 78},
            {'filters':  512, 'kernel': 1, 'stride': 1,
            'bnorm': True, 'leaky': True, 'layer_idx': 79},
        ], skip=False)
        ##################################################
        yolo_82 = self._conv_block(x, [
            {'filters': 1024, 'kernel': 3, 'stride': 1,
            'bnorm': True,  'leaky': True,  'layer_idx': 80},
            {'filters':  255, 'kernel': 1, 'stride': 1,
            'bnorm': False, 'leaky': False, 'layer_idx': 81},
        ], skip=False)
        ##################################################
        x = self._conv_block(x, [
            {'filters': 256, 'kernel': 1, 'stride': 1,
            'bnorm': True, 'leaky': True, 'layer_idx': 84}
        ], skip=False)
        x = tf.keras.layers.UpSampling2D(2)(x)
        x = tf.concat([x, skip_61], axis=-1)
        ##################################################
        x = self._conv_block(x, [
            {'filters': 256, 'kernel': 1, 'stride': 1,
            'bnorm': True, 'leaky': True, 'layer_idx': 87},
            {'filters': 512, 'kernel': 3, 'stride': 1,
            'bnorm': True, 'leaky': True, 'layer_idx': 88},
            {'filters': 256, 'kernel': 1, 'stride': 1,
            'bnorm': True, 'leaky': True, 'layer_idx': 89},
            {'filters': 512, 'kernel': 3, 'stride': 1,
            'bnorm': True, 'leaky': True, 'layer_idx': 90},
            {'filters': 256, 'kernel': 1, 'stride': 1,
            'bnorm': True, 'leaky': True, 'layer_idx': 91},
        ], skip=False)
        ##################################################
        yolo_94 = self._conv_block(x, [
            {'filters': 512, 'kernel': 3, 'stride': 1,
            'bnorm': True,  'leaky': True,  'layer_idx': 92},
            {'filters': 255, 'kernel': 1, 'stride': 1,
            'bnorm': False, 'leaky': False, 'layer_idx': 93}
        ], skip=False)
        ##################################################
        x = self._conv_block(x, [
            {'filters': 128, 'kernel': 1, 'stride': 1,
            'bnorm': True, 'leaky': True, 'layer_idx': 96}
        ], skip=False)
        x = tf.keras.layers.UpSampling2D(2)(x)
        x = tf.concat([x, skip_36], axis=-1)
        ##################################################
        yolo_106 = self._conv_block(x, [
            {'filters': 128, 'kernel': 1, 'stride': 1,
            'bnorm': True,  'leaky': True,  'layer_idx': 99},
            {'filters': 256, 'kernel': 3, 'stride': 1,
            'bnorm': True,  'leaky': True,  'layer_idx': 100},
            {'filters': 128, 'kernel': 1, 'stride': 1,
            'bnorm': True,  'leaky': True,  'layer_idx': 101},
            {'filters': 256, 'kernel': 3, 'stride': 1,
            'bnorm': True,  'leaky': True,  'layer_idx': 102},
            {'filters': 128, 'kernel': 1, 'stride': 1,
            'bnorm': True,  'leaky': True,  'layer_idx': 103},
            {'filters': 256, 'kernel': 3, 'stride': 1,
            'bnorm': True,  'leaky': True,  'layer_idx': 104},
            {'filters': 255, 'kernel': 1, 'stride': 1,
            'bnorm': False, 'leaky': False, 'layer_idx': 105}
        ], skip=False)
        ##################################################
        return [yolo_82, yolo_94, yolo_106]
