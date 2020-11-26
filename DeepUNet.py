import tensorflow as tf

def inference(img):
    assert img is not None
    with tf.compat.v1.variable_scope("inference"):
        # Conv0 + MaxPooling1
        conv0_1 = tf.layers.conv2d(img, 64, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv0_2 = tf.layers.conv2d(conv0_1, 32, 2, activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        pool0 = tf.layers.max_pooling2d(conv0_2, pool_size=2, strides=2, padding='same')

        # Conv1 + MaxPooling1
        conv1_1 = tf.layers.conv2d(pool0, 64, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer = tf.contrib.layers.xavier_initializer())
        conv1_2 = tf.layers.conv2d(conv1_1, 32, 2, activation=tf.nn.relu, padding='same',
            kernel_initializer = tf.contrib.layers.xavier_initializer())
        plus1 = tf.concat([pool0, conv1_2], axis=3)
        pool1 = tf.layers.max_pooling2d(plus1, pool_size=2, strides=2, padding='same')

        # Conv2 + MaxPooling2
        conv2_1 = tf.layers.conv2d(pool1, 64, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv2_2 = tf.layers.conv2d(conv2_1, 32, 2, activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        plus2 = tf.concat([pool1, conv2_2], axis=3)
        pool2 = tf.layers.max_pooling2d(plus2, pool_size=2, strides=2, padding='same')

        # Conv3 + MaxPooling3
        conv3_1 = tf.layers.conv2d(pool2, 64, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv3_2 = tf.layers.conv2d(conv3_1, 32, 2, activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        plus3 = tf.concat([pool2, conv3_2], axis=3)
        pool3 = tf.layers.max_pooling2d(plus3, pool_size=2, strides=2, padding='same')

        # Conv4 + MaxPooling4
        conv4_1 = tf.layers.conv2d(pool3, 64, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv4_2 = tf.layers.conv2d(conv4_1, 32, 2, activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        plus4 = tf.concat([pool3, conv4_2], axis=3)
        pool4 = tf.layers.max_pooling2d(plus4, pool_size=2, strides=2, padding='same')

        # Conv5 + MaxPooling5
        conv5_1 = tf.layers.conv2d(pool4, 64, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv5_2 = tf.layers.conv2d(conv5_1, 32, 2, activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        plus5 = tf.concat([pool4, conv5_2], axis=3)
        pool5 = tf.layers.max_pooling2d(plus5, pool_size=2, strides=2, padding='same')

        # Conv6 + MaxPooling6
        conv6_1 = tf.layers.conv2d(pool5, 64, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv6_2 = tf.layers.conv2d(conv6_1, 32, 2, activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        plus6 = tf.concat([pool5, conv6_2], axis=3)
        pool6 = tf.layers.max_pooling2d(plus6, pool_size=2, strides=2, padding='same')

        # Conv7 + MaxPooling7
        conv7_1 = tf.layers.conv2d(pool6, 64, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv7_2 = tf.layers.conv2d(conv7_1, 32, 2, activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        plus7 = tf.concat([pool6, conv7_2], axis=3)
        pool7 = tf.layers.max_pooling2d(plus7, pool_size=2, strides=2, padding='same')


        # UpSampling + Concat8 + Conv8
        up8 = tf.keras.layers.UpSampling2D(size=(2, 2))(pool7)
        concat8 = tf.concat([up8, plus7], axis=3)  # concat channel
        conv8_1 = tf.layers.conv2d(concat8, 64, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv8_2 = tf.layers.conv2d(conv8_1, 32, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        plus8 = tf.concat([up8, conv8_2], axis=3)

        up9 = tf.keras.layers.UpSampling2D(size=(2, 2))(plus8)
        concat9 = tf.concat([up9, plus6], axis=3)
        conv9_1 = tf.layers.conv2d(concat9, 64, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv9_2 = tf.layers.conv2d(conv9_1, 32, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        plus9 = tf.concat([up9, conv9_2], axis=3)

        up10 = tf.keras.layers.UpSampling2D(size=(2, 2))(plus9)
        concat10 = tf.concat([up10, plus5], axis=3)
        conv10_1 = tf.layers.conv2d(concat10, 64, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv10_2 = tf.layers.conv2d(conv10_1, 32, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        plus10 = tf.concat([up10, conv10_2], axis=3)

        up11 = tf.keras.layers.UpSampling2D(size=(2, 2))(plus10)
        concat11 = tf.concat([up11, plus4], axis=3)
        conv11_1 = tf.layers.conv2d(concat11, 64, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv11_2 = tf.layers.conv2d(conv11_1, 32, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        plus11 = tf.concat([up11, conv11_2], axis=3)

        up12 = tf.keras.layers.UpSampling2D(size=(2, 2))(plus11)
        concat12 = tf.concat([up12, plus3], axis=3)
        conv12_1 = tf.layers.conv2d(concat12, 64, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv12_2 = tf.layers.conv2d(conv12_1, 32, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        plus12 = tf.concat([up12, conv12_2], axis=3)

        up13 = tf.keras.layers.UpSampling2D(size=(2, 2))(plus12)
        concat13 = tf.concat([up13, plus2], axis=3)
        conv13_1 = tf.layers.conv2d(concat13, 64, 3, activation=tf.nn.relu, padding='same',
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv13_2 = tf.layers.conv2d(conv13_1, 32, 3, activation=tf.nn.relu, padding='same',
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
        plus13 = tf.concat([up13, conv13_2], axis=3)

        up14 = tf.keras.layers.UpSampling2D(size=(2, 2))(plus13)
        concat14 = tf.concat([up14, plus1], axis=3)
        conv14_1 = tf.layers.conv2d(concat14, 64, 3, activation=tf.nn.relu, padding='same',
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv14_2 = tf.layers.conv2d(conv14_1, 32, 3, activation=tf.nn.relu, padding='same',
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
        plus14 = tf.concat([up14, conv14_2], axis=3)

        up15 = tf.keras.layers.UpSampling2D(size=(2, 2))(plus14)
        final = tf.layers.conv2d(up15, 1, 1, activation=tf.nn.sigmoid, padding='same',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
    return final