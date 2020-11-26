import tensorflow as tf

def inference(img):
    assert img is not None
    with tf.compat.v1.variable_scope("inference"):
        # Conv1 + MaxPooling1 + Crop1
        conv1_1 = tf.layers.conv2d(img, 64, 3, activation=tf.nn.relu, padding='same',
           kernel_initializer = tf.contrib.layers.xavier_initializer())

        conv1_2 = tf.layers.conv2d(conv1_1, 64, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer = tf.contrib.layers.xavier_initializer())
        pool1 = tf.layers.max_pooling2d(conv1_2, pool_size=2, strides=2, padding='same')
        crop1 = tf.keras.layers.Cropping2D(cropping=((88, 88), (88, 88)))(conv1_2)

        # Conv2 + MaxPooling2 + Crop2
        conv2_1 = tf.layers.conv2d(pool1, 128, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer = tf.contrib.layers.xavier_initializer())
        conv2_2 = tf.layers.conv2d(conv2_1, 128, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer = tf.contrib.layers.xavier_initializer())
        pool2 = tf.layers.max_pooling2d(conv2_2, pool_size=2, strides=2, padding='same')
        crop2 = tf.keras.layers.Cropping2D(cropping=((40, 40), (40, 40)))(conv2_2)

        # Conv3 + MaxPooling3 + Crop3
        conv3_1 = tf.layers.conv2d(pool2, 256, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer = tf.contrib.layers.xavier_initializer())
        conv3_2 = tf.layers.conv2d(conv3_1, 256, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer = tf.contrib.layers.xavier_initializer())
        pool3 = tf.layers.max_pooling2d(conv3_2, pool_size=2, strides=2, padding='same')
        crop3 = tf.keras.layers.Cropping2D(cropping=((16, 16), (16, 16)))(conv3_2)

        # Conv4 + MaxPooling4 + Crop4
        conv4_1 = tf.layers.conv2d(pool3, 512, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer = tf.contrib.layers.xavier_initializer())
        conv4_2 = tf.layers.conv2d(conv4_1, 512, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer = tf.contrib.layers.xavier_initializer())
        pool4 = tf.layers.max_pooling2d(conv4_2, pool_size=2, strides=2, padding='same')
        # crop4 = tf.keras.layers.Cropping2D(cropping=((4, 4), (4, 4)))(conv4_2)

        # Conv5
        conv5_1 = tf.layers.conv2d(pool4, 1024, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer = tf.contrib.layers.xavier_initializer())
        conv5_2 = tf.layers.conv2d(conv5_1, 1024, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer = tf.contrib.layers.xavier_initializer())

        # UpConv + Concat + Conv6
        up6 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv5_2)
        # merge6 = tf.concat([crop4, up6], axis=3) # concat channel
        merge6 = tf.concat([conv4_2, up6], axis=3)  # concat channel
        conv6_1 = tf.layers.conv2d(merge6, 512, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv6_2 = tf.layers.conv2d(conv6_1, 512, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())

        # UpConv + Concat + Conv7
        up7 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv6_2)
        # merge7 = tf.concat([crop3, up7], axis=3) # concat channel
        merge7 = tf.concat([conv3_2, up7], axis=3)  # concat channel
        conv7_1 = tf.layers.conv2d(merge7, 256, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer = tf.contrib.layers.xavier_initializer())
        conv7_2 = tf.layers.conv2d(conv7_1, 256, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer = tf.contrib.layers.xavier_initializer())

        # UpConv + Concat + Conv8
        up8 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv7_2)
        # merge8 = tf.concat([crop2, up8], axis=3) # concat channel
        merge8 = tf.concat([conv2_2, up8], axis=3)  # concat channel
        conv8_1 = tf.layers.conv2d(merge8, 128, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer = tf.contrib.layers.xavier_initializer())
        conv8_2 = tf.layers.conv2d(conv8_1, 128, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer = tf.contrib.layers.xavier_initializer())

        # UpConv + Concat + Conv9
        up9 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv8_2)
        # merge9 = tf.concat([crop1, up9], axis=3) # concat channel
        merge9 = tf.concat([conv1_2, up9], axis=3)  # concat channel
        conv9_1 = tf.layers.conv2d(merge9, 64, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer = tf.contrib.layers.xavier_initializer())
        conv9_2 = tf.layers.conv2d(conv9_1, 64, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer = tf.contrib.layers.xavier_initializer())

        final = tf.layers.conv2d(conv9_2, 1, 1, activation=tf.nn.sigmoid, padding='same',
            kernel_initializer = tf.contrib.layers.xavier_initializer())

    return final