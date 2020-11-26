import tensorflow as tf

def inference(img):
    assert img is not None
    with tf.compat.v1.variable_scope("inference"):
        # VGG16
        conv1_1 = tf.layers.conv2d(img, 64, 3, activation=tf.nn.relu, padding='same',
           kernel_initializer = tf.contrib.layers.xavier_initializer())
        conv1_2 = tf.layers.conv2d(conv1_1, 64, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer = tf.contrib.layers.xavier_initializer())
        pool1 = tf.layers.max_pooling2d(conv1_2, pool_size=2, strides=2, padding='same')

        conv2_1 = tf.layers.conv2d(pool1, 128, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer = tf.contrib.layers.xavier_initializer())
        conv2_2 = tf.layers.conv2d(conv2_1, 128, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer = tf.contrib.layers.xavier_initializer())
        pool2 = tf.layers.max_pooling2d(conv2_2, pool_size=2, strides=2, padding='same')

        conv3_1 = tf.layers.conv2d(pool2, 256, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer = tf.contrib.layers.xavier_initializer())
        conv3_2 = tf.layers.conv2d(conv3_1, 256, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer = tf.contrib.layers.xavier_initializer())
        conv3_3 = tf.layers.conv2d(conv3_2, 256, 3, activation=tf.nn.relu, padding='same',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())
        pool3 = tf.layers.max_pooling2d(conv3_3, pool_size=2, strides=2, padding='same')

        conv4_1 = tf.layers.conv2d(pool3, 512, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer = tf.contrib.layers.xavier_initializer())
        conv4_2 = tf.layers.conv2d(conv4_1, 512, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer = tf.contrib.layers.xavier_initializer())
        conv4_3 = tf.layers.conv2d(conv4_2, 512, 3, activation=tf.nn.relu, padding='same',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())
        pool4 = tf.layers.max_pooling2d(conv4_3, pool_size=2, strides=2, padding='same')

        conv5_1 = tf.layers.conv2d(pool4, 512, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer = tf.contrib.layers.xavier_initializer())
        conv5_2 = tf.layers.conv2d(conv5_1, 512, 3, activation=tf.nn.relu, padding='same',
            kernel_initializer = tf.contrib.layers.xavier_initializer())
        conv5_3 = tf.layers.conv2d(conv5_2, 512, 3, activation=tf.nn.relu, padding='same',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())
        pool5 = tf.layers.max_pooling2d(conv5_3, pool_size=2, strides=2, padding='same')

        # FCN
        conv6 = tf.layers.conv2d(pool5, 4096, 7, activation=tf.nn.relu, padding='same',
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv7 = tf.layers.conv2d(conv6, 4096, 1, activation=tf.nn.relu, padding='same',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
        conv8 = tf.layers.conv2d(conv7, 2, 1, activation=tf.nn.relu, padding='same',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer()) # NUM of Classes = 2

        # Upscale to the original img size
        conv_t1_shape = pool4.get_shape()
        conv_t1 = tf.layers.conv2d_transpose(conv8, filters=conv_t1_shape[3].value, kernel_size=4, strides=2, padding='same',
                                             kernel_initializer=tf.contrib.layers.xavier_initializer())
        fuse1 = tf.add(conv_t1, pool4)

        conv_t2_shape = pool3.get_shape()
        conv_t2 = tf.layers.conv2d_transpose(fuse1, filters=conv_t2_shape[3].value, kernel_size=8, strides=2, padding='same',
                                             kernel_initializer=tf.contrib.layers.xavier_initializer())
        fuse2 = tf.add(conv_t2, pool3)

        final = tf.layers.conv2d_transpose(fuse2, filters=1, kernel_size=16, strides=8, padding='same',
                                             kernel_initializer=tf.contrib.layers.xavier_initializer()) # NUM of Classes = 2

    return final