import tensorflow as tf
import numpy as np
import sys
import os
import glob
import FCN_VGG16
from PIL import Image
import random
import datetime
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt


def train(loss, var):
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.lr)
    grads = optimizer.compute_gradients(loss, var_list=var)
    return optimizer.apply_gradients(grads)

def load(img_list, mask_list, i):
    origin_img = Image.open(img_list[i])
    origin_mask = Image.open(mask_list[i])
    input_img = np.expand_dims(np.expand_dims(np.array(origin_img), axis=0), axis=3)
    input_mask = np.uint16(np.expand_dims(np.expand_dims(np.array(origin_mask), axis=0), axis=3) / 255)
    return input_img, input_mask, origin_img, origin_mask



def main(_):
    FLAGS = tf.flags.FLAGS
    # gpu config.
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True

    image = tf.compat.v1.placeholder(tf.float32, shape=[None, 1024, 1024, 1], name="image")
    mask = tf.compat.v1.placeholder(tf.int32, shape=[None, 1024, 1024, 1], name="mask")

    train_model = FCN_VGG16.inference(image)
    loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(mask, train_model))

    trainable_var = tf.compat.v1.trainable_variables()
    train_op = train(loss, trainable_var)

    saver = tf.compat.v1.train.Saver()

    if FLAGS.mode == "train":
        with tf.compat.v1.Session(config=config) as sess:
            ckpt = tf.train.get_checkpoint_state(os.path.join(FLAGS.weight_dir, str(FLAGS.threshold)))
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.compat.v1.global_variables_initializer())

            img_list = sorted(glob.glob(FLAGS.training_set + '/img/*.tif'))
            mask_list = sorted(glob.glob(FLAGS.training_set + '/mask/*.tif'))

            val_img_list = sorted(glob.glob(FLAGS.validation_set + '/img/*.tif'))
            val_mask_list = sorted(glob.glob(FLAGS.validation_set + '/mask/*.tif'))

            train_data_size = len(img_list)
            train_iterations = int(train_data_size/FLAGS.bs)
            val_data_size = len(val_img_list)

            for cur_epoch in range(FLAGS.epochs):
                # Shuffle
                tmp = [[x,y] for x,y in zip(img_list, mask_list)]
                random.shuffle(tmp)
                img_list = [n[0] for n in tmp]
                mask_list = [n[1] for n in tmp]

                now = datetime.datetime.now()
                print(now)
                print("threshold: ", FLAGS.threshold, " Start training!, epoch: ", cur_epoch)
                for i in range(train_data_size):
                    # print('i : ',i)
                    input_img, input_mask,_,_ = load(img_list, mask_list, i)

                    feed_dict = {image:input_img, mask:input_mask}
                    sess.run(train_op, feed_dict=feed_dict)

                # validation
                if cur_epoch % 1 ==0:
                    avg_loss = 0
                    iou_score_list = []
                    for i in range(val_data_size):
                        input_img, input_mask,_,_ = load(val_img_list, val_mask_list, i)

                        feed_dict = {image: input_img, mask: input_mask}
                        loss1, pred = sess.run([loss, train_model], feed_dict=feed_dict)
                        avg_loss += loss1

                        # save result
                        pred1 = np.where(pred < FLAGS.threshold, 0.0, 1.0)
                        ttt = Image.fromarray(pred1[0, :, :, 0])
                        img_save_path = os.path.join(FLAGS.result, str(FLAGS.threshold))
                        if not os.path.exists(img_save_path):
                            os.makedirs(img_save_path)
                        file_name = img_save_path +'/epoch' + '%02d' % cur_epoch + '_' + val_mask_list[i].split('/')[-1]
                        ttt.save(file_name)

                        # calculate IoU
                        if np.sum(input_mask) == 0:
                            input_mask = np.logical_not(input_mask)
                            pred1 = np.logical_not(pred1)
                        intersection = np.logical_and(input_mask, pred1)
                        union = np.logical_or(input_mask, pred1)
                        iou_score_list.append(np.sum(intersection) / np.sum(union))

                avg_loss = avg_loss / val_data_size
                print('epoch: ', cur_epoch, 'average loss: ', avg_loss)
                print(iou_score_list)
                print('average iou: ', sum(iou_score_list) / len(iou_score_list))

                weight_save_path = os.path.join(FLAGS.weight_dir, str(FLAGS.threshold), FLAGS.weight)
                if not os.path.exists(weight_save_path):
                    os.makedirs(weight_save_path)
                saver.save(sess, weight_save_path, global_step=cur_epoch)

    if FLAGS.mode == "val":
        val_img_list = sorted(glob.glob(FLAGS.validation_set + '/img/*.tif'))
        val_mask_list = sorted(glob.glob(FLAGS.validation_set + '/mask/*.tif'))

        val_data_size = len(val_img_list)

        with tf.Session(config=config) as sess:
            ckpt = tf.train.get_checkpoint_state(os.path.join(FLAGS.weight_dir, str(FLAGS.threshold)))
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sys.exit("No weights!!")
            avg_loss = 0
            iou_score_list = []
            for i in range(val_data_size):
                input_img, input_mask, origin_img, origin_mask = load(val_img_list, val_mask_list, i)
                feed_dict = {image: input_img, mask: input_mask}
                loss1, pred = sess.run([loss, train_model], feed_dict=feed_dict)
                pred1 = np.where(pred < FLAGS.threshold, 0, 1)
                avg_loss += loss1

                # display
                # image_show = Image.fromarray(np.uint8(origin_img)).convert('RGB')
                # image_show = image_show.resize((400,400))
                # image_show.show()
                fig = plt.figure(figsize=(16,8))
                rows = 1
                cols = 2

                ax1 = fig.add_subplot(rows, cols, 1)
                ax1.imshow(input_mask[0,:,:,0], cmap='gray')
                ax1.set_title('Ground truth')
                ax1.axis("off")

                ax2 = fig.add_subplot(rows, cols, 2)
                ax2.imshow(pred1[0, :, :, 0], cmap='gray')
                ax2.set_title('Predicted result')
                ax2.axis("off")

                plt.show(block=False)
                plt.pause(3) # 3 seconds
                plt.close()


                # calculate IoU
                if np.sum(input_mask)==0:
                    input_mask = np.logical_not(input_mask)
                    pred1 = np.logical_not(pred1)
                intersection = np.logical_and(input_mask, pred1)
                union = np.logical_or(input_mask, pred1)
                iou_score_list.append(np.sum(intersection) / np.sum(union))

            print(iou_score_list)
            print('average iou: ', sum(iou_score_list)/len(iou_score_list))


if __name__ == '__main__':
    FLAGS = tf.flags.FLAGS
    tf.flags.DEFINE_string("mode", "val", "mode: train/val/test")
    tf.flags.DEFINE_string("weight_dir", "./weight_FCN_VGG16", "weight_FCN_VGG16 directory.")
    tf.flags.DEFINE_string("weight_FCN_VGG16", "FCN-VGG16", "the latest weight_FCN_VGG16 saved.")
    tf.flags.DEFINE_float("lr", "1e-4", "learning rate.")
    tf.flags.DEFINE_float("threshold", "0.5", "threshold.")
    tf.flags.DEFINE_string("training_set", "./data/train_ver2", "dataset path for training.")
    tf.flags.DEFINE_string("validation_set", "./data/val", "dataset path for validation.")
    tf.flags.DEFINE_string("result", "./data/val/result_FCN_VGG16", "path for validation result.")
    tf.flags.DEFINE_integer("bs", 1, "batch size for training.")
    tf.flags.DEFINE_integer("epochs", 200, "total training epochs.")

    tf.compat.v1.app.run(main=main)