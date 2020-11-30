# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import logging
import os
import time
import glob

import numpy as np
import numpy.ma as ma
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.utils import AverageMeter
from utils.utils import get_confusion_matrix
from utils.utils import adjust_learning_rate
from matplotlib import pyplot as plt

from PIL import Image

def train(config, epoch, num_epoch, epoch_iters, base_lr, 
        num_iters, trainloader, optimizer, model, writer_dict):
    # Training
    model.train()
    batch_time = AverageMeter()
    ave_loss = AverageMeter()
    tic = time.time()
    cur_iters = epoch*epoch_iters
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    for i_iter, batch in enumerate(trainloader, 0):
        # print('i_iter :', i_iter)
        images, labels, _, _ = batch
        labels = labels.long().cuda()

        losses, _ = model(images, labels)
        loss = losses.mean()

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        ave_loss.update(loss.item())

        lr = adjust_learning_rate(optimizer,
                                  base_lr,
                                  num_iters,
                                  i_iter+cur_iters)

        if i_iter % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                  'lr: {:.6f}, Loss: {:.6f}' .format(
                      epoch, num_epoch, i_iter, epoch_iters, 
                      batch_time.average(), lr, ave_loss.average())
            logging.info(msg)

    writer.add_scalar('train_loss', ave_loss.average(), global_steps)
    writer_dict['train_global_steps'] = global_steps + 1

def validate(config, testloader, model, writer_dict, epoch):
    model.eval()
    ave_loss = AverageMeter()

    val_mask_list = sorted(glob.glob('../../data/val_full/mask/*.tif'))
    # print('len val_mask list: ', len(val_mask_list))

    iou_score_list_1 = []
    c = 0
    with torch.no_grad():
        for _, batch in enumerate(testloader):
            image, label, _, name = batch
            size = label.size()
            label = label.long().cuda()

            losses, pred = model(image, label)
            pred = F.upsample(input=pred, size=(
                        size[-2], size[-1]), mode='bilinear')
            loss = losses.mean()
            ave_loss.update(loss.item())

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1)
            pred = np.asarray(np.argmax(pred, axis=3), dtype=np.uint8)
            label = label.cpu().numpy()

            num = pred.shape[0]

            for i in range(c, c+num):
                # save the result
                threshold = 0.5
                pred1 = np.where(pred[i-c,:,:] < threshold, 0.0, 1.0)
                ttt = Image.fromarray(pred1)
                img_save_path = os.path.join(config.DATASET.ROOT, config.DATASET.TEST_RESULT, str(threshold))
                if not os.path.exists(img_save_path):
                    os.makedirs(img_save_path)
                file_name = img_save_path + '/epoch' + '%02d' % epoch + '_' + 'WTR000' + '%02d' % (i+1) + '.tif'
                ttt.save(file_name)

                # calculate IoU
                if np.sum(label[i-c,:,:]) == 0:
                    label[i-c,:,:] = np.logical_not(label[i-c,:,:])
                    pred1 = np.logical_not(pred1)
                intersection = np.logical_and(label[i-c,:,:], pred1)
                union = np.logical_or(label[i-c,:,:], pred1)
                iou_score_list_1.append(np.sum(intersection) / np.sum(union))

                # display
                # fig = plt.figure(figsize=(16, 8))
                # rows = 1
                # cols = 2
                #
                # ax1 = fig.add_subplot(rows, cols, 1)
                # ax1.imshow(label[i-c,:,:], cmap='gray')
                # ax1.set_title('Ground truth')
                # ax1.axis("off")
                #
                # ax2 = fig.add_subplot(rows, cols, 2)
                # ax2.imshow(pred1, cmap='gray')
                # ax2.set_title('Predicted result')
                # ax2.axis("off")
                #
                # plt.show(block=False)
                # plt.pause(3)  # 3 seconds
                # plt.close()

            c += num
        print(iou_score_list_1)
        mean_IoU = sum(iou_score_list_1) / len(iou_score_list_1)
        print('average IoU 1: ', mean_IoU)


    #         confusion_matrix += get_confusion_matrix(
    #             label,
    #             pred,
    #             size,
    #             config.DATASET.NUM_CLASSES,
    #             config.TRAIN.IGNORE_LABEL)
    #
    # pos = confusion_matrix.sum(1)
    # res = confusion_matrix.sum(0)
    # tp = np.diag(confusion_matrix)
    # IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    # mean_IoU = IoU_array.mean()

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', ave_loss.average(), global_steps)
    writer.add_scalar('valid_mIoU', mean_IoU, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return ave_loss.average(), mean_IoU

def testval(config, test_dataset, testloader, model, 
        sv_dir='', sv_pred=False):
    model.eval()
    confusion_matrix = np.zeros(
        (config.DATASET.NUM_CLASSES, config.DATASET.NUM_CLASSES))
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name = batch
            size = label.size()
            pred = test_dataset.multi_scale_inference(
                        model, 
                        image, 
                        scales=config.TEST.SCALE_LIST, 
                        flip=config.TEST.FLIP_TEST)
            
            if pred.size()[-2] != size[-2] or pred.size()[-1] != size[-1]:
                pred = F.upsample(pred, (size[-2], size[-1]), 
                                   mode='bilinear')

            confusion_matrix += get_confusion_matrix(
                label,
                pred,
                size,
                config.DATASET.NUM_CLASSES,
                config.TRAIN.IGNORE_LABEL)

            if sv_pred:
                sv_path = os.path.join(sv_dir,'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
            
            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                pos = confusion_matrix.sum(1)
                res = confusion_matrix.sum(0)
                tp = np.diag(confusion_matrix)
                IoU_array = (tp / np.maximum(1.0, pos + res - tp))
                mean_IoU = IoU_array.mean()
                logging.info('mIoU: %.4f' % (mean_IoU))

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    pixel_acc = tp.sum()/pos.sum()
    mean_acc = (tp/np.maximum(1.0, pos)).mean()
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()

    return mean_IoU, IoU_array, pixel_acc, mean_acc

def test(config, test_dataset, testloader, model, 
        sv_dir='', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = test_dataset.multi_scale_inference(
                        model, 
                        image, 
                        scales=config.TEST.SCALE_LIST, 
                        flip=config.TEST.FLIP_TEST)
            
            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.upsample(pred, (size[-2], size[-1]), 
                                   mode='bilinear')

            if sv_pred:
                sv_path = os.path.join(sv_dir,'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)