import numpy as np
import glob
from PIL import Image

val_mask_list = sorted(glob.glob('./mask/*.tif'))
pred_list = sorted(glob.glob('result_FCN_VGG16/0.5/*.tif'))

print(val_mask_list)
print(pred_list)

num = len(val_mask_list)
set_num = int(len(pred_list)/10)
pred = 0

for s in range(set_num):
    iou_score_list = []
    for i in range(num):
        input_mask = np.uint16(np.expand_dims(np.expand_dims(np.array(Image.open(val_mask_list[i])), axis=0), axis=3) / 255)
        pred1 = np.expand_dims(np.expand_dims(np.array(Image.open(pred_list[i+pred])), axis=0), axis=3)

        print(pred_list[i+pred])

        if np.sum(input_mask) == 0:
            input_mask = np.logical_not(input_mask)
            pred1 = np.logical_not(pred1)
        intersection = np.logical_and(input_mask, pred1)
        union = np.logical_or(input_mask, pred1)
        iou_score_list.append(np.sum(intersection) / np.sum(union))
    print('set_num: ', s, ' ', iou_score_list)
    mean = sum(iou_score_list) / len(iou_score_list)
    print('average IoU: ', mean)

    pred += 10


