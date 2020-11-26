import numpy as np
import glob
from six.moves import xrange
import scipy.ndimage as ndi
import os
from PIL import Image, ImageOps

def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def apply_transform(x, transform_matrix, channel_axis=0, fill_mode='nearest', cval=0.):
    """Apply the image transformation specified by a matrix.

    # Arguments
        x: 2D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        The transformed version of the input.
    """
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=1,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x

def transform(x, rotation_deg, fill_mode='nearest', cval=0):
    channel_axis = 0
    row_axis = 1
    col_axis = 2

    # Rotation: Performs a rotation of a Numpy image tensor.
    theta = np.deg2rad(rotation_deg)
    transform_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]

    transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
    matrix_result = apply_transform(x, transform_matrix, channel_axis, fill_mode=fill_mode, cval=cval)

    return matrix_result

image_files = sorted(glob.glob('./data/train/gim/*.tif'))

save_folder = './data/train_ver2/'

num_img = len(image_files)

rotation_deg = [-15, -10, -5, 0, 5, 10, 15]
X_flip = [0, 1]
Y_flip = [0, 1]
#rotation_deg = [-10]
#shift_X_percentage = [0]
#shift_Y_percentage = [0]


for i in xrange(num_img):

    print ('image num: %d' % i)
    for r in xrange(len(rotation_deg)):
        for x in xrange(len(X_flip)):
            for y in xrange(len(Y_flip)):

                image = Image.open(image_files[i])


                if X_flip[x] == 1:
                    image = ImageOps.mirror(image)
                if Y_flip[y] == 1:
                    image = ImageOps.flip(image)

                image = np.expand_dims(np.array(image), axis=2)


                image = np.moveaxis(image, 2, 0) # h, w, channel --> channel, h, w


                image_tmp = transform(image, rotation_deg=rotation_deg[r])


                image_tmp = np.moveaxis(image_tmp, 0, 2)  # channel, h, w  --> h, w, channel


                out_file = '_rotate_%+02d-xflip_%1d-yflip_%1d' % (rotation_deg[r], X_flip[x], Y_flip[y])

                img_name = image_files[i].split('/')[-1].split('.')[0] + out_file + '.tif'

                image_save_path = os.path.join(save_folder,'gim', img_name)


                image_final = Image.fromarray(image_tmp[:, :, 0])


                image_final.save(image_save_path)
