import glob

img_path = './val/img/*.tif'
label_path = './val/mask/*.tif'
gim_path = './val/gim/*.tif'

output_filename = './train_ver2/val_gim.txt'

img_list = sorted(glob.glob(img_path))
mask_list = sorted(glob.glob(label_path))
gim_list = sorted(glob.glob(gim_path))

file_num = len(img_list)

with open(output_filename, 'w') as f:
    for s in range(file_num):
        f.write(img_list[s][2:] + '\t' + mask_list[s][2:] + '\t' + gim_list[s][2:] + '\n')