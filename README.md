# Deep Learning based Water Segmentation Using KOMPSAT-5 SAR Images
## Some results
<img src="https://user-images.githubusercontent.com/26617052/106559149-50b55f80-6568-11eb-9fcc-f78bb63819b9.JPG"  width="450">
- This shows the performance result of HRNet with our KOMPSAT-5 dataset. The images on the left are ground truth, and the images on the right are predicted results. The IoU of the upper and lower images are 94.76% and 97.45%, respectively.

## Abstract
Depending on weather conditions, optical satellites might not acquire image information of a region of interest (ROI). This is a major drawback in emergency disaster which require real-time monitoring and damage analysis. In particular, one of the most serious disasters, floods, always accompany clouds. As a result, there are difficulties in flood detection, i.e., water detection, using optical satellite images. While, Synthetic Aperture Radar (SAR) satellite has the advantage of acquiring images regardless of weather conditions such as cloud and rain. Therefore, we can effectively perform flood monitoring and damage analysis for the ROI through water detection using SAR satellite images. 

In this paper, we propose a deep learning-based water segmentation using KOrean Multi-Purpose SATellite (KOMPSAT-5) images. To efficiently develop the deep learning-based model, we create a SAR water dataset for over 3,000 sheets based on KOMPSAT-5. And We perform water segmentation using representative deep learning-based segmentation models such as Fully Convolutional Networks (FCN), U-Net, DeepUNet, and High Resolution Network (HRNet). Experimental results show that HRNet performs the highest accuracy, i.e, this model achieves more than 80\% IoU (Intersection over Union). 

## How to use it
- First of all, clone the code
```
git clone https://github.com/myeungun/SAR-water-segmentation.git
```

## Prerequisites
- CUDA 10.0
- Python 3.7.7
- Tensorflow 1.14.0
- PyTorch 0.4.1
- Keras 2.3.

## Install all the python dependencies using pip:
```
cd /SAR-water-segmentation/HRNet/
pip install –r requirements.txt
```

## Data Preparation
Please download dataset below and put them in each path
- Training dataset [[Link]](https://drive.google.com/file/d/1Ugk6c_iadvlD-ycxNQlw9SHLDDqNAn1f/view?usp=sharing)
```
/SAR-water-segmentation/data/train_full/
```
- Test dataset [[Link]](https://drive.google.com/file/d/1MbyK4ljGmin5JeRroO80qTicbYfxGVAu/view?usp=sharing)
```
/SAR-water-segmentation/data/val_full/
```

## Pretrained model
- When training the HRNet, download this model and put it in the path below
This model is a pre-trained model with ImageNet dataset, NOT our KOMPSAT-5 dataset. [[Link]](https://drive.google.com/file/d/1euYbOpJbs9di7W8IO4_hDizN_EoRWfAA/view?usp=sharing)
```
/SAR-water-segmentation/HRNet/tools/pretrained_models/
```

- When testing the HRNet, download this model and put it in the path below
This model is a pre-trained model with ImageNet dataset, AND our KOMPSAT-5 dataset. [[Link]](https://drive.google.com/file/d/1gfLbsv9_6ZNtG7K3bmUf2r1Ig0CfQHIo/view?usp=sharing)
```
/SAR-water-segmentation/HRNet/tools/output/K5/K5/models/best.pth
```

# HRNet(High Resolution Network) [[Paper]](https://arxiv.org/pdf/1908.07919.pdf), [[Code]](https://github.com/HRNet/HRNet-Semantic-Segmentation)
- HRNet is one of the latest models for learning-based image segmentation. A typical features of this model is that while the model is being trained, (1) the features of the high-resolution are retained while simultaneously extracting the low-resolution features in parallel. (2) This model repeatedly exchanges feature information between different resolutions. (3) Since this has a large number of layers and a lot of weights to be stored, it consumes a lot of memory and the speed of processing one image is relatively slow, however its performance is superior compared to other previous models.

## Training code
```
cd /SAR-water-segmentation/HRNet/tools/
python main_HRNet_train.py
```
## Test code
```
cd /SAR-water-segmentation/HRNet/tools/
python main_HRNet.py
```


# FCN(Fully Convolutional Network) [[Paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf)
- FCN is one of the most classic and representative models of image semantic segmentation methods. It is a model modified for the purpose of image semantic segmentation by changing the fully connected layer of the last layer into a convolutional layer in the existing classification model such as VGG16. Representative characteristics of this model, (1) by adding up sampling layers to the coarse feature map predicted through the convolutional layer, it is possible to predict dense features and restore back to the original image size. (2) By adding the skip architecture, local information of the shallow layer and semantic information of the deep layer can be combined.

## Test code 
- Backbone: VGG16
```
python main_FCN_VGG16.py
```
- Backbone: VGG19
```
python main_FCN_VGG19.py
```

# U-Net [[Paper]](https://arxiv.org/pdf/1505.04597.pdf)
- U-Net is an end-to-end FCN-based model proposed for image semantic segmentation. This model consists of the contracting path to obtain the overall context information of the input image and the expanding path to obtain the dense prediction from the coarse map in a symmetrical form. Because of this symmetry, the shape of the network is in the form of 'U' and is named U-Net.

## Test code
```
python main_UNet.py
```

# Deep U-Net [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8370071)
- Compared with U-Net, DeepUNet is a deeper model with more layers. Unlike U-Net, this model is characterized by an added 'plus layer'. The plus layer connects two adjacent layers, while the skip architecture commonly used in FCN, U-Net, and DeepUNet connects the shallow layer and the deep layer. This plus layer has the effect of preventing the loss of deep network from expanding to infinity and the model from getting trapped into the local optima.

## Test code 
```
python main_DeepUNet.py
```
