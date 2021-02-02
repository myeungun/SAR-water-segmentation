# Deep Learning based Water Segmentation Using KOMPSAT-5 SAR Images
![image](https://user-images.githubusercontent.com/26617052/100533894-de18c280-324c-11eb-89b2-57352d229753.png)

Depending on weather conditions, optical satellites might not acquire image information of a region of interest (ROI). This is a major drawback in emergency disaster which require real-time monitoring and damage analysis. In particular, one of the most serious disasters, floods, always accompany clouds. As a result, there are difficulties in flood detection, i.e., water detection, using optical satellite images. While, Synthetic Aperture Radar (SAR) satellite has the advantage of acquiring images regardless of weather conditions such as cloud and rain. Therefore, we can effectively perform flood monitoring and damage analysis for the ROI through water detection using SAR satellite images. 

In this paper, we propose a deep learning-based water segmentation using KOrean Multi-Purpose SATellite (KOMPSAT-5) images. To efficiently develop the deep learning-based model, we create a SAR water dataset for over 3,000 sheets based on KOMPSAT-5. And We perform water segmentation using representative deep learning-based segmentation models such as Fully Convolutional Networks (FCN), U-Net, DeepUNet, and High Resolution Network (HRNet). Experimental results show that HRNet performs the highest accuracy, i.e, this model achieves more than 80\% IoU (Intersection over Union). 

```
git clone https://github.com/kari-ai/kari_water_seg.git
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
- Training dataset 위치 [[Link]](https://arxiv.org/pdf/1505.04597.pdf)
```
/SAR-water-segmentation/data/train_full/
```
- Test dataset 위치 [[Link]](https://arxiv.org/pdf/1505.04597.pdf)
```
/SAR-water-segmentation/data/val_full/
```

## Pretrained model
- HRNetV2를 새롭게 training을 할 경우
ImageNet dataset으로 미리 학습된 모델 위치 [[Link]](https://arxiv.org/pdf/1505.04597.pdf)
```
/SAR-water-segmentation/HRNet/tools/pretrained_models/
```

- HRNetV2를 test 할 경우
K5 Training dataset으로 학습된 모델 위치 [[Link]](https://arxiv.org/pdf/1505.04597.pdf)
```
/SAR-water-segmentation/HRNet/tools/output/K5/K5/models/best.pth
```

# 딥러닝 알고리즘 1: HRNet(High Resolution Network) [[Paper]](https://arxiv.org/pdf/1908.07919.pdf), [[Code]](https://github.com/HRNet/HRNet-Semantic-Segmentation)
![image](https://user-images.githubusercontent.com/26617052/100533991-f50be480-324d-11eb-8802-d470b8e5b012.png)
## Training code 실행
```
cd /SAR-water-segmentation/HRNet/tools/
python main_HRNet_train.py
```
## Test code 실행
```
cd /SAR-water-segmentation/HRNet/tools/
python main_HRNet.py
```


# 딥러닝 알고리즘 2: FCN(Fully Convolutional Network) [[Paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf)
![image](https://user-images.githubusercontent.com/26617052/100533941-7020cb00-324d-11eb-976c-863e07dc98c5.png)
## Test code 실행
- Backbone: VGG16
```
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit 
python main_FCN_VGG16.py
```
- Backbone: VGG19
```
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit 
python main_FCN_VGG19.py
```

# 딥러닝 알고리즘 3: U-Net [[Paper]](https://arxiv.org/pdf/1505.04597.pdf)
![image](https://user-images.githubusercontent.com/26617052/100533952-89c21280-324d-11eb-820b-d4378713b470.png)
## Test code 실행
```
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit 
python main_UNet.py
```

# 딥러닝 알고리즘 4: Deep U-Net [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8370071)
![image](https://user-images.githubusercontent.com/26617052/100533982-dad20680-324d-11eb-9964-85a6f9946985.png)
## Test code 실행
```
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit 
python main_DeepUNet.py
```
