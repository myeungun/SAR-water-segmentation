# 아리랑 위성 5호 영상(KOMPSAT-5)을 활용한 
# 딥러닝 기반 수계 검출 알고리즘의 적용
![image](https://user-images.githubusercontent.com/26617052/100533894-de18c280-324c-11eb-89b2-57352d229753.png)

## 개발 환경
- CUDA 10.0
- Python 3.7.7
- Tensorflow 1.14.0
- PyTorch 0.4.1
- Keras 2.3.

## Dependencies 설치
```
cd /SAR-water-segmentation/HRNet/
pip install –r requirements.txt
```

## Dataset
### Training dataset 위치
```
/SAR-water-segmentation/data/train_full/
```
### Test dataset 위치
```
/SAR-water-segmentation/data/test_full/
```

# 딥러닝 알고리즘 1: FCN(Fully Convolutional Network) [[Paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf)
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

# 딥러닝 알고리즘 2: U-Net [[Paper]](https://arxiv.org/pdf/1505.04597.pdf)
![image](https://user-images.githubusercontent.com/26617052/100533952-89c21280-324d-11eb-820b-d4378713b470.png)
## Test code 실행
```
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit 
python main_UNet.py
```

# 딥러닝 알고리즘 3: Deep U-Net [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8370071)
![image](https://user-images.githubusercontent.com/26617052/100533982-dad20680-324d-11eb-9964-85a6f9946985.png)
## Test code 실행
```
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit 
python main_DeepUNet.py
```

# 딥러닝 알고리즘 4: HRNet(High Resolution Network) [[Paper]](https://arxiv.org/pdf/1908.07919.pdf), [[Code]](https://github.com/HRNet/HRNet-Semantic-Segmentation)
![image](https://user-images.githubusercontent.com/26617052/100533991-f50be480-324d-11eb-8802-d470b8e5b012.png)
## Test code 실행
```
cd /SAR-water-segmentation/HRNet/tools/
python main_HRNet.py
```
