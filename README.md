# KOMPSAT-5 (K5) 영상을 활용한 딥러닝 기반 수계 검출 알고리즘 적용
![image](https://user-images.githubusercontent.com/26617052/100533894-de18c280-324c-11eb-89b2-57352d229753.png)

다목적실용위성 5호(KOMPSAT-5)라고도 불리는 아리랑 5호 위성은 SAR 센서를 이용해 신호를 전송하고 지표면에서 반사된 신호를 다시 수신하여 영상을 생성한다. SAR 센서는 일반 광학 센서와는 달리 구름, 비와 같은 기상 조건에 관계없이 전천후로 지구 관측이 가능하기에 해양 유류사고, 화산 폭발 같은 지표면 모니터링 및 관리에 용이하다. 그 중, SAR 영상에서의 수계 지역 검출은 우리 생활에 밀접한 영향을 미친다. 집중 호우 및 태풍으로 인한 수재해 피해 전/후의 광역적인 수면적 변화 분석 등에 활용 가능하다. 

전통적인 SAR 영상 기반 수계 탐지 기법 중 하나로 SAR 영상의 후방산란계수(Backscattering coefficient)를 활용한 Gamma-distribution fitting 기법이 있다. 이는 Multi-looking 한 SAR 영상에서 수계의 후방산란계수의 확률 분포를 감마 분포로 가정하여 최적의 임계값을 설정하여 수계 부분과 그렇지 않은 부분을 구분하는 기법이다. 이 기법은 수계의 형태(평지, 도심지, 산악지 등)에 따라 분류 오류가 나타나기도 하며, 하나의 SAR 영상 마다 임계값을 수작업으로 다 다르게 설정해야한다는 단점이 있다. 

이러한 전통적인 수계 탐지 기법의 단점을 극복하며, 다량의 SAR 영상을 자동으로 빠르게 처리하기 위해서 딥러닝의 적용은 필수불가결하다. 이에 한국항공우주연구원 인공지능연구실에서는 FCN, U-Net, HRNet 등 대표적인 딥러닝 이미지 분할 기법 활용과 더불어 SAR 영상의 지역적 입사각 정보, 레이더 그림자 및 레이오버 정보를 담고 있는 GIM(Geocoded Incidence angle Mask) 정보를 활용하여 수계 검출 자동화 연구를 진행 중이다.

```
git clone https://github.com/kari-ai/kari_water_seg.git
```

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

## Anaconda로 yml 파일을 이용한 환경 설정
```
conda env create -f water_seg.yml
```

## Dataset
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
