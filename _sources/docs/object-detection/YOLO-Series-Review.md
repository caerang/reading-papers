# YOLO Series Review

1-State Detector: 특징 추출과 객체 분류를 동시에(한 번에) 처리하는 방법, 대표적으로 YOLO 계열과 SSD 계열이 있음

# YOLOv1: **You Only Look Once: Unified, Real-Time Object Detection**

YOLO는 객체 검출 문제를 이미지 픽셀에서 경계 상자 좌표와 분류 확률로 직접적으로 변환하는 단일 회귀 문제로 정의 (박스 검출과 분류를 동시에 처리 함)

![**YOLOv1 Detection System**](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8d42649d-e86e-449a-8002-a22f2ffda50e/Untitled.png)

**YOLOv1 Detection System**

### 동시 처리 방식이 갖는 장점

나중에 논문에서 정리

## 경계 상자 예측 방식

1. 입력 이미지를 S x S grid로 분할한다.
2. 객체의 중심이 grid cell에 포함되면, grid cell은 cell에 포함된 객체를 검출할 의무가 있다.
3. 각 그리드 셀Grid Cell은 B 개의 경계 상자Bounding Box를 예측하고 경계 상자에 대한 신뢰도Confidence Score를 계산한다.
    1. Confidence Score는 Box가 객체를 포함할 가능성(신뢰도)가 얼마나 높은지, 예측한 영역이 얼마나 정확한지 나타내는 지표이다.
    2. Confidence Score는 Pr(Object) * IOU
    3. 그리드 셀에 객체가 없으면 Confidence Score는 0
    4. 그리드 셀에 객체가 있는 경우 예측 결과와  GT의 IOU와 Confidence Score가 같기를 기대
4. 각 경계 상자는 5개의 예측값(x, y, w, h, confidence)을 갖는다.
    1. x, y는 경계 상자의 중심 좌표를 나타내며 각 그리드 셀의 경계에 상대적인 값이다.

![출처: [https://leedakyeong.tistory.com/230](https://leedakyeong.tistory.com/230)](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/387aad55-982a-4d78-a46f-e9980746263d/Untitled.png)

출처: [https://leedakyeong.tistory.com/230](https://leedakyeong.tistory.com/230)

1. 각 그리드 셀은 객체가 있을 경우 각 클래스에 해당할 조건부 확률을 계산한다. (B 개의 경계 상자를 예측하더라도 클래스에 해당할 확률은 그리드 셀 당 하나의 확률만 계산)
    1. 테스트 시에는 클래스 조건부 확률과 각 경계 상자의 예측 신뢰도를 곱해서 각 상자가 특정 클래스에 속할 신뢰도를 계산
    
    ![출처: 논문](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4fec8119-0688-445d-a167-5bc8d3a0168c/Untitled.png)
    
    출처: 논문
    

## NMS

각 그리드 셀에서 B개의 경계 상자를 예측한다. 객체가 여러 개의 경계 상자에 포함될 수 있기 때문에 GT에 가장 가까운 경계 상자를 찾기 위해 NMS를 적용한다.

![출처: [https://oi.readthedocs.io/en/latest/computer_vision/object_detection/yolo.html](https://oi.readthedocs.io/en/latest/computer_vision/object_detection/yolo.html)](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8f1d87b5-bad5-4a6c-9137-b2003ac5b5a9/Untitled.png)

출처: [https://oi.readthedocs.io/en/latest/computer_vision/object_detection/yolo.html](https://oi.readthedocs.io/en/latest/computer_vision/object_detection/yolo.html)

## 네트워크 구조Network Architecture

## Loss

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/cc4e698d-9f2e-44e9-b3e2-acf983d37a61/Untitled.png)

## 문제점

- 그리드라는 공간적인 제약을 두고 경계 상자를 예측하기 때문에 가까이 있는 객체를 검출할 수 있는 수가 제한된다. (B 개의 경계 상자)
    - 특히 그룹으로 여러 개의 작은 객체가 있는 경우 검출에 문제가 된다. (예, 새 떼Flock of birds)
- 데이터에서 예측할 경계 상자의 크기를 학습하기 때문에 학습 데이터에서 보지 못한 크기의 경계 상자나 새로운 종횡비의 경계 상자는 예측을 잘 하지 못한다.
- Pooling layer를 많이 사용해서  Downsampling이 많아지면 Feature의 질이 다소 떨어진다.(Coarse features)
- 하나의 그리드에 여러 개의 객체가 있는 경우 문제가 된다.
    - (공통) Output 크기를 3x3에서 19x19로 더 늘리면 하나의 그리드에 여러 개의 객체가 올 확률을 낮출 수 있다.
    - (Class가 같은 경우) 하나의 그리드에서 2개의 경계 상자를 감지할 수 있게 만들어 일부 해결 가능하다. (2개가 넘어가면?)
    - (Class가 다른 경우) 그리드별로 하나의 조건부 클래스 확률을 계산하는데?
- 하나의 객체가 여러 개의 그리드에서 감지되는 경우는 NMS로 일부 해결 할 수 있다.

# YOLOv2: YOLO9000: Better, Faster, Stronger

## Better

- YOLO를 Fast R-CNN과 비교 했을 때 Localization에서 더 많은 에러를 보임
- Region Proposal-Based 방법에 비해 상대적으로 낮은 Recall을 보임

### Batch Normalization

- BN을 적용하면 다른 형태의 Regularization 기법을 사용하지 않아도 학습이 빠르게 수렴되게 함
- 모든 Convolutional Layer에 BN을 적용
- BN은 모델을 regularize 하는데 도움이 됨

### High Resolution Classifier

- 기존 SOTA 모델(그 당시만 해도… 지금도 그런가?)은 ImageNet으로 학습된 classifier를 사용
- Classifier는 256x256(AlexNet), 224x224(YOLOv2)로 학습을 하는데, detection을 위해서는 448을 사용
    - 네트워크가 객체 검출 학습으로 전환과 새로운 해상도의 입력에 대한 적응을 동시에 해야 함
- Classification 네트워크는 448로 파인 튜닝 → 네트워크가 큰 입력 해상도에 잘 동작하게 적응할 수 있도록 함

### Convolutional With Anchor Boxes