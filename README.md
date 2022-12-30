# FindColorProject

_딥러닝을 활용한 흑백 사진 복원 프로젝트_
</br>

## 연구 배경 및 목표

오래된 기록물, 콘텐츠의 흑백 이미지에 컬러를 입히는 AI 복원 기술은 **사회/경제적 가치 측면**에서 필요성이 높아지고 있다.</br>
흑백 이미지를 **시대적 배경 및 상황**(전쟁, 해방, 시위 등)에 맞는 **컬러이미지로 복원**하는 AI 기술 및 UI/UX 개발


## Deep leanring

### 데이터셋

`gettyImages, ActivityNet, ImageNet`</br>
학습 데이터 : 30만장</br>
검증 데이터 : 3만장</br>
테스트 데이터 : 3천장</br>

### 모델

Context-adaptive Colorization

<img src="https://user-images.githubusercontent.com/86578246/210068754-be027cfb-3295-4863-ba00-8a9795f87a22.png" height="200px">

### Hyperparameters

Batch Size : 256</br>
Epoch : 50</br>
Learning Rate : 1e-4, Adam Optimizer</br>

## Results

### Evaluation

PSNR

<img src="https://user-images.githubusercontent.com/86578246/210068936-c32232f0-b4d8-4bde-97f9-c66f60f5cc8c.png" height="200px">

Survey

<img src="https://user-images.githubusercontent.com/86578246/210068975-bd801f39-0873-4e32-a169-6433b57d7a71.png" height="200px">

### WebApp

![Web](https://user-images.githubusercontent.com/86578246/210068862-2632dfac-04df-472b-9da5-5bc3835701e8.png)

Demo Video

<https://youtu.be/H4TLLVDJ_A8>

### Local UI

![local demo](https://user-images.githubusercontent.com/86578246/210069274-8d27236e-6379-42e2-b5c4-11a12a8ffaa0.gif)

