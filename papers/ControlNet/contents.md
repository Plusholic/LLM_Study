# 3 LINE SUMMARY

# Abstract

> 대형 Text-to-Image Diffusion Model에 Spatial Conditioning Contol을 추가하는 신경망 구조인 ControlNet을 제안. ControlNet은 사전 훈련된 Large Diffusion Model의 파라미터를 고정하고, 그 깊고 강력한 인코딩 계층을 재사용하여 다양한 조건 제어를 학습함. ControlNet의 신경망 구조는 초기 파라미터를 0으로 설정한 합성곱 계층을 사용하여, `훈련 중 유해한 노이즈가 모델에 영향을 미치지 않도록 설계`되었음.
> 
- Canny Edge, Depth, Segmentation, Human pose 등을 포함한 다양한 조건 제어를 사용하여 Stable Diffusion 모델을 테스트.
- ControlNet은 작은 데이터셋(<50k)과 큰 데이터셋(>1m) 모두에서 안정적으로 작동하며, 단일 GPU에서 훈련된 모델도 산업 모델과 경쟁력 있는 성능을 보여줌.

# Introduction

> Text-to-Image Diffusion Model의 발전으로 인해 우리는 텍스트 프롬프트만으로도 시각적으로 놀라운 이미지를 생성할 수 있게 됨. 그러나 `이미지의 Spatial Composition 표현에 한계가 있음`. 원하는 이미지를 생성하기 위해서는 프롬프트를 여러 번 수정하고 결과를 확인하며 재수정하는 반복적인 과정이 필요함.
> 
- Large Text-to-Image Diffusion Model에서 특정 조건에 대한 훈련 데이터는 일반 Text-to-Image 훈련 데이터에 비해 상당히 적기 때문에 어려움. 제한된 데이터로 대형 모델을 직접 미세 조정하거나 추가 훈련하는 것은 과적합과 기억 상실 문제를 일으킬 수 있음.
- ControlNet은 사전 훈련된 Large Text-to-Image Diffusion Model에 공간적으로 국한된 조건 제어를 추가하여 모델의 품질과 능력을 유지하면서 다양한 조건을 학습.
- 실험 결과, ControlNet은 단일 또는 다중 조건을 사용하여 Stable Diffusion 모델을 효과적으로 제어할 수 있으며, 다양한 조건 데이터셋에서 안정적으로 작동하였음

# Related Work

## Finetuning Neural Networks

> 추가 훈련 데이터를 사용하여 계속 훈련하는 것은 Overfitting, Mode Collapse 및 망각을 초래할 수 있음. 이러한 문제를 피하기 위한 다양한 미세 조정 전략이 개발되었음.
> 
- **HyperNetwork**
    - 작은 순환 신경망을 훈련시켜 더 큰 신경망의 가중치에 영향을 미치는 방법으로, 원래 자연어 처리(NLP) 분야에서 시작되었음.
    - 이 접근법은 생성적 적대 신경망(GAN)을 사용한 이미지 생성에도 적용되었음.
- **Adapter**
    - 사전 훈련된 트랜스포머 모델에 새로운 모듈 계층을 추가하여 다른 작업에 맞게 조정하는 데 사용됨.
    - 이 방법은 컴퓨터 비전에서 점진적 학습과 도메인 적응에 사용되며, CLIP과 함께 사용되어 사전 훈련된 백본 모델을 다른 작업으로 전이하는 데 활용됨.
    - 최근에는 Vision Transformer와 ViT-Adapter에서도 성공적인 결과를 보였음.
- **Additive Learning**
    - 원래 모델의 가중치를 고정하고 새로운 파라미터를 추가하여 망각을 피함.
    - 이는 학습된 가중치 마스크, 가지치기 또는 하드 어텐션을 사용하여 수행됩니다.
- **Low-Rank Adaptation (LoRA)**
    - 신경망 전체의 복잡도나 크기에 비해 특정 부분의 변화에 필요한 정보의 양이 적다는 것에 기인
- **Zero-Initialized Layers**
    - ControlNet에서 네트워크 블록을 연결하는 데 사용됩니다. 이는 신경망 가중치의 초기화 및 조작에 대한 연구에서 논의된 바 있으며, 훈련 초기에 유해한 노이즈가 모델에 영향을 미치지 않도록 합니다.

## Image Diffusion

**Image Diffusion Model**

- Sohl-Dickstein에 의해 최초로 도입되었고, 최근에는 이미지 생성에 적용되었음.
- Latent Diffusion Model(LDM)은 Latent Image Sapce에서 확산 단계를 수행하여 계산 비용을 줄임.
- Text-to-Image Diffusion Model은 사전 훈련된 언어 모델(예: CLIP)을 통해 텍스트 입력을 잠재 벡터로 인코딩하여 최첨단 이미지 생성 결과를 달성.
- Glide
    - 이미지 생성 및 편집을 지원하는 Text-guided Diffusion Model
- Disco Diffusion
    - 텍스트 프롬프트를 CLIP Guidance와 함께 처리.
- Stable Diffusion
    - Latent Diffusion의 대규모 구현입니다
- Imagen
    - Latent Image를 사용하지 않고 피라미드 구조를 사용하여 직접 픽셀을 확산

**Controlling Image Diffusion Models**

- Text-guided 제어 방법은 Prompt Tuning, CLIP 기능 조작 및 Cross Attention 수정에 중점.
- MakeAScene은 Segmentation Mask를 토큰으로 인코딩하여 이미지 생성을 제어.
- SpaText는 Segmentation Mask를 로컬 토큰 임베딩으로 매핑.
- GLIGEN은 확산 모델의 주의 계층에서 새로운 파라미터를 학습하여 고정된 생성을 수행.
- Textual Inversion과 DreamBooth는 사용자가 제공한 예제 이미지를 사용하여 이미지 확산 모델을 미세 조정하여 생성된 이미지의 콘텐츠를 개인화할 수 있음.
- Ptompt Based 이미지 편집은 프롬프트로 이미지를 조작할 수 있는 실용적인 도구를 제공.

## **Image-to-Image Translation**

- Conditional GANs 및 트랜스포머는 다양한 이미지 도메인 간의 매핑을 학습할 수 있음.
- Taming Transformer는 비전 트랜스포머 접근법을 사용하며, Palette는 조건부 확산 모델로 처음부터 훈련된 모델.
- PITI는 사전 학습 기반 조건부 확산 모델로 Image-to-Image tanslation에 사용됨. 사전 훈련된 GAN을 조작하여 특정 이미지-이미지 작업을 처리할 수도 있음.
- 예를 들어, StyleGANs는 추가 인코더를 통해 제어될 수 있으며, 이외에도 다양한 응용이 연구되고 있습니다.
- Image-to-Image Translation Model이 어떻게 다른 이미지 도메인 간의 매핑을 학습하고, 사전 훈련된 GAN을 조작하여 특정 이미지-이미지 작업을 처리하는 방법을 다룸.

# **Method**

- ControlNet은 대형 사전 훈련된 텍스트-이미지 확산 모델에 공간적으로 국한된, 작업별 이미지 조건을 추가할 수 있는 신경망 구조입니다. 이 섹션에서는 ControlNet의 기본 구조를 소개하고, 이를 Stable Diffusion 모델에 적용하는 방법을 설명합니다. 또한, 훈련 과정과 여러 ControlNets를 구성하는 등의 추가 고려 사항을 다룹니다.
- ControlNet은 신경망의 블록에 추가 조건을 삽입합니다. 여기서 네트워크 블록은 일반적으로 단일 유닛을 형성하기 위해 함께 배치되는 신경층 집합을 의미합니다. ControlNet을 사전 훈련된 블록에 추가할 때, 원래 블록의 파라미터를 고정(freeze)하고, 학습 가능한 복사본을 동시에 생성합니다. 이 학습 가능한 복사본은 외부 조건 벡터를 입력으로 받습니다.
- ControlNet은 대형 모델의 잠긴 파라미터를 보존하면서, 학습 가능한 복사본이 다양한 입력 조건을 처리할 수 있는 강력한 백본(backbone) 역할을 합니다. 이러한 구조는 훈련 초기 단계에서 해로운 노이즈가 모델에 영향을 미치지 않도록 보호합니다.
- 이 섹션에서는 Stable Diffusion 모델에 ControlNet을 추가하는 방법, 훈련 과정, 그리고 추가 조건들이 확산 과정에 어떻게 영향을 미치는지에 대해 설명합니다.

## **ControlNet**

- ControlNet은 신경망 블록에 추가 조건을 주입하여 대형 사전 훈련된 Text-to-Image Diffusion Model을 개선하는 신경망 구조.
- $\mathcal{F}$는 사전 훈련된 신경망 블록, $\Theta$는 파라미터, $x$ 는 인풋 피처맵

$$
y=\mathcal{F}(x;\Theta)
$$

- $x, y$는 2D Feature Map, $\mathbb{R}^{h \times x \times c}$ 이며, $h, x, c$는 Height, Width, Number of Channels in the map
- ControlNet을 사전 훈련된 블록에 추가할 때, 원래 블록의 파라미터 $\Theta$를 고정하고, $\Theta_c$를 사용하는 Trainable Copy 신경망 블록을 동시에 생성함. Trainable Copy는 외부 조건 벡터 $c$ 를 입력으로 받음.
- 
- ControlNet은 대형 모델의 잠긴 파라미터를 보존하면서, Trainable Copy가 다양한 입력 조건을 처리할 수 있는 강력한 백본 역할. Trainable Copy는 1 x 1 Zero Convolution Layer $\mathcal{Z}(\cdot;\cdot)$에 연결됨
- Zero Convolution은 1 x 1 컨볼루션 레이어로 초기 가중치와 편향을 0으로 설정하여 훈련 초기 단계에서 해로운 노이즈가 모델에 영향을 미치지 않도록 함.
- ControlNet을 빌드하기 위해서 2 개의 Zero Convolution 파라미터 $\Theta_{z1}, \Theta_{z2}$를 사용함. 완성된 ControlNet의 수식은 다음과 같음

$$
y_c = \mathcal{F}(x;\Theta)+\mathcal{Z}(\mathcal{F}(x+\mathcal{Z}(c;\Theta_{z1});\Theta_{c});\Theta_{z2})
$$

- $\mathcal{F}(x; Θ)$
    - 원본 네트워크 블록에서 입력 $x$를 변환하여 출력 $y$를 생성함.
- $\mathcal{Z}(c; Θ_{z1})$
    - 조건 $c$ 에 대해 첫 번째 Zero Convolution 레이어를 통해 연산.
- $\mathcal{F}(x + \mathcal{Z}(c; Θ_{z1}); Θ_c)$
    - 첫 번째 Zero Convolution의 output을 특성 맵 $x$에 더한 후, 훈련 가능한 복사본에서 연산.
- $\mathcal{Z}(\mathcal{F}(x + \mathcal{Z}(c; Θ_{z1}); Θ_c); Θ_{z2})$
    - 두 번째 Zero Convolution 레이어를 통해 최종 연산.

- $y_c$는 ControlNet block의 아웃풋이고, 첫 번째 Training Step에서 Zero Convolution Layer의 Weight와 Bias Parameter는 0으로 초기화 되기 때문에 위의 수식 역시 0이 되고, $y_c = y$ 가 됨.
- 이러한 방식으로 유해한 노이즈는 훈련이 시작될 때 훈련 가능한 복사본의 신경망 레이어의 숨겨진 상태에 영향을 미칠 수 없음.
- 또한 $\mathcal{Z}(c;\Theta_{z1})=0$이고 Trainable Copy도 입력 이미지 $x$를 수신하므로 Trainable Copy는 완전한 기능을 갖추고 사전 훈련된 대규모 모델의 기능을 유지하여 추가 학습을 위한 강력한 백본 역할을 할 수 있음.
- 제로 컨볼루션은 초기 훈련 단계에서 무작위 노이즈를 그라데이션으로 제거하여 이 백본을 보호함.

- ControlNet의 주요 아이디어는 원래 모델을 고정된 상태로 유지하면서 조건 제어 기능을 추가하는 것. 이 접근 방식은 대규모 사전 훈련된 모델의 강력한 기능을 활용하면서도 새로운 조건에 맞게 모델을 미세 조정할 수 있게 해줌. 이를 통해 다양한 조건 입력을 효과적으로 처리하고, 모델의 전반적인 성능을 향상시킬 수 있음.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/16a2e13a-9a6d-4189-b385-acd680111454/Untitled.png)

## **ControlNet for Text-to-Image Diffusion**

- Stable Diffusion 모델을 예시로 ControlNet이 대형 사전 훈련된 확산 모델에 조건 제어를 추가하는 방법을 설명합니다. Stable Diffusion은 인코더, 중간 블록, 스킵 연결 디코더로 구성된 U-Net 구조를 따름.
- ControlNet은 Stable Diffusion의 각 인코더 단계에 적용되며, 이를 통해 모델이 다양한 조건 입력을 처리할 수 있음.
- ControlNet 구조는 인코더의 12개 블록과 1개의 중간 블록에 학습 가능한 복사본을 생성하고, 원래 잠긴 블록과 1x1 합성곱 계층으로 연결함.
- 이 학습 가능한 복사본은 조건 벡터를 입력으로 받아, 모델의 원래 기능을 유지하면서도 추가적인 조건 제어 기능을 학습할 수 있게 함.
- Stable Diffusion은 잠재 이미지 공간을 사용하여 훈련 데이터를 인코딩합니다. ControlNet을 Stable Diffusion에 추가하기 위해 입력 조건 이미지를 작은 네트워크를 통해 잠재 공간 벡터로 변환함.
- ControlNet을 Stable Diffusion에 추가하려면, 먼저 각 입력 조건 이미지(예: 엣지, 포즈, 깊이 등)를 512 × 512 크기에서 Stable Diffusion 크기와 일치하는 64 × 64 특징 공간 벡터로 변환해야 함.
- 특히, 4개의 컨볼루션 레이어(각각 16, 32, 64, 128 채널을 사용하고, Gaussian 가중치로 초기화된 후 전체 모델과 함께 훈련된 ReLU 활성화 및 4 × 4 커널과 2 × 2 스트라이드)를 가진 작은 네트워크 E(·)를 사용하여 이미지 공간 조건 $c_i$를 특징 공간 조건 벡터 $c_f$로 인코딩함.

$$
c_f = \Epsilon(c_i)
$$

- 이 조건 벡터 $c_f$는 ControlNet으로 전달됨.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/76391581-1c03-4dd7-b835-87aeadab73f6/Untitled.png)

## **Training**

1. **노이즈 추가**
    1. 이미지 확산 알고리즘은 입력 이미지 $z_0$에 점진적으로 노이즈를 추가하여 $z_t$라는 노이즈 이미지로 변환합니다. 여기서 $t$는 노이즈가 추가된 횟수를 나타냄.
2. **학습 목표**
    1. 이미지 확산 알고리즘은 시간 단계 $t$, 텍스트 프롬프트 $c_t$, 특정 조건 $c_f$를 포함한 조건 세트를 사용하여 네트워크 $\epsilon_\theta$를 학습함. 이 네트워크는 노이즈 이미지를 예측하여 노이즈를 제거함.

$$
\mathcal{L}=\mathbb{E}_{z_0,t,ct,cf,ϵ∼\mathcal{N}(0,1)}[∥ϵ−ϵθ(zt,t,ct,cf)∥^2_2]
$$

- 훈련 과정에서 텍스트 프롬프트 $c_t$의 50%를 빈 문자열로 대체함. 이를 통해 모델이 텍스트 프롬프트 없이도 입력 조건 이미지를 직접 인식할 수 있게 함.
- 제로 합성곱 계층을 사용하여 훈련 초기 단계에서 모델에 해로운 노이즈가 추가되지 않도록 함. 이를 통해 모델은 훈련 내내 고품질 이미지를 예측할 수 있음.
- 모델은 조건 이미지를 따르는 방법을 점진적으로 학습하지 않고, 특정 훈련 단계에서 갑작스럽게 성공합니다. 이 현상을 "갑작스러운 수렴 현상"이라고 합니다.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/63719c45-7121-4b8c-936a-be6e06269c7d/Untitled.png)

## **Inference**

ControlNet의 추론 과정은 다음과 같은 방식으로 진행됨

1. **Classifier-Free Guidance (CFG) Resolution Weighting**:
    - Stable Diffusion 모델은 고품질 이미지를 생성하기 위해 Classifier-Free Guidance (CFG)를 사용. CFG는 모델의 최종 출력, 무조건적 출력, 조건적 출력, 사용자 지정 가중치로 구성됨.
    
    $$
    \epsilon_{\text{prd}} = \epsilon_{\text{uc}} + \beta_{\text{cfg}}(\epsilon_{\text{c}} - \epsilon_{\text{uc}})
    $$
    
    - $\epsilon_{\text{prd}}$ : 모델의 최종 출력 (예측된 노이즈)
    - $\epsilon_{\text{uc}}$ : 무조건적인(unconditional) 출력 (텍스트 조건이 없는 상황)
    - $\epsilon_{\text{c}}$ : 조건부(conditional) 출력 (텍스트 조건이 있는 상황)
    - $\beta_{\text{cfg}}$ : 사용자 지정 가중치 (정도 조정을 위한 파라미터)
    - ControlNet의 조건 이미지를 CFG의 무조건적 출력과 조건적 출력 모두에 추가하거나, 조건적 출력에만 추가할 수 있음. 조건 이미지를 두 출력 모두에 추가하면 CFG 가이던스가 제거되며, 조건적 출력에만 추가하면 가이던스가 매우 강해짐.
    - 조건 이미지를 먼저 조건적 출력에 추가한 후, ControlNet과 Stable Diffusion 간의 각 연결에 해상도에 따른 가중치를 곱해서 CFG 가이던스의 강도를 줄일 수 있습니다.
2. **ControlNets**:
    - 여러 조건 이미지를 단일 Stable Diffusion 인스턴스에 적용하기 위해, 해당 조건 이미지의 ControlNet 출력을 Stable Diffusion 모델에 직접 추가할 수 있습니다.
    - 경우 1: 조건부 이미지가 $\epsilon_{\text{uc}}$와 $\epsilon_{\text{c}}$에 모두 추가되면 CFG 지침이 완전히 사라짐.
    - 경우 2: 조건부 이미지가 오직 $\epsilon_{\text{c}}$에만 추가되면 지침이 너무 강해짐.
    - 조건부 이미지를 먼저 $\epsilon_{\text{c}}$에 추가하고 각 블록의 해상도에 따라 가중치 $w_i$ 를 부여:
        - $w_i = 64 / h_i$
        - 예시: $h_1 = 8, h_2 = 16, ... ,h_{13} = 64$
    - CFG 지침 강도를 줄여 균형 잡힌 결과 도출.

## **Qualitative Results**

- Figure 1과 Figure 7에서 여러 프롬프트 설정에서 생성된 이미지를 제시하며, ControlNet이 다양한 조건에서도 안정적으로 작동함을 강조.
- ControlNet은 Canny Edge, Depth Map, Normal Map, HED Soft Edge, Segmentation Map, Human Pose, User Sketches 등을 조건으로 사용하여 이미지를 생성함.
- ControlNet은 입력 조건 이미지의 내용을 잘 해석하여 다양한 조건에서 고품질 이미지를 생성할 수 있음

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/bd173328-248f-4c2f-943c-a4cbbb7cced3/Untitled.png)

## **Ablative Study**

- Ablative Study 섹션에서는 ControlNet의 다양한 구조적 변형을 비교하여 성능을 평가합니다. 주요 실험은 다음과 같습니다:
1. **Zero Convolutions 교체**: Zero Convolutions를 표준 Convolution Layers로 교체한 경우를 테스트합니다. Gaussian 가중치로 초기화된 표준 합성곱 계층을 사용하여 성능을 비교합니다.
2. **ControlNet-lite**: 각 블록의 Trainable Copy를 단일 합성곱 계층으로 대체한 ControlNet-lite 구조를 테스트합니다.
- 4가지 프롬프트 설정을 통해 실제 사용자 행동을 시뮬레이션합니다:
    - **No Prompt :** 조건 이미지만을 사용하여 이미지를 생성.
    - **Insufficient Prompts :** 조건 이미지를 완전히 설명하지 않는 프롬프트를 사용.
    - **Conflicting Prompts :** 조건 이미지의 의미를 변경하는 프롬프트를 사용.
    - **Perfect Prompts :** 필요한 내용 의미를 설명하는 프롬프트를 사용.
- 실험 결과, ControlNet은 4가지 설정 모두에서 성공적으로 조건 이미지를 반영함.
- 그러나 ControlNet-lite는 Insufficient Prompt와 No Prompt에서 실패하며, Zero Convolutions를 대체한 경우 성능이 ControlNet-lite 수준으로 떨어짐.
- 이는 Trainable Copy의 사전 훈련된 백본이 훈련 중 손상되었음을 시사.
- 이를 통해 ControlNet의 구조적 구성 요소가 모델의 성능에 미치는 영향을 확인하고, 최적의 구조를 찾는 데 도움을 줍니다.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/8d144b67-b6a4-4eca-890a-da09ade064f0/Untitled.png)

## **Quantitative Evaluation**

정량적 평가 섹션에서는 ControlNet의 성능을 다양한 기준으로 평가

1. **User study**
    - 20개의 손으로 그린 스케치를 샘플링하고, 5개의 다른 방법과 비교합니다: PITI, Sketch-Guided Diffusion, ControlNet-lite, ControlNet 등.
    - 12명의 사용자를 초청하여 결과 이미지의 품질과 조건 충실도를 평가.
    - 사용자 선호도 랭킹(Average Human Ranking, AHR)을 사용하여 각 방법의 결과를 평가. 사용자는 결과를 1에서 5까지의 스케일로 평가.
2. **Comparison to industrial models**
    - Stable Diffusion V2 Depth-to-Image 모델과 비교.
    - ControlNet은 단일 GPU에서 훈련되었으며, 200k의 훈련 샘플만을 사용.
    - 12명의 사용자가 생성된 200개의 이미지를 보고, 어떤 모델이 생성했는지 구분하도록 함.
    - 평균 정밀도는 0.52로, 두 방법이 거의 구분되지 않는 결과를 나타냄.
3. **Condition reconstruction and FID score**
    - ADE20K 테스트 세트를 사용하여 조건 충실도를 평가.
    - OneFormer 세그멘테이션 방법을 사용하여 생성된 이미지의 세그멘테이션을 다시 검출하고, 재구성된 교차 영역 합집합(IoU)을 계산.
    - FID(Frechet Inception Distance)를 사용하여 각 방법의 분포 거리를 측정.
    - 텍스트-이미지 CLIP 점수와 CLIP 미적 점수도 함께 평가.

## **Comparison to Previous Methods**

- PITI (Pretraining-based Image-to-Image Translation)
    - 사전 학습 기반 이미지-이미지 변환 모델.
- Sketch-Guided Diffusion
    - 스케치를 사용하여 이미지 생성을 안내하는 확산 모델.
- Taming Transformers
    - 비전 트랜스포머 접근법을 사용하는 모델.
- ControlNet은 다양한 조건 입력에 대해 더 나은 성능을 보여줌.
- 조건 이미지의 내용을 잘 해석하고, 고품질의 이미지를 생성함.
- 다른 방법들에 비해, ControlNet은 더 정교한 세부 사항을 포함한 이미지를 생성함.
- ControlNet은 여러 조건을 조합하여 복잡한 이미지 구성을 생성할 수 있음.
- 다양한 조건 입력에 대해 안정적으로 작동하며, 사용자 선호도 평가에서 높은 점수를 받았음.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/2e26fef1-c600-4e69-84f6-9dcec0a53ae7/Untitled.png)

## **Discussion**

- **Influence of training dataset sizes**
    - **1k 이미지를 사용한 훈련**: ControlNet은 단 1k(천 장)의 이미지로도 훈련이 가능합니다. 이 단계에서도 모델은 무너지지 않고 견고함을 유지함.
    - **50k 이미지를 사용한 훈련**: 데이터셋 크기를 증가시키면 성능이 향상됩니다. 50k 이미지를 사용하면 더 상세하고 정확한 결과를 얻을 수 있음.
    - **3m 이미지를 사용한 훈련**: 3백만 장의 이미지를 이용한 훈련에서는 더 많은 데이터가 주어졌을 때 ControlNet의 학습 능력이 확장됨을 확인할 수 있음.
- **Capability to interpret contents**
    - 사용자가 객체 내용을 명시하지 않고 애매한 프롬프트를 입력하면, ControlNet은 입력된 형상(shape)을 해석하여 이미지를 생성하려고 시도함.
    - 예를 들어, "a high-quality and extremely detailed image"라는 프롬프트로 다양한 내용을 해석한 이미지를 생성함.
- **Transferring to community models**
    - ControlNets는 사전훈련된 Stable Diffusion(SD) 모델의 네트워크 토폴로지를 변경하지 않음.
    - Comic Diffusion이나 Protogen 3.4 등 Srable Diffusion Model 커뮤니티에서도 추가적인 훈련 없이 바로 적용할 수 있음.
    - Comic Diffusion과 Protogen 3.4 모델에서도 ControlNets가 적용된 결과를 확인할 수 있음.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/36213ce7-a824-43af-99e5-835885ac22f2/Untitled.png)

## **Conclusion**

- ControlNet은 대형 사전 훈련된 텍스트-이미지 확산 모델에 공간적 조건 제어를 추가하는 새로운 방법을 제안.
- 다양한 조건 입력을 사용하여 안정적으로 작동하며, 고품질 이미지를 생성할 수 있음을 입증함.
- 실험을 통해 ControlNet의 우수한 성능을 확인했으며, 사용자 연구에서도 높은 평가를 받았음.
- ControlNet은 대형 모델의 파라미터를 고정하면서 조건 제어 기능을 추가하여, 다양한 입력 조건을 효과적으로 학습할 수 있는 강력한 백본 역할.