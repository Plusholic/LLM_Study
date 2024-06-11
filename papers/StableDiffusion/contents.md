# Abstract

> 이미지 생성을 위한 최첨단 방법 중 하나인 확산 모델(Diffusion Models, DMs)을 소개. 기존의 DMs는 주로 픽셀 공간에서 작동하여 높은 GPU 자원을 소비하고, 추론 과정이 복잡하여 비싸다는 단점. 이러한 문제를 해결하기 위해, 이 논문은 강력한 사전 학습된 오토인코더의 잠재 공간(latent space)에서 확산 모델을 적용하는 방법을 제안. 이를 통해 복잡성을 줄이고 세부 사항을 보존하여 시각적 충실도를 크게 향상시킬 수 있음. 또한, Cross-Attention Layers를 모델 구조에 도입하여 텍스트나 바운딩 박스와 같은 일반적인 조건 입력을 위한 강력하고 유연한 생성기를 구현. 제안된 잠재 확산 모델(Latent Diffusion Models, LDMs)은 이미지 인페인팅(image inpainting), 클래스 조건부 이미지 생성(class-conditional image synthesis) 및 다양한 작업에서 새로운 최첨단 성능을 달성하면서도, 기존의 픽셀 기반 DMs에 비해 계산 요구 사항을 줄임.
> 
- Latent Space에서 Diffusion Model을 훈련하여 복잡성을 줄이면서도 세부 사항을 보존하는 방법 제시.
- Cross Attention Layer를 도입하여 텍스트나 경계 상자 등 다양한 조건 입력을 처리할 수 있는 강력한 제너레이터 구현.
- 기존의 픽셀 기반 확산 모델에 비해 계산 요구 사항을 줄이면서도 높은 시각적 품질을 유지하는 모델 개발.

# Introduction

> 이미지 합성은 최근 컴퓨터 비전 분야에서 매우 놀라운 발전을 이루었으나, 높은 계산 자원을 요구하는 분야 중 하나. 특히, 복잡하고 자연스러운 고해상도 장면의 합성은 수십억 개의 매개변수를 포함하는 Autoregressive 트랜스포머 모델의 확장을 통해 이루어짐. 반면, 생성적 적대 신경망(GANs)의 결과는 제한된 변동성을 가진 데이터에 국한되며, 복잡하고 다중 모달 분포를 모델링하기 어려움. 최근에는 노이즈 제거 오토인코더 계층을 사용하는 확산 모델(diffusion models)이 이미지 합성뿐만 아니라 여러 작업에서 인상적인 성과를 보여주고 있음.
> 
- 확산 모델은 Mode Collapse나 훈련 불안정성을 보이지 않으며, 파라미터 공유를 통해 자연 이미지의 매우 복잡한 분포를 수십억 개의 매개변수를 사용하지 않고도 모델링할 수 있음.
- 그러나 이러한 모델들은 RGB 이미지의 고차원 공간에서 반복적인 함수 평가와 그래디언트 계산을 필요로 하기 때문에 여전히 계산 비용이 높음
- 이 논문은 사전 학습된 Autoencoder의 Latent Space에서 Diffusion Model을 훈련하여 이러한 문제를 해결.
    - **1단계: 오토인코더 학습**:
        - 낮은 차원의 표현 공간을 제공하여 이미지 공간과 지각적으로 동일하지만 계산적으로 더 효율적인 공간을 학습.
    - **2단계: Latent Diffusion Models (LDMs) 학습**:
        - 학습된 latent (잠재) 공간에서 DMs을 훈련함으로써, 고화질 이미지 합성 시 계산 복잡도를 줄여 효율성을 극대화.
        - 이러한 접근 방식은 공간 차원 수에 대한 스케일링 특성이 뛰어남.
- 이를 통해 모델의 복잡성을 줄이면서도 세부 사항을 보존하고, 텍스트나 경계 상자와 같은 조건 입력을 처리할 수 있는 강력한 제너레이터를 구현.
- 이 접근법은 이미지 인페인팅, 클래스 조건부 이미지 생성 및 다양한 작업에서 높은 성능을 달성하면서도 기존의 픽셀 기반 확산 모델에 비해 계산 비용을 줄일 수 있음.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/544db11f-4ff8-464f-911e-626b1c6563a9/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/7ec69133-1d0e-4055-be18-967654647e3a/Untitled.png)

# Related Work

이 논문의 관련 연구 섹션은 이미지 합성을 위한 생성 모델들을 다루고 있습니다.

1. **Generative Adversarial Networks (GANs)**:
    - 고해상도 이미지를 효율적으로 생성함
    - 최적화가 어렵고 전체 데이터 분포를 포착하는 데 어려움.
2. **Likelihood-Based Methods**:
    - Variational Auto Encoder(VAE), Flow-Based Models 등 존재
    - 이들은 고해상도 이미지 합성에서는 효율적이지만 샘플 품질이 GAN보다 낮음.
3. **Autoregressive Models (ARMs)**:
    - 강력한 밀도 추정 성능
    - 계산 요구량이 많고 순차적 샘플링 과정이 저해상도 이미지로 제한됨.
4. **Diffusion Probabilistic Models (DMs)**:
    - 최신 밀도 추정 및 샘플 품질에서 좋은 성능
    - 픽셀 공간에서 평가하고 최적화할 때 추론 속도가 느리고 훈련 비용이 높음.
5. **Two-Stage Approaches**:
    - 두 단계 접근법은 다양한 생성 방법의 장점을 결합하여 더 효율적이고 성능이 뛰어난 모델 생성.
    - VQ-VAE, VQGAN 등이 있으며, 압축된 잠재 이미지 공간을 모델링하여 고해상도 이미지 합성에 사용됨.
- 논문은 이러한 기존 방법들의 한계를 극복하기 위해 잠재 공간에서 작동하는 확산 모델(Latent Diffusion Models, LDMs)을 제안
- 이는 계산 비용을 크게 줄이면서도 높은 합성 품질을 유지할 수 있음을 강조합니다

# Method

> Diffusion Model은 통계적 특성상 정보량이 많아 고해상도 이미지 생성을 위해 많은 계산 자원과 시간이 필요함. 학습 단계에서 압축과 생성을 분리하는 방법을 제안함. 기존 Diffusion Model은 모든 계산을 고해상도 픽셀 공간에서 수행하여 비효율적인 반면, Autoencoder 모델은 이미지 공간과 인식적으로 동등한 공간을 학습하지만 계산복잡도가 크게 낮아짐.
> 

## **Perceptual Image Compression**

- 고해상도 이미지의 생성에 필요한 계산 비용을 줄이기 위해, 이미지 공간을 대신할 잠재 공간(latent space)을 학습하는 오토인코더(autoencoder)를 사용.
- 오토인코더는 이미지를 잠재 표현으로 인코딩하고, 이를 다시 원본 이미지로 디코딩하여 재구성. 이 과정에서 이미지의 고주파 세부 사항을 제거하면서도 지각적으로 중요한 정보를 보존.
- Perceptual Loss와 Patch-Based Adversarial Objective를 결합하여 로컬 리얼리즘을 보장하고 블러 현상을 방지.
- 이미지 $x \in \mathbb{R}^{H \times W \times 3}$가 RGB 공간에서 주어졌을 때
    - Encoder($\mathcal{E}$)는 $x$를 Latenr Representation $z=\mathcal{E}(x)$,  $z \in \mathbb{R}^{h \times w \times c}$ 로 인코딩
    - Decoder($\mathcal{D}$)는 Latent Representation에서 이미지 Reconstruct $\tilde{x} = \mathcal{D}(z) = \mathcal{D}(\mathcal{E}(x))$
- 인코더는 이미지의 크기를 $f = \frac{H}{h} = \frac{W}{w}$로 다운샘플링.
- 다양한 다운샘플링 계수 ( $f = 2^m$ ) (단, $m \in \mathbb{N}$)를 조사.
- Latent Space에서의 High Variance를 피하기 위해서 두 가지의 Regularization 실험
    - **KL-reg**
        - $z$를 표준 정규 분포로 조금씩 되돌아가도록 경감시키는 패널티를 부과.
        - VAE에서 사용된 방식과 유사.
    - **VQ-reg**
        - 디코더 내에 벡터 양자화 (Vector Quantization) 층을 사용하여 잠재 공간을 정규화.
        - 양자화 층을 디코더에 통합한 VQGAN([23])으로 해석 가능.
- 학습된 Latent Space가 2차원 구조로 되어 있어, 상대적으로 부드러운 압축률 (mild compression rates)을 사용해도 좋은 Reconstruction 성능을 얻을 수 있음.

## **Latent Diffusion Models (LDMs)**:

- 고차원 이미지 공간을 피하고 낮은 차원의 잠재 공간에서 확산 모델을 훈련. 이를 통해 샘플링 효율성을 크게 높이고 계산 비용을 줄일 수 있음.
- 잠재 공간의 유도 바이어스(inductive bias)를 활용하여 공간 구조를 가진 데이터에 효과적으로 적용할 수 있음.
- 제안된 LDMs는 일반적인 압축 모델로서 다양한 생성 모델을 훈련하는 데 사용될 수 있으며, 단일 이미지 CLIP 기반 합성 등의 다른 다운스트림 응용에도 활용될 수 있습니다.

**Diffusion Models:**

- 점진적으로 노이즈를 제거하여 데이터 분포 $p(x)$를 학습하도록 설계된 확률론적 모델
- 고정된 길이 $T$의 Markov Chain의 역과정을 학습하는 것에 해당
- 노이즈가 추가된 변수에서 점진적으로 노이즈를 제거하여 원래의 데이터 분포를 학습
    - Reweighted Variational Lower Bound 사용
    - 이는 노이즈 제거 스코어 매칭(denoising score-matching)과 유사함
- 이 모델은 Denoising autoencoder로 해석될 수 있음
- 각 오토인코더는 입력값 $x_t$의 노이즈 제거된 변형을 예측하도록 훈련됨
- 목표 함수 ( LDM ):
    
    $$
    L_{DM} = \mathbb{E}_{{x, \epsilon \sim \mathcal{N}(0, 1), t}} \left[ | \epsilon - \epsilon\theta(x_t, t) |_2^2 \right]
    $$
    
- $t$는 $1, \ldots, T$ 범위에서 균일하게 샘플링됨
- $x$ : 원본 데이터
- $\epsilon \sim \mathcal{N}(0, 1)$ : 평균이 0이고 분산이 1인 정규 분포에서 샘플링된 노이즈
- $x_t$: 입력 ( x )에 노이즈가 추가된 버전
- $\epsilon_\theta(x_t, t)$: 노이즈가 추가된 입력 ( x_t )와 시간 단계 ( t )가 주어졌을 때 노이즈를 제거한 예측값
- $\mathbb{E}$: 기대값(Expectation)

**Generative Modeling of Latent Representations**

- $\mathcal{E}, \mathcal{D}$로 구성된 Perceptual Compression Model을 통해 디테일이 추상화된 Low-Dimensional Latent Space에 접근할 수 있게 됨
    - 고차원 픽셀 공간에 비해 데이터의 중요하고 의미 있는 비트에 집중할 수 있음
    - 계산적으로 훨씬 더 효율적인 저차원 공간에서 훈련할 수 있기 때문에 가능성 기반 생성 모델에 더 적합.
- 기존의 연구들은 고도로 압축된 이산적 잠재 공간에서 autoregressive, attention 기반 transformer 모델들을 사용한 반면, 본 논문의 모델은 이미지 별 Inductivd Bias를 활용할 수 있음
- 여기에는 주로 2D 컨볼루션 레이어에서 기본 UNet을 구축하는 기능과 가중치 바운드를 사용하여 지각적으로 가장 관련성이 높은 비트에 목표를 더욱 집중하는 기능이 포함됨.
    - 이미지에 특화된 귀납적 편향을 활용하여 성능을 높임.
    - 2D 합성곱 계층(convolutional layers)을 주로 사용하여 UNet을 구축.
    - Reweighted bound 를 사용하여 중요하고 의미 있는 정보에 초점을 맞춤.

$$
L_{\text{LDM}} := \mathbb{E}_{{\mathcal{E}(x), \varepsilon \sim \mathcal{N}(0, 1), t}} [| \varepsilon - \varepsilon\theta (z_t, t) |^2_2]
$$

- $\mathbb{E}$ : 기대 값 (기대치).
- $\mathcal{E}(x)$ : 입력 이미지 $x$의 잠재 표현.
- $\varepsilon \sim \mathcal{N}(0, 1)$ : 정규 분포를 따르는 잡음.
- $t$ : 시간 단계.
- $| \cdot |^2_2$ : L2 노름 (유클리드 거리를 제곱).
- Neural Backbone $\theta(\cdot, t)$는 시간 조건부 인 UNet 구조로 구현됨
- Forward Process가 고정되어 있기 때문에 E에서 $z_t$를 효율적으로 얻을 수 있음
- $p(z)$의 샘플은 $\mathcal{D}$를 한 번 통과하여 이미지 공간으로 디코딩 할 수 있음

- 기존과 달리 autoregressive 모델 대신 잠재 공간을 활용하여 더 나은 성능 제공.
- 이미지에 특화된 구조적 편향을 활용함으로써 합성의 질을 높임.
- 효율적인 잠재 표현을 통해 훈련 시 높은 계산 효율성을 제공.

`여기까지`

## **Conditioning Mechanisms**:

- 제안된 모델은 다양한 조건 입력(예: 텍스트, 경계 상자)을 처리할 수 있는 크로스 어텐션(cross-attention) 기반의 조건 메커니즘을 사용합니다.
- 이러한 메커니즘은 텍스트-이미지 합성, 레이아웃-이미지 합성 등 다양한 작업에 대해 강력한 성능을 제공합니다.

이 접근 방식을 통해 기존의 픽셀 기반 확산 모델보다 효율적이고 성능이 뛰어난 모델을 개발할 수 있음을 강조합니다.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/182009c0-dc8a-4c72-8583-e0ae25583379/Untitled.png)

### 4. Experiments

이 섹션에서는 제안된 잠재 확산 모델(LDMs)의 다양한 이미지 합성 작업에 대한 성능을 실험을 통해 분석합니다. 주요 실험은 다음과 같습니다:

### 4.1 Perceptual Compression Tradeoffs

LDMs의 다운샘플링 요인(f ∈ {1, 2, 4, 8, 16, 32})에 따른 성능을 비교합니다. 실험 결과, 너무 작은 다운샘플링 요인은 훈련 속도를 느리게 하고, 너무 큰 요인은 정보 손실을 초래하여 품질이 낮아집니다. LDM-4와 LDM-8은 효율성과 품질 사이의 균형을 잘 맞추며, 픽셀 기반 확산 모델(LDM-1)에 비해 높은 성능을 보입니다.

### 4.2 Image Generation with Latent Diffusion

LDM을 사용한 256x256 이미지 생성 실험에서는 CelebA-HQ, ImageNet 등의 데이터셋에서 샘플링 속도와 FID 점수를 비교합니다. LDM-4와 LDM-8은 복잡한 데이터셋에서도 높은 품질의 샘플을 빠르게 생성할 수 있음을 보여줍니다.

### 4.3 Conditional Latent Diffusion

크로스 어텐션을 활용한 조건부 이미지 생성 실험에서는 텍스트-이미지 합성, 레이아웃-이미지 합성 등 다양한 조건 입력을 처리하는 모델을 평가합니다. Transformer 인코더를 사용하여 텍스트 조건을 처리하고, 256x256 이상의 해상도에서도 우수한 성능을 보입니다.

### 4.4 Super-Resolution with Latent Diffusion

ImageNet 데이터셋을 사용한 초해상도 생성 실험에서는 LDM이 다양한 저해상도 이미지를 고해상도로 변환하는 성능을 평가합니다. LDM-4는 기존 방법들에 비해 우수한 FID와 PSNR 점수를 기록합니다.

### 4.5 Inpainting with Latent Diffusion

마스킹된 이미지 영역을 채우는 인페인팅 실험에서는 Places 데이터셋을 사용하여 LDM의 성능을 평가합니다. LDM-4는 다른 최신 방법들에 비해 더 현실적인 이미지를 생성하며, 사용자 선호도 조사에서도 높은 점수를 받았습니다.

이 실험들을 통해 제안된 LDM이 기존의 픽셀 기반 확산 모델보다 계산 비용을 줄이면서도 높은 성능을 유지할 수 있음을 입증합니다.

【23:2†2112.10752v2.pdf】

다음은 업로드된 문서의 "Limitations & Societal Impact" 섹션 요약입니다:

# Limitations & Societal Impact

## Limitations

잠재 확산 모델(LDMs)은 픽셀 기반 접근 방식에 비해 계산 요구 사항을 크게 줄이지만, 여전히 샘플링 과정은 GANs보다 느립니다. 또한, 높은 정밀도가 필요한 경우에는 LDMs의 사용이 문제가 될 수 있습니다. f = 4 오토인코딩 모델에서 이미지 품질 손실은 매우 작지만, 재구성 능력이 픽셀 공간에서의 세밀한 정확성을 요구하는 작업에서는 병목이 될 수 있습니다. 특히, 초해상도 모델은 이러한 제한이 이미 존재한다고 가정합니다.

## Societal Impact

이미지와 같은 미디어를 위한 생성 모델은 양날의 검입니다. 한편으로는 창의적 응용 프로그램을 가능하게 하고, 특히 우리의 접근 방식처럼 훈련과 추론 비용을 줄이는 방법은 이 기술에 대한 접근성을 높이고 탐구를 민주화할 수 있습니다. 그러나 다른 한편으로는 조작된 데이터를 생성하고 유포하거나 허위 정보를 퍼뜨리는 것이 더 쉬워집니다. 특히 여성들이 비율적으로 더 많이 영향을 받는 "딥페이크" 문제와 같은 이미지 조작이 일반적인 문제입니다.

생성 모델은 훈련 데이터도 드러낼 수 있는데, 이는 데이터가 민감하거나 개인 정보가 포함된 경우 특히 문제가 됩니다. 이러한 데이터가 명시적 동의 없이 수집된 경우 더욱 그렇습니다. 그러나 이러한 문제가 이미지 DMs에도 동일하게 적용되는지는 아직 완전히 이해되지 않았습니다.

마지막으로, 딥 러닝 모듈은 데이터에 이미 존재하는 편향을 재현하거나 악화시키는 경향이 있습니다. 확산 모델은 GAN 기반 접근 방식보다 데이터 분포를 더 잘 커버하지만, 적대적 훈련과 우도 기반 목표를 결합한 우리의 두 단계 접근 방식이 데이터를 잘못 표현하는 정도는 중요한 연구 질문으로 남아 있습니다.

윤리적 고려 사항에 대한 일반적이고 상세한 논의는 다른 문헌을 참조할 수 있습니다.

【27:0†2112.10752v2.pdf】

다음은 업로드된 문서의 "Conclusion" 섹션 요약입니다:

## Conclusion

이 논문에서는 노이즈 제거 확산 모델(denoising diffusion models)의 훈련 및 샘플링 효율성을 크게 향상시키는 잠재 확산 모델(Latent Diffusion Models, LDMs)을 제안했습니다. 제안된 LDMs는 모델의 품질을 저하시키지 않으면서도 효율성을 높이는 간단하고 효과적인 방법을 제공합니다. 또한 크로스 어텐션 조건 메커니즘(cross-attention conditioning mechanism)을 기반으로, 다양한 조건부 이미지 합성 작업에서 최첨단 성능을 달성했습니다. 제안된 접근 방식은 특정 작업에 특화된 아키텍처 없이도 뛰어난 성능을 보여주었습니다.

이 연구는 독일 연방 경제 에너지부의 'KI-Absicherung - Safe AI for automated driving' 프로젝트와 독일 연구재단(DFG)의 프로젝트 421703927의 지원을 받아 수행되었습니다.

【27:0†2112.10752v2.pdf】