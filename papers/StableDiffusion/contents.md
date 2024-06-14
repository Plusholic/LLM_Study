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
- 

## **Conditioning Mechanisms**

- 디퓨전 모델 (Diffusion Models, DMs)는 $p(z|y)$ 형태의 조건부 분포(conditional distribution)를 모델링할 수 있음. 이는 Conditional Denoising Autoencoder $θ (z_t, t, y)$를 통해 구현.
- Diffusion Model의 기본 UNet Backbone Cross Attention Mechanism을 추가하여 다양한 입력 모달리티를 학습.
    - 다양한 양식(언어 프롬프트 등)을 전처리 하기 위해 $y$를 중간 표현 Projection하는 Domain Specific Encoder $τ_θ$를 도입
    - $τ_θ(y) ∈ R^{M ×d_τ}$ UNet의 중간 레이어에 매핑 되며, 다양한 모달리티 간의 Attention을 계산.

$$
Attention(Q, K, V) = softmax({QK^T\over \sqrt{d}})
$$

$$
Q = W^{(i)}_Q · ϕ_i(z_t) \ \ \ K = W^{(i)}_K · τ_θ (y) \ \ \ V = W^{(i)}_V · τ_θ (y)
$$

- **$ϕ_i(z_t)$ :** UNet이 구현하는 $ϕ_θ$의 중간 표현으로, 다차원 벡터를 평탄화한 상태. $ϕ_θ$모델에서 시간에 따라 변화함.
- W는 각각 학습 가능한 투영 행렬로, 다차원 입력을 적절한 차원으로 변환.
    - $W^{(i)}_V ∈ \mathbb{R}^{d×d^{\epsilon}_i}$
    - $W^{(i)}_Q ∈ \mathbb{R}^{d×d_τ}$
    - $W^{(i)}_K ∈ \mathbb{R}^{d×d_τ}$
- **목적 함수(LLDM):**
    
    $$
    
    L_{LDM} := \mathbb{E}_{\mathcal{E}(x),y,\epsilon \sim \mathcal{N}(0,1),t} [ | \epsilon - \epsilon{\theta}(z_t, t, \tau_{\theta}(y)) |^2_2 ]
    
    $$
    
- **$\mathbb{E}_{\mathcal{E}(x),y,\epsilon \sim \mathcal{N}(0,1),t}$:** 입력 이미지($x$)를 압축한 후($t$ 시간 동안), 텍스트 프롬프트($y$)를 사용하여 모델이 $\epsilon$을 예측.
- **$| \epsilon - \epsilon_{\theta}(z_t, t, \tau_{\theta}(y)) |^2_2$ :** 실제 $\epsilon$과 모델이 예측한 $\epsilon$ 사이의 L2 노름 거리를 최소화.
- **$τθ$:** 도메인별 전문가로 파라미터화 가능하여 다양한 입력 모달리티(예: 텍스트 프롬프트 처리)를 수행.
- 텍스트 프롬프트($y$)가 있을 때, 이를 중간 표현으로 변환 후 UNet에 적용하여 결과물 생성.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/182009c0-dc8a-4c72-8583-e0ae25583379/Untitled.png)

### 4. Experiments

> LDMs(잠재 확산 모델)은 다양한 이미지 모달리티에 대해 유연하고 계산적으로 효율적인 확산 기반 이미지 생성을 가능하게 함. LDM들은 훈련과 추론 측면에서 픽셀 기반 확산 모델(DMs)보다 더 나은 성과를 보였음. VQ(벡터 양자화)로 정규화된 잠재 공간에서 훈련된 LDM들은 샘플 품질이 더 우수한 경우가 많지만, VQ 정규화된 첫 단계 모델의 재구축 능력은 연속형 잠재 공간을 사용하는 모델에 비해 조금 부족함.
> 

### 4.1 Perceptual Compression Tradeoffs

- 다양한 다운샘플링 팩터 $f \in {1, 2, 4, 8, 16, 32}$ 사용 시, LDM(Latent Diffusion Models)의 성능 분석.
    - 여기서 $\text{LDM-}f$는 다운샘플링 팩터 $f$를 가리키며, $\text{LDM-1}$은 픽셀 기반의 DMs을 의미함.
- 모든 모델은 동일한 컴퓨팅 자원(단일 NVIDIA A100)과 동일한 훈련 단계 및 파라미터 수로 훈련됨.
- $\text{LDM-1}, \text{LDM-2}$
    - 느린 훈련 진행.
- $\text{LDM-32}$
    - 정보 손실로 인해 훈련 초기에는 품질이 정체됨.
- $\text{LDM-4}, \text{LDM-8}, \text{LDM-16}$
    - 효율성과 세부 표현 간의 균형이 좋아, 최적의 성능을 보임.
- **샘플 퀄리티와 훈련 진행**(2M 단계 동안 ImageNet 데이터셋에서):
    - 픽셀 기반의 $\text{LDM-1}$과 $\text{LDM-8}$ 사이에 FID(Frechet Inception Distance) 점수가 38 차이남.
    - 작은 다운샘플링: 대부분의 퍼셉추얼 압축을 디퓨전 모델에 맡김.
    - 큰 다운샘플링: 첫 번째 단계의 압축이 너무 강해 정보 손실이 발생.
- **시간과 성능 비교**(CelebA-HQ와 ImageNet 데이터셋에서):
    - $\text{LDM-4}$와 $\text{LDM-8}$이 다양한 샘플링 단계 수에서 월등한 성능을 보이며, 특히 픽셀 기반의 $\text{LDM-1}$과 비교 시 FID 점수가 훨씬 낮아짐.
    - $\text{LDM-4}$와 $\text{LDM-8}$이 최고의 품질을 보여줌.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/e6a42b47-c828-4bf7-bd6f-25db7b9cbf09/Untitled.png)

### 4.2 Image Generation with Latent Diffusion

- LDM을 사용한 256x256 이미지 생성 실험에서는 CelebA-HQ, ImageNet 등의 데이터셋에서 샘플링 속도와 FID 점수를 비교.
- LDM-4와 LDM-8은 복잡한 데이터셋에서도 높은 품질의 샘플을 빠르게 생성할 수 있음을 보여줌.

### 4.3 Conditional Latent Diffusion

- **Cross Attention**을 도입하여 LDM(Latent Diffusion Model)을 다양한 조건부 양식에 맞게 확장.
- 텍스트-이미지 모델링을 위해 14.5억 파라미터의 **KL 정규화된** LDM을 LAION-400M 데이터베이스의 언어 프롬프트에 맞춰 학습.
    - **BERT 토크나이저**를 사용.
    - 잠재 코드 유도를 위한 **트랜스포머**를 도입하고, 이를 **UNet**에 Multi Head Cross Attention으로 매핑.
    - 복잡한 사용자 정의 텍스트 프롬프트에 일반화가 가능한 강력한 모델 생성.
- **질적 분석 및 평가**:
    - 이전 연구를 따라 **MS-COCO 검증 세트**에서 텍스트-이미지 생성 평가.
    - 모델이 강력한 AR(자기회귀) 및 GAN 기반 방법들을 능가하는 성능을 보임.
    - **Classifier-Free Diffusion Guidance**를 적용하여 샘플 품질 향상.
    - **LDM-KL-8-G** 모델이 최근 최신 AR 및 확산 모델과 동등한 성능을 보이지만 파라미터 수는 크게 감소.
- **유연성 분석**:
    - **Semantic Layouts**에 기반한 이미지 생성 모델도 학습.
    - OpenImages에서 학습한 후 COCO에서 미세 조정.
    - 질적 평가 및 구현 세부 정보는 섹션 D.3 참조.
- **최신 성능 모델 평가**:
    - 클래스 조건부 ImageNet 모델 평가.
    - 최신 확산 모델 ADM보다 우수한 성능을 보이면서도 계산 요구사항과 파라미터 수는 크게 감소.

- **컨볼루션 샘플링 설명**
    - LDMs은 입력 정보에 공간적으로 정렬된 조건 정보를 연결하여 효율적인 범용 이미지-이미지 변환 모델로 사용할 수 있Dma.
    - 이 방식을 통해 다양한 응용 분야에서 활용할 수 있습니다. 예를 들어:
        - **Semantic synthesis**: 풍경 이미지와 시멘틱 맵을 활용하여 트레이닝.
        - **Super-resolution**: 저해상도 이미지를 고해상도로 변환.
        - **Inpainting**: 손상되거나 불필요한 부분을 새로운 콘텐츠로 채움.
    - 시멘틱 신세시스의 경우, 풍경 이미지와 해당 시멘틱 맵을 사용하여 트레이닝하며, 입력 해상도는 256²입니다(384²에서 크롭된 부분을 사용).
- 모델은 큰 해상도로 일반화될 수 있음. 예를 들어, 256² 해상도에서 훈련된 모델은 512×1024 해상도로도 이미지 생성이 가능함.

### 4.4 Super-Resolution with Latent Diffusion

ImageNet 데이터셋을 사용한 초해상도 생성 실험에서는 LDM이 다양한 저해상도 이미지를 고해상도로 변환하는 성능을 평가합니다. LDM-4는 기존 방법들에 비해 우수한 FID와 PSNR 점수를 기록합니다.

- **이미지 열화와 다운샘플링**:
    - SR3의 데이터 처리 파이프라인을 따라 이미지를 4배 다운샘플링하는 비큐빅 보간을 사용했습니다.
    - OpenImages 데이터셋에서 사전 훈련된 $f = 4$ 오토인코딩 모델을 사용했습니다.
    - 저해상도 조건 $y$와 UNet의 입력을 연결하여 사용했습니다. 이때 $\tau_\theta$는 신원(identity)으로 간주했습니다.
- **성능 비교**:
    - 정성적 및 정량적 결과에서 LDM-SR이 FID 점수에서 SR3를 능가했습니다.
    - 하지만 SR3는 IS 점수에서 더 나은 결과를 보였습니다.
    - 간단한 이미지 회귀 모델은 PSNR 및 SSIM 점수에서 최고의 성능을 보였지만, 이는 인간의 인식과 잘 맞지 않고 블러리한 성향을 띠었습니다.
- **사용자 연구**:
    - SR3의 방식을 따랐습니다. 인간 피험자들에게 저해상도 이미지를 두 고해상도 이미지 사이에 보여주고 선호도를 물었습니다.
    - 표 4의 결과는 LDM-SR의 좋은 성능을 확인해 줍니다.
- **후처리 기법**:
    - PSNR 및 SSIM 점수를 향상시키기 위해 사후 가이딩 메커니즘을 사용했습니다.
    - 이미지 기반의 가이더를 사후 손실 함수(perceptual loss)를 통해 구현했습니다.

### 4.5 Inpainting with Latent Diffusion

- 이미지를 채우는 작업을 의미하며, 손상된 이미지 부분을 새로운 콘텐츠로 채우거나, 원하지 않는 콘텐츠를 대체하는데 사용됨.
- LaMa[88] 모델의 프로토콜을 따릅니다. 이 모델은 Fast Fourier Convolutions[8]을 사용한 특수한 아키텍처를 도입한 최근의 Inpainting 모델입니다.
- LDM-1 (픽셀 기반의 조건부 DM), LDM-4 (KL 및 VQ 정규화를 사용), VQ-LDM-4 (첫 번째 단계에서 주의(attention)를 사용하지 않은 경우)의 효율성을 비교
- 비교를 위해 모든 모델의 파라미터 수를 동일하게 고정
- LDM-1과 LDM-4의 훈련 및 샘플링 처리량: 해상도 256^2 및 512^2에서 훈련 및 샘플링 처리량, 에포크당 총 훈련 시간 및 6 에포크 후의 Validation Split에 대한 FID 점수(FID: Fréchet Inception Distance)
- 픽셀 기반 모델과 잠복 공간(Latent Space) 기반 모델 간에 속도 향상 2.7배 이상, FID 점수 1.6배 향상
- **다른 inpainting 접근법과의 비교**
    - 주의(attention)를 사용하는 모델이 LaMa[88] 대비 FID 점수가 향상됨
    - LPIPS 지수는 LaMa[88]보다 약간 높음: LaMa는 단일 결과만 생성, LDM은 다양한 결과 생성
    - 사용자 연구에서는 참가자들이 LDM의 결과를 LaMa의 결과보다 선호

# Limitations & Societal Impact

## Limitations

- **제한 사항:**
    - LDMs(Latent Diffusion Models)은 픽셀 기반 접근 방식에 비해 계산 요구 사항을 크게 줄이지만, 샘플링 과정이 여전히 GANs(Generative Adversarial Networks)보다 느리다고 말합니다.
    - 또한 높은 정밀도가 요구되는 경우 LDMs의 사용이 문제가 될 수 있습니다.
    - 비록 $f = 4$ 오토인코딩 모델에서 이미지 품질 손실이 매우 적다고 하더라도, 픽셀 단위의 정확도가 필요한 작업에서는 재구성 능력이 병목이 될 수 있습니다.
    - 초고해상도 모델은 이미 이와 같은 한계가 일부 존재하는 것으로 가정합니다.
- **사회적 영향:**
    - 생성 모델은 이중적 측면을 가집니다.
        - 한편으로는 다양한 창의적 응용을 가능하게 하고 기술에 접근하는 비용을 줄임으로 민주화될 수 있습니다.
        - 다른 한편으로는 조작된 데이터를 만들어 유포하거나 허위 정보를 퍼뜨리는 등의 부작용을 초래할 수 있습니다.
    - 특히 이미지의 고의적인 조작("딥페이크")은 여성에게 특히 많이 발생하는 문제입니다.
    - 생성 모델은 훈련 데이터도 드러낼 수 있습니다. 이는 데이터에 민감하거나 개인적인 정보가 포함되어 있을 때 큰 문제가 될 수 있습니다.
    - 또한, 딥러닝 모델들은 기존 데이터에 존재하는 편향을 재현하거나 악화시키는 경향이 있습니다.
    - 확산 모델은 GAN 기반 접근법보다 데이터 분포를 더 잘 포괄하지만, 우리의 이중 단계 접근법(GAN과 가능도 기반 목표를 결합한 것)이 데이터를 잘못 표현할 가능성이 있는지 연구가 필요합니다.

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