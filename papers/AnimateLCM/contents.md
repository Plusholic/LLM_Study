[원문링크](https://arxiv.org/abs/2402.00769)

# 3 LINE SUMMARY

AnimateLCM은 고화질 비디오 생성을 빠르게 하기 위해 설계된 모델입니다. 이 모델은 이미지 생성 및 움직임 생성을 분리하여 학습 효율성을 높이고 생성 품질을 향상시킵니다. 또한, 다양한 기능을 수행하는 어댑터와 통합할 수 있으며, 실험 결과 우수한 성능을 입증했.

# **Abstract**

> 비디오 확산 모델은 일관성과 높은 품질의 비디오를 생성할 수 있는 능력으로 인해 점점 더 많은 주목을 받고 있지만 반복적인 노이즈 제거 과정이 필요하여 계산 비용이 높고 시간이 많이 소요되는 단점이 있음. 일관성 모델(Consistency Model, CM)은 사전 학습된 이미지 확산 모델을 최소 단계로 샘플링을 가속화할 수 있도록 증류하는 기법으로, 이러한 문제를 해결하기 위해 개발되었음. 이를 통해 고품질의 이미지 생성을 가능하게 하며, 확장된 버전인 잠재 일관성 모델(Latent Consistency Model, LCM)은 조건부 이미지 생성에서도 성공을 거두었음.
> 
- 최소 단계로 고화질 비디오 생성을 하기 위해 Image Generation과 Motion Generation의 Consistency Learning을 분리하는 전략으로 학습 효율성을 높이고 시각적 품질을 향상시킴.
- Stable Diffusion Community에서 플러그 앤 플레이 어댑터를 조합하여 다양한 기능(예: 제어 가능한 생성을 위한 콘트롤넷)을 구현할 수 있음
- 샘플링 속도에 영향을 미치지 않고 어댑터를 학습하거나 기존 어댑터를 효율적으로 사용할 수 있음.
- Image-Conditioned Video Generation 및 Layout-Conditioned Video Generation에서 제안된 전략의 유효성을 검증하였으며, 모든 실험에서 우수한 결과를 얻었음

# **Introduction**

> 확산 모델의 고품질 생성은 반복적인 노이즈 제거 과정을 통해 고차원 가우시안 노이즈를 실제 데이터로 점진적으로 변환하는 과정에 의존함. 이미지 생성의 대표적인 모델 중 하나는 Stable Diffusion (SD)으로, 이는 Variational Autoencoder(VAE)를 사용하여 실제 이미지와 다운샘플링된 잠재 특징 간의 매핑을 구축하여 생성 비용을 줄이고, Cross-Attention 메커니즘을 통해 Text-Conditioned image generation을 실현함. Stable Diffusion을 기반으로 ControlNet과 같은 여러 플러그 앤 플레이 어댑터가 개발되어 결합함으로써 더욱 혁신적인 기능을 수행할 수 있음.
> 
- 그러나 반복적 샘플링의 특성으로 인해 확산 모델의 생성 속도는 느리며 계산 비용이 크며, 다른 생성 모델(예: GAN)보다 훨씬 느림.
- 최근 일관성 모델(Consistency models, CM)이 생성 과정을 가속화하기 위한 유망한 대안으로 제안되었음.
- 사전 학습된 확산 모델이 유도한 PF-ODE(Probability Flow Ordinary Differential Equation) 궤적에서 Self-Consistency을 유지하는 일관성 매핑을 학습함으로써, CM은 매우 적은 단계로 고품질 이미지를 생성할 수 있어 계산 집약적인 반복의 필요성을 제거합니다.
    - Self-Consistency는 모델이 예측하는 값들이 서로 일관되도록 하는 속성
- Latent Consistency Model(LCM)은 SD를 기반으로 구축되었으며, Web-UI에 기존 어댑터와 통합되어 다양한 기능(예: 실시간 이미지-이미지 변환)을 실현할 수 있음.
- 반면, 비디오 확산 모델은 많은 진전을 이루었지만, 비디오 생성의 높은 계산 비용으로 인해 비디오 샘플링의 가속화는 여전히 탐구되지 않은 시급한 문제로 남아 있음.

- 이 논문에서는 최소한의 단계로 고화질 비디오 생성을 위한 AnimateLCM을 제안함.
    - LCM을 따르며, Reverse Diffusion Process를 CFG(Classifier-Free Guidance)가 없는 `Probability Flow ODE(PF-ODE)`를 푸는 것으로 간주하고, 잠재 공간에서 이러한 `ODE의 해를 직접 예측하도록 모델을 훈련`함.
    - 낮은 품질의 문제와 높은 훈련 자원이 필요한 원시 비디오 데이터에서 직접 일관성 학습을 수행하는 대신, Image Generation과 Motion Generation을 분리하여 Consistency Distillation를 수행하는 전략을 제안함.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/5f45148f-c890-4da1-929d-e7db99eb7f0b/Untitled.png)

- 먼저 Image Diffusion Model을 Image Consistency Model로 적응시키기 위해 `Consistency Distillation를 수행`함.
- 이후 두 모델 모두에 3D Infllation을 적용하여 3D 비디오 특징을 수용(그림 1)
    - 비디오 데이터에서 Consistency Distillation를 수행하여 최종 Video Consistency Model을 얻게 됨.
    - 인플레이션(주로 2D 모델을 3D 모델로 확장하는 방법) 과정에서 발생할 수 있는 잠재적인 특징 손상을 완화하기 위해 특별히 설계된 초기화 전략을 추가로 제안.
    - 실험을 통해 이 과정이 훈련 효율성을 높일 뿐만 아니라 최종 생성 품질을 향상시킨다는 것을 입증.
- AnimateLCM을 SD에 구축하여, 훈련된 Vidoe Consistency 모델의 공간 가중치를 공개된 개인화된 Image Diffusion 가중치로 대체하여 혁신적인 생성 결과를 달성할 수 있음.
    - 커뮤니티의 대부분의 어댑터가 훈련된 비디오 일관성 모델과 직접 통합될 수 있지만, 세부 사항의 제어를 잃거나 결과에 깜박임이 발생할 수 있음.
- 커뮤니티의 기존 어댑터에 더 적합하거나 비디오 일관성 모델로 특정 어댑터를 처음부터 훈련하기 위해 특정 교사 모델 없이 어댑터를 "가속화"할 수 있는 효과적인 전략을 제안.
    - 전략의 효과를 보여주기 위해, 이미지 인코더를 비디오 일관성 모델과 함께 처음부터 훈련하여 최소한의 단계로 고화질 이미지-비디오 생성을 추가로 달성.
    - 기존의 레이아웃 조건부 어댑터를 바닐라 이미지 확산 모델로 사전 훈련된 어댑터에 맞춰 적응하여 더 나은 호환성과 더 나은 제어 가능한 비디오 생성을 최소한의 단계로 실현

# **Related Works**

## **Diffusion Models**

> 확산 모델은 스코어 기반 생성 모델(score-based generative models)로도 알려져 있으며, 이미지 생성에서 큰 성공을 거두었음
> 
- 이 모델들은 노이즈로 오염된 데이터를 점진적으로 노이즈를 제거하는 방향으로 샘플링 과정을 통해 높은 품질의 이미지를 생성.
- 현재 성공적인 일반화된 비디오 확산 모델들은 대부분 사전 학습된 이미지 확산 모델을 사용하여 훈련되며, 이를 기반으로 시간적 레이어를 추가하여 비디오 생성을 수행함.
- 이 모델들은 Image-Video Joint Tuning 또는 Spatial Weights를 고정하는 방식으로 훈련됨.

# **Sampling Acceleration**

> 확산 모델에서 느린 생성 속도를 해결하기 위해 여러 방법이 제안됨.
> 
- 초기 연구들은 개선된 ODE Solver를 통해 샘플링 속도를 향상시키는 데 중점.
- 최근에는 Distillation 기반 가속화 방법이 원래 확산 가중치를 조정하거나 아키텍처를 개선하여 더 빠른 가속 속도를 보여주고 있음.
- 일관성 모델(Consistency Model)은 새로운 버전의 모델로, Self-Consistency 속성을 강화하여 훈련됨.
- 잠재 일관성 모델(Latent Consistency Model)은 이러한 아이디어를 안정적 확산 모델에 적용하여 조건부 이미지 생성에서 높은 성능을 보여줌.

# **Preliminaries**

> 제안된 모델은 Stable Diffusion을 기반으로 하며, 이는 DDPM(Denoising Diffusion Probabilistic Models)의 확장 버전
> 

## **1. Diffusion Models**

- Diffusion Model에서는 훈련 데이터 $x_0 \sim p_{data}(x)$ 는 점진적으로 노이즈를 추가하는 방식으로 변형됨. 이 과정은 다음과 같은 `Discrete Markov Chain으로 표현`됨.
    
    $$
    p(x_i | x_{i-1}) = \mathcal{N}(x_i;\sqrt{1-\beta_i}x_{i-1},\beta_iI)
    $$
    
- 여기서, $\beta$는 각 단계에서의 노이즈 스케줄링 파라미터. 정규분포에서 $\sqrt{1-\beta}x_{i-1}$를 평균으로, $\beta_iI$를 표준편차로 하는 노이즈.
- 각 타임스텝 $t$에서의 노이즈 데이터 분포는 다음과 같이 표현됨.

$$
p_i(x_i|x_0) = \mathcal{N}(x_i ; \sqrt{\alpha_i}x_0, (1-\alpha_i)I) \qquad where \ \ \alpha_i := \Pi_{j=1}^{i}(1-\beta_i) \qquad N \rightarrow \infty
$$

- Discrete Markov Chain에서는 각 타임스텝마다 노이즈가 추가됨. 타임스텝이 무한대로 증가하면 연속적인 과정으로 간주될 수 있음. 확률 미분방정식은 연속적인 시간 동안 확률적인 변화를 모델링 함, 무한히 진행될 때, 이는 다음과 같은 `SDE(Stochastic Differential Equation)로 수렴`
- dx는 Deterministic drift term 과 Stochastic diffusion term의 합으로 정의됨
- dw는 Wiener process 또는 Brownian motion을 의미
    - 디퓨전 모델에서 x의 변화는 Deterministic 한 요소가 포함됨. 평균적으로 x가 어떻게 변화하는지를 나타내는 항
    - 여기서는 $-{1\over2}x\beta$로 설정
    - Stochastic Diffusion Term은 노이즈가 추가되는 부분이고, Wiener Process로 표현됨. 이는 시간이 지남에 따라 누적되는 무작위 변화를 모델링하며, $g(t)$는 노이즈의 강도를 조절하는 역할
    - 여기서는 $\sqrt{\beta(t)}$로 설정됨
    - [https://ko.wikipedia.org/wiki/포커르-플랑크_방정식](https://ko.wikipedia.org/wiki/%ED%8F%AC%EC%BB%A4%EB%A5%B4-%ED%94%8C%EB%9E%91%ED%81%AC_%EB%B0%A9%EC%A0%95%EC%8B%9D)

$$
dx = f(x, t)dt + g(t)dw \qquad where \ \ f(x, t)dt = -{1\over2}x\beta(t) \ \ and \ \ g(t) = \sqrt{\beta(t)}
$$

- 이 SDE의 역방향 시간 ODE(Ordinary Differential Equation)는 다음과 같이 표현됨
    
    $$
    dx = [f(x, t) - g^2(t)\nabla_x \log p_t(x)]dt
    $$
    
- DDPM에서 noise prediction neural network $\epsilon_{\theta}(x_t, t)$는 현재 데이터 포인트에서 노이즈를 제거하도록 훈련되며, Score Function의 반대 방향을 모방
- 원래 SDE로부터 확률밀도함수 $p_t(x)$의 변화를 설명하는 Fokker-Planck 방정식을 도출
- $\nabla_x \log p_t(x)$는 확률 밀도 함수 $p_t(x)$의 로그의 그라디언트를 의미하며, 이는 확률 흐름의 방향을 나타냅니다.
    
    $$
    \nabla_{x_t\log p_t(x_t)} \approx - {1\over\sqrt{1-\alpha_t}}\epsilon_{\theta}(x_t, t)
    $$
    

## **2. Consistency Models**

- Consistency Model은 위에서 논의된 PF ODE의 솔루션 궤적 $\{x_t\}_{t\in[\epsilon, T]}$을 따라 모든 데이터 포인트가 동일한 솔루션을 직접 예측하도록 함. 이를 위해서는 다음과 같은 자체 일관성 속성이 강화됨.

$$
f(x_t, t)=f(x_{t'}, t') \ \ for \ all \ t, t' \in [\epsilon, T]
$$

- 이를 달성하기 위해 Consistency Model은 Skip Connection을 사용하여 다음과 같이 공식화됨.

$$
f_{\theta}(x_t, t) = c_{skip}(t)x_t + c_{out}(t)F_{\theta}(x, t)
$$

- 여기서 $where \ \ c_{skip}(t), c_{out}(t)$ 은 미분 가능한 함수로, $c_{skip}(\epsilon)=1, c_{out}(\epsilon)=0$  을 만족.
- 일관성 속성을 강화하기 위해 Target Model $\theta^-$이 Exponential Moving Average(EMA)로 업데이트되며, 이는 다음과 같은 Consistency Distillation Loss를 최소화함
    
    $$
    \mathcal{L}(\theta, \theta^-;\Phi) = \mathbb{E}_{x, t}[d(f_{\theta}(x_{t_{n+1}},t_{n+1}), f_{\theta^-}(\hat{x}_t^{\phi}, t_n))]
    $$
    
- 여기서 $d(\cdot \ , \cdot)$는 거리 측정 함수이며, $\hat{x}_{t_n}^{\phi}$는 사전 학습된 확산 모델을 사용하여 ODE 솔버로부터 얻어진 값.

# **Methodology**

> AnimateLCM은 안정적 확산 기반 비디오 모델을 일관성 속성을 따르도록 조정하여 최소한의 단계로 고화질 비디오 생성을 목표로 함. 확산 모델을 일관성 모델로 변환하고, 분리된 일관성 학습, 교사 모델 없는 어댑터 적응에 대해 다룸
> 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/706f9d65-533e-4ec3-a86c-0f54e703b806/Untitled.png)

## **From DMs to CMs (확산 모델에서 일관성 모델로)**

**From $\epsilon$-prediction to $x_0$-prediction ($\epsilon$-예측에서 $x_0$-예측으로)**

- Stable Diffusion Model은 일반적으로 주어진 샘플에 추가된 노이즈를 예측함.
- Consistency Model은 PF-ODE 궤적의 솔루션 **$x_0$**을 직접 예측하는 것을 목표로 함. 이를 위해 다음과 같이 모델을 파라미터화.

$$
f_{\theta}(x_t, c, t) = c_{skip}(t)x_t + c_{out}(t) (x_t - \sqrt{1-\alpha_t}\epsilon_{\theta}(x_t, c, t)) / \sqrt{\alpha_t}
$$

- $c$는 Text Condition Embedding.

**Classifier-free guidance augmented ODE solver (CFG 증강 ODE 솔버)**

- 고품질 생성을 위해 CFG(Classifier-Free Guidance)가 필수적입니다. 이는 다음과 같이 표현됨

$$
\epsilon_{\phi}(x_t, c, t, w) = (1+w) \epsilon_{\theta}(x_t, c, t) - w \epsilon_{\phi}(x_t, \empty, t)
$$

- 이에 따라 PF ODE 솔버는 다음과 같이 나타낼 수 있습니다.

$$
\Phi_w(x_t, c, t, t_{n+1};\phi) = (1+w)\Phi(x_t, c, t, t_{n+1}l\phi) - w\Phi(x_t, \empty, t, t_{n+1};\phi)
$$

**Sample sparse timesteps from dense timesteps**

- 훈련 효율성을 위해 전체 1000개의 타임스텝 중 50개의 타임스텝을 균등하게 샘플링.

## **Decoupled Consistency Learning (분리된 일관성 학습)**

**Decoupling image generation and motion priors (이미지 생성과 움직임 생성 분리)**

- 고품질 이미지 텍스트 데이터셋에서 Stable Diffusion 모델을 Image Consistency Model로 Distillation.
- 이후 2D 컨볼루션 커널을 3D 커널로 팽창시키고 추가적인 시간적 레이어를 추가.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/bb393418-f00e-453c-9e6e-adc07d5c58c2/Untitled.png)

**Initialization strategy**

- 초기화 과정에서 Online Consistency Model에만 사전 학습된 공간적 LoRA 가중치를 삽입하고, Target Consistency Model에는 삽입하지 않음.
- 이를 통해 훈련 중 EMA를 통해 점진적으로 가중치를 목표 모델로 전달합니다.

$$
⁍
$$

- 여기서 $\lambda_n$은 시간에 따른 가중치 함수이며, 허버 손실을 거리 측정 함수로 사용

## **Teacher-Free Adaptation**

**One-step MCMC approximation of the actual score** 

- 스코어를 근사하기 위해 다음과 같이 표현.

$$
⁍
$$

- 이는 실제로 두 개의 같은 노이즈를 공유하는 SDE로서의 스코어 근사를 나타냄.

**Image-to-video generation**

- 입력 이미지를 사전 처리하여 이미지를 잠재 공간으로 인코딩하고, 시간 차원에서 반복하여 비디오 프레임 수와 맞춤.
- 그런 다음 해당 특징을 추출하여 U-Net의 다양한 레이어에 추가.

**Controllable video generation**

- 이미지 확산 모델에서 훈련된 레이아웃 제어 어댑터를 통합하여 제어 가능한 비디오 생성을 수행.
- 이 과정에서 LoRA 레이어를 튜닝하여 호환성을 높임.
- 이와 같은 방법론을 통해 AnimateLCM은 높은 효율성과 품질로 고화질 비디오 생성을 실현.

# **Experiments**

## **5.1 Experimental Setup (실험 설정)**

**Implementation details**

- 대부분의 실험에서 Stable Diffusion v1.5를 기본 모델로 사용하며, DDIM ODE 솔버를 사용하여 훈련.
- Latent Consistency Model을 따르며, 전체 1000 타임스텝 중 50 타임스텝을 균등하게 샘플링하여 훈련.
- Stable Diffusion v1.5와 공개된 모션 가중치를 사용하여 교사 비디오 확산 모델로 사용.
- 모든 실험(제어 가능한 비디오 생성 제외)은 WebVid-2M 공개 데이터셋에서 수행되었으며, 추가적인 데이터 증강이나 추가 데이터 없이 진행.
- 제어 가능한 비디오 생성을 위해 BLIP로 캡션된 TikTok 데이터셋을 사용하여 모델을 훈련.

**Benchmarks**

- 평가를 위해, 동작 인식 작업을 위해 큐레이션된 UCF-101 데이터셋을 사용.
- 각 카테고리에 대해 GPT-4를 사용하여 매우 간결한 캡션을 생성하고, 각 카테고리마다 16 프레임의 512 × 512 해상도의 비디오 24개를 생성.
- FVD(Frechet Video Distance)와 CLIP-SIM(CLIP Similarity)를 평가 메트릭으로 사용.
- FVD를 위해 UCF-101 데이터셋과 생성된 비디오 각각에서 무작위로 2048개의 비디오를 선택.
- CLIP-SIM은 CLIP ViT-H/14 LAION-2B를 사용하여 비디오의 모든 프레임과 간결한 캡션의 유사성 평균 값을 계산.

## **Qualitative Results (정성적 결과)**

**Generation Results (생성 결과)**:

- 텍스트에서 비디오 생성, 이미지에서 비디오 생성, 제어 가능한 비디오 생성을 포함한 4단계 생성 결과(그림 7)
- 다른 평가 단계에서의 생성 결과는 AnimateLCM이 일관성을 잘 따르며 다양한 단계에서도 비슷한 스타일과 움직임을 유지함을 보여줌.
- NFE(Number of Function Evaluations)가 증가할수록 생성 품질도 증가하여 25 및 50 단계에서 교사 모델과 경쟁적인 성능을 달성.

## **Quantitative Experiments (정량적 실험)**

**Comparison with Baselines (기준 모델과의 비교)**:

- Table 1에서 AnimateLCM과 DDIM, DPM++ 같은 강력한 기준 모델과 비교한 정량적 메트릭 결과를 보여줌.
- AnimateLCM은 특히 Low Steps(1~4)에서 기준 모델을 크게 능가.
- AnimateLCM은 분류기 없는 가이던스를 적용하지 않고도 모든 메트릭에서 높은 성능을 보이며, 이는 추론 피크 메모리 비용과 시간을 절약함.
- 공용 개인화된 리얼리틱 스타일 모델의 공간 가중치를 교체하여 추가 성능 향상을 확인함.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/997f9e5d-4d8f-4198-a197-6f57801721b8/Untitled.png)

## **Discussion (논의)**

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/9fe9eb8b-14b7-4eb1-bf4b-a109153d4376/Untitled.png)

### **Effectiveness of Decoupled Consistency Learning (분리된 일관성 학습의 효과)**:

- 분리된 학습 전략과 특정 초기화 설계의 효과를 검증합니다. "w/o decouple"과 "w/o init" 두 가지 기준과 비교함.
- 분리된 일관성 학습과 초기화 전략이 결합된 경우 가장 높은 성능을 보여줌.

### **Effectiveness of Teacher-Free Adaptation (교사 없는 적응의 효과)**:

- Fig. 6에서 교사 없는 적응 전략을 적용한 후 제어 가능한 비디오 생성 결과를 보여줍니다. 적응 후 안정된 제어와 더 나은 시각적 품질을 달성함.

### **Properties of Personalized Models (개인화된 모델의 특성)**:

- 개인화된 모델을 사용하여 다양한 스타일의 고품질 비디오 생성을 실현할 수 있음. 공용 개인화된 리얼리틱 비전 V5.0의 성능을 검증한 결과, FVD 및 CLIPSIM 메트릭에서 향상을 확인.
- 개인화된 모델을 적용함으로써 시각적 품질뿐만 아니라 움직임의 부드러움과 일관성도 개선됨.

# **6. Conclusion (결론)**

> AnimateLCM은 비디오 생성 가속화에 있어 혁신적인 발전을 이룸. 이미지 생성 및 움직임 생성을 분리한 일관성 학습 전략을 통해 훈련 효율성과 생성 품질을 높였으며, 다양한 어댑터를 효율적으로 통합할 수 있는 전략을 제안함.
> 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/bb81fa66-cec1-4f03-ba99-73d6d48ae3b7/Untitled.png)