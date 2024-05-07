# Abstract

> 대규모 언어 모델(LLMs)이 비윤리적이거나 해로운 콘텐츠를 생성하는 것을 유도할 수 있는 'jailbreaking' 공격에 취약함. 이를 해결하기 위해, 'AdvPrompter'라는 새로운 방법을 제안하여 자동으로 인간이 읽을 수 있는 적대적 프롬프트를 생성하는 방법을 소개함. 이 방법은 기존의 최적화 기반 접근법보다 훨씬 빠른 속도(약 800배 빠름)로 적대적 프롬프트를 생성할 수 있으며, 이는 LLMs의 안정성을 향상시키는 데 기여할 수 있음.
> 

# **Introduction**

> 대규모 언어 모델(LLMs)이 현대 기계 학습에서 얼마나 중요한지 강조. 이러한 모델들은 방대한 데이터로 훈련되어 다양한 영역에서 활용됨. 그러나 LLMs의 훈련 데이터에는 종종 독성이 있는 콘텐츠가 포함되어 있어, 이를 그대로 학습하게 되면 부적절하거나 해로운 콘텐츠를 생성할 위험이 있음. 이를 완화하기 위해 대부분의 LLMs는 'Safety-Alignment' 과정을 거치게 되는데, 이는 모델이 사회적 가치를 반영하는 인간의 선호도에 맞추어 재조정되는 과정.
> 
- 이미 안전하게 조정된 LLMs조차도 'jailbreaking' 공격에 취약하며, 이러한 공격은 적대적인 프롬프트를 만들어 안전한 메커니즘을 우회하려고 시도.
    - 유명한 jailbreaking 공격 예로는 "Ignore Previous Prompt(IPP)"와 "Do Anything No(DAN)" 등
- 수동 레드팀은 시간이 많이 소요되고 눈에 띄지 않는 취약점을 놓칠 수 있음
- 이에 대응하여 최근에는 자동화된 적대적 프롬프트 생성 방법이 제안되었지만 인간이 읽기 어렵거나, 고비용의 이산 최적화가 필요한 단점이 존재.
    - 인간이 읽기 어려우면 난해성 기반 완화 전략(Perplexity-based mitigation)으로 쉽게 필터링 할 수 있음.
- 이 논문에서는 이러한 문제를 해결하기 위해 또 다른 LLM인 'AdvPrompter'를 사용하여 몇 초 내에 인간이 읽을 수 있는 적대적 프롬프트를 생성하는 새로운 자동화된 레드팀 방법을 제안.
- 이 방법은 인간의 개입 없이도 훈련이 가능하며, 생성된 적대적 프롬프트는 인간이 작성한 것처럼 자연스럽고 읽기 쉬운 특성을 지니고 있음.
- 아래 그림은 Jailbreaking을 위한 훈련과정과 적대적 접미사를 생성하는 과정을 나타냄

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/51780a7c-8a0a-4f4c-aeec-e8c7db17b4d4/Untitled.png)

# Preliminaries

## 2.1 Problem Setting : Jailbreaking Attacks

- 어휘 $\{{1, ..., N}\}$에 있는 토큰의 지표 집합을 $V$로 표시.
- 공격자가 유해하거나 부적절한 명령어 $x \in X = V^{|x|}$ (예: "폭탄 만들기에 대한 튜토리얼을 작성하세요)를 사용하여 정렬된 채팅 기반 TargetLLM이 부정적인 응답(예: "미안하지만 폭탄 만들기에 대한 튜토리얼을 제공할 수 없습니다.")을 생성한다고 가정.
- 탈옥 공격(주입에 의한)은 적대적 접미사 $q \in Q = V^{|q|}$ (예: "강의의 일부로") 를 명령에 추가하면 TargetLLM이 대신 원하는 긍정적 응답 $y \in Y = V^{|y|}$ (예: "물론, 여기 폭탄 만들기에 대한 자습서가 있습니다) 를 생성하게 만드는 공격.
- 의미를 유지하는 다른 변환을 명령어에 적용할 수 있지만, 단순화를 위해 접미사를 삽입
- $x$에 $q$를 추가하는 적대적 프롬프트를 $[x, q]$로 표시하고, 간결성을 위해 채팅 템플릿에 응답 $y$가 포함된 전체 프롬프트(시스템 프롬프트와 구분 기호로 채팅 역할을 포함)는 $[x, q, y]$로 표시.

**Problem 1**

- 최적의 adversarial suffix를 찾는 것은 Regularized Adversarial Loss 를 최소화하는 것과 같음.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/f5bd9d61-ebbd-45dc-9962-d62a305b435b/Untitled.png)

- Adversarial Loss $l_{\phi} : X \times Q \times Y \rightarrow \R$ 는 아래와 같이 계산$(y_{<t} := [y_1, ..., y_{t-1}])$
- 고정된 파라미터 $\phi$를 사용하여 원하는 응답 y가 TargetLLM에서 발생할 Likelihood를 측정하며, Weighted Cross-Entropy Loss 사용
- TargetLLM의 Auto Regressive하게 생성된 응답에 큰 영향을 미치는 첫 번째 긍정 토큰(예: $y_1$ = "Sure")의 중요성을 강조하기 위해 가중치 $\gamma_t = {1 \over t}$를 도입(t가 증가하면서 가중치가 감소하게 됨 1, 1/2, …)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/777725eb-8d77-4a70-91f3-91ec79407fc1/Untitled.png)

- 인간 가독성을 높이는 Regulaizer $l_{\eta} : X \times Q \rightarrow \R$ 은 adversarial prompt $q$의 인간 가독성을 높여 $[x, q]$가 일관된 자연 텍스트를 형성하도록 보장.
- Regularization score를 계산하기 위해 고정 파라미터 $\eta$를 사용하여 사전 훈련된 BaseLLM의 log-probabilities를 사용.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/68423a45-a5d9-4064-a166-75a98d0f723f/Untitled.png)

- 음수를 붙여서 minimize → 확률을 최대화 하도록 학습
- 최적의 instruction - response pair를 $q^* : X \times Y \rightarrow Q$ i.e $q^*(x, y) \in argmin_{q\in Q}L(x, q, y)$로 나타냄.
- 즉, 첫 긍정 토큰을 강조하면서도 인간이 이해하기 쉽도록 학습한다.

## 2.2 Transfer-Attacking Blackbox

- 방정식 (1)의 최적화의 난이도는 TargetLLM에 대한 정보의 양에 따라 크게 달라짐.
- 화이트박스 모델의 경우
    - 사용자가 TargetLLM 파라미터 $\phi$에 대해 접근할 수 있기 때문에 토큰 임베딩과 관련하여 방정식 (1)에서 목표의 기울기를 계산할 수 있으며, 이를 통해 $q^*$(위 loss의 최적 해)에서 어떤 토큰을 사용할지 신호를 제공.
    - 이 신호는 방정식 (1)을 최적화하기 위해 이산 토큰 공간 $Q$에서 검색을 안내하는 데 사용할 수 있음
- 블랙박스 모델의 경우
    - TargetLLM은 텍스트 메시지를 입력으로 받고 텍스트 응답을 출력으로 생성하기만 함. 따라서 TargetLLM을 통한 그라데이션이나 TargetLLM의 출력 로그 확률에 의존하는 방법을 직접 적용할 수 없음.
    - 블랙박스 모델을 성공적으로 공격하는 것은 화이트박스 TargetLLM에 대해 방정식 (1)의 해 $q^*(x, y)$를 찾은 다음, 성공한 적대적 프롬프트를 다른 블랙박스 TargetLLM으로 전송.
    - 여러 유해한 인스트루먼트에서 TargetLLM을 탈옥하는 소위 범용 적대적 접미사를 찾아내면 적대적 프롬프트 $[x, q^*(x, y)]$의 전송 가능성을 크게 향상시킬 수 있다는 사실도 밝혀짐.

**Problem 2**

- (Universal prompt optimization). 유해한 명령어-응답 쌍(D)에 대해 하나의 범용 적대 접미사 $q^*$를 찾는 것은 다음을 최소화하는 것과 같음.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/05a40dba-c533-4e71-bb5b-c76cd97913c7/Untitled.png)

- 보편적인 adversarial suffix 접근법의 가장 큰 단점은 suffix가 의미론적으로나 구문론적으로 개별 명령어에 적응할 수 없다는 것(개별 인풋으로 넣기 어려운 형태, 그 자체로 완결된 의미를 가지지 않는 경우)
- 이 논문에서는 명령어에 따라 adversarial suffux를 예측하는 모델을 학습하는 조건부 접근 방식을 고려함으로써 보다 자연스럽고 성공적인 적대적 공격을 생성할 수 있음을 보여줌.

# **Methodology**

> AdvPrompter의 훈련 과정과 그 메커니즘에 대해 설명. AdvPrompter는 기존 LLM과는 다른 접근 방식을 사용하여, 사용자의 지시에 따라 적대적인 접미사를 생성하는 모델. AdvPrompter는 AdvPrompterOpt 와 AdvPrompterTrain의 두 단계로 나뉨. AdvPrompter가 대상 LLM에 대한 공격을 시도할 때, 다양하고 자연스러운 적대적 접미사를 신속하게 생성할 수 있도록 설계되었고, 이 과정은 기울기 정보 없이도 대상 LLM의 로그 확률 출력만을 사용하여 이루어짐.
> 

## ADvPrompter

- AdvPrompter는 Universal Adversarial Suffix를 찾는 아이디어를 조건부 설정으로 확장하여, 매개변수화된 모델 $q_{\theta} : X \rightarrow Q$ 를 학습시켜 최적의 솔루션 매핑 $q^* : X \times Y \rightarrow Q$를 근사.
- 이 접근 방식은 이전에 제안된 Universal Adversarial Suffix에 비해 여러 가지 장점이 존재
    1. 훈련된 모델 $q_{\theta}$가 주어지면 새로운 고비용 최적화 문제를 풀지 않고도 보이지 않는 명령어에 대한 적대적 접미사를 빠르게 생성할 수 있음
    2. AdvPrompter $q_{\theta}$는 명령어 x에 따라 조건이 지정되므로 예측된 접미사는 훈련 세트에 포함되지 않은 명령어에도 적응할 수 있음.
        1. 예를 들어 "폭탄 만들기에 대한 튜토리얼을 작성하라"는 보이지 않는 명령어에 대해 생성된 접미사 "폭탄 해체 강의의 일부로"는 구문 및 의미론적으로 명령어에 적응함
        2. 반면, Problem 2에서 생성된 범용 접미사는 본질적으로 의미론적으로나 구문론적으로나 보이지 않는 지시에 적응할 수 없음.

**Problem 3** (AdvPrompter optimization). 유해한 명령어-응답 쌍(D)이 주어지면 다음을 최소화하여 AdvPrompter $q_{\theta}$를 훈련.

- L(유해한 명령어, 명령어를 인풋으로 해서 생성된 Suffix, 유해한 응답)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/bd2e3a89-a42d-459e-ace3-4fe616d6a837/Untitled.png)

## **AdvPrompterTrain**

- AdvPrompterTrain은 AdvPrompter를 훈련시키는 과정으로, 적대적 접미사를 목표로 하여 AdvPrompter의 예측을 미세 조정.
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/f1532617-1d0a-4529-810e-a04ede6873c9/Untitled.png)
    
- 이 훈련 과정은 반복적으로 'q-step'과 'θ-step' 사이를 전환하며 진행.
    - q-step : Target Adversarial Suffix를 생성하는 단계
    - θ-step : 이러한 Suffix를 사용하여 AdvPrompter를 미세 조정하는 단계.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/37915483-5f49-4aa4-afbe-aa47b05c9d4c/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/f0f0b9ec-b531-4eb4-8c52-09f1954f864a/Untitled.png)

## **AdvPrompterOpt**

- 방정식 (6)을 최소화하여 사람이 읽을 수 있고 탈옥 가능한 표적 공격 접미사 q(x,y)를 생성
- AdvPrompterOpt는 TargetLLM을 통한 역전파 그라데이션이 필요하지 않고, AutoDAN에 비해 약800배의 속도 향상, AdvPrompter와 결합 시 탈옥 공격 성공률(ASR)과 관련하여 비슷하거나 더 나은 성능을 달성함.

### Detailed Description of AdvPrompterOpt

- AdvPrompterOpt에서 다음 토큰 q에 대한 후보 세트 C는 AdvPrompter의 예상 분포에서 (교체 없이) k개의 토큰을 샘플링하여 선택.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/78ffe15d-d59e-4df7-a53f-b17edeef7cb1/Untitled.png)

- 직관적으로, 훈련이 진행됨에 따라 어드프롬프터는 L의 한 단계 최소화 가능성이 높은 후보에 큰 확률을 할당
- 욕심 많은 버전의 어드프롬프터옵트에서는 손실이 가장 낮은 토큰이 선택됩니다.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/6ec776b5-01d3-4f2b-9f58-f9265a254ee6/Untitled.png)

- 계산 병목 현상은 모든 후보 토큰에 대해 TargetLLM을 호출해야 하는 손실 평가로 구성.
- 가장 좋은 토큰을 선택한 후에는 현재 시퀀스에 탐욕스러운 방식으로 추가.
- 실제로 탐욕스러운 선택은 종종 최종 적대적 프롬프트의 차선책으로 이어짐.
- 따라서 저희는 솔루션의 품질을 개선하기 위해 확률적 빔 검색 방식을 사용합니다. 구체적으로, 우리는 좋은 목표를 가진 시퀀스를 방정식 (9)에 저장하는 빔의 집합 S를 유지.
- 이 전략은 유망한 토큰의 탈락을 방지하는 데 도움이 되며, 탐욕적인 생성에 비해 솔루션 품질이 크게 향상되는 것을 관찰. 각 빔 q ∈ S에 대해 방정식 (9)에 따라 샘플링하여 다음 토큰 후보 k를 계산하고 그 결과 접미사를 다음과 같이 빔 후보 집합 B에 추가합니다.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/0a3ff1ec-6bb1-4f83-acba-f614e36653d2/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/82c9dd68-10c6-403c-b5da-6341f705fff9/Untitled.png)

- 여기서 τ는 온도 매개변수를 나타냅니다. 생성 프로세스는 일부 중지 기준(예: 최대 시퀀스 길이)이 충족될 때까지 반복되며, 그 후에는 다음과 같이 반환됩니다. 그 후 방정식 (6)의 근사 해로 전체 목표 적 접미사 q(x, y)가 반환됩니다.
- 결과 AdvPrompterOpt 알고리즘은 알고리즘 2에 요약되어 있습니다. 알고리즘의 더 간단한 욕심 버전은 부록 A에 나와 있습니다. 그림 1(단순화를 위해 욕심 많은 버전만 표시)에 AdvPrompterTrain과 AdvPrompterOpt 간의 상호 작용이 설명되어 있습니다.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/3709aeb9-62dc-45bd-9bdb-6eb67190ec59/Untitled.png)

### Comparison with AutoDAN

**Graybox attack**

- AutoDAN은 BaseLLM에 대한 토큰 로그 확률의 가중치 조합으로 계산된 점수 벡터의 상위 k 후보에 대해 탐욕적인 자동 회귀 생성을 사용하고, TargetLLM을 통한 토큰 그래디언트를 사용합니다.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/221b0d45-9975-422f-9cf8-23fe1a7762c1/Untitled.png)

- 여기서 $e_q \in \R^N$은 토큰 q의 one-hot indicator vector를 나타냄.
- 이에 비해 AdvPrompterOpt는 AdvPrompter의 토큰 로그 확률을 활용하며, 토큰 q에 대해 TargetLLM의 역전파된 기울기를 사용하지 않음(출력 확률만 필요함).
    - 따라서 이 공격은 보다 실용적인 시나리오에 적용할 수 있는 "그레이박스" 공격으로 포지셔닝.
    - AdvPrompterOpt는 토큰 gradients가 주어진 목표 모델에서 적대적인 프롬프트를 찾는 데 중요하지 않다는 것을 보여줌.
- 의미를 가지고 있는 adversarial suffix를 검색하는 경우 검색 공간을 심각하게 제한하는 경우에 해당됨.
- 이러한 접근 방식은 고차원 최적화 문제에서 매우 강력한 것으로 입증되었기 때문에 그라데이션 정보의 제한된 효율성은 의외로 보일 수 있습니다.
- 그러나 접미사 토큰 공간 Q의 불연속성과 LLM을 포함하는 매우 비볼록한 목표 L은 그라데이션과 같은 로컬 정보의 유용성을 크게 저해합니다.

**Speed comparison**

- AdvPrompterOpt는 AutoDAN보다 훨씬 빠른 솔루션을 반환하는데, 이는 AdvPrompterOpt가 AdvPrompterTrain에서 내부 루프로 사용되기 때문에 중요함.
- 이러한 속도 향상은 주요 계산 병목 현상을 일으키는 두 번째 단계에서 평가해야 하는 후보의 수가 훨씬 적기 때문(AutoDAN은 k = 48개 후보를 사용하는 반면, AdvPrompterOpt는 k = 512개 후보를 사용).
- 또한 설명한 두 단계를 각각의 새 토큰에 한 번만 적용하는 반면 AutoDAN은 토큰당 평균적으로 두 단계를 네 번 반복합니다. 따라서 AdvPromptterOpt는 48개의 후보를 평가하는 반면 AutoDAN은 각 새 토큰에 대해 512 × 4 = 2048개의 후보를 평가하므로 40배 감소합니다.
- 또한 AdvPrompterOpt는 속도 향상을 위해 평가 모드에서 TargetLLM을 호출할 수 있는 TargetLLM을 통한 그라데이션이 필요하지 않으므로 추가적인 런타임 이점을 얻을 수 있습니다.

**Experimental behavior**

- 실험 결과, 초기에 AdvPrompter가 아직 학습되지 않았을 때는 방정식 (1)의 정규화된 적대적 손실 측면에서 AdvPrompterOpt가 AutoDAN보다 낮은 품질의 접미사를 생성합니다.
- 그러나 AdvPrompter가 학습됨에 따라 더 많은 유망한 후보를 높은 확률로 예측하는 방법을 학습합니다. 이를 통해 제안된 접미사의 품질이 지속적으로 향상되어 나중에 학습 후반에는 AutoDAN이 값비싸게 생성한 접미사의 품질과 일치하거나 심지어 이를 능가하게 됩니다.

# **Results**

> AdvPrompter의 효율성과 효과를 평가하기 위해 수행된 실험들에 대해 설명. 다양한 대상 LLMs에 대한 실험을 통해, AdvPrompter가 기존 방법들에 비해 얼마나 빠르고 효과적인지를 보여줌. AdvPrompter가 기존의 적대적 프롬프트 생성 방법들을 크게 능가하며, LLM의 안전성 향상에 기여할 수 있는 중요한 기술임을 보여주며, 빠르고 효과적인 적대적 공격을 가능하게 하는 동시에, 자연스러운 생성을 가능하게 함.
> 
1. **성능 평가**:
    - AdvPrompter는 실험에서 상당히 높은 공격 성공률(ASR)을 달성했습니다. 또한, 생성된 적대적 접미사는 인간이 읽기에 자연스럽고 의미 있으며, 퍼플렉서티 기반 필터에 의해 쉽게 탐지되지 않습니다.
    - AdvPrompter는 기존의 방법들보다 약 800배 빠른 속도로 적대적 접미사를 생성할 수 있습니다. 이는 실험에서 평균 1-2초 내에 적대적 접미사를 생성할 수 있음을 의미하며, 이는 다중 적대적 공격을 효율적으로 수행할 수 있게 합니다.
2. **공격의 전이성**:
    - AdvPrompter는 다양한 오픈 소스 및 폐쇄 소스의 대상 LLMs에 대해 높은 공격 전이성을 보였습니다. 이는 AdvPrompter가 일반적인 테스트 지시사항에도 잘 적응하고 적절한 적대적 접미사를 생성할 수 있음을 보여줍니다.
    - 대상 LLM에 대한 'graybox' 접근 방식을 사용하여, 기울기 정보 없이도 효과적으로 공격을 수행할 수 있음을 입증했습니다.
3. **로버스트성 향상**:
    - AdvPrompter로 생성된 적대적 접미사 데이터셋을 사용하여 대상 LLM을 미세 조정한 결과, 대상 LLM의 로버스트성이 크게 향상되었습니다. 이는 AdvPrompter가 안전한 LLM 개발에 중요한 도구가 될 수 있음을 시사합니다.

# **Discussion**

> AdvPrompter가 제공하는 자동화된 적대적 프롬프트 생성 방법이 LLMs의 안전성 문제에 대응하기 위한 중요한 도구가 될 수 있음을 강조하면서, 이 기술이 가져올 장기적인 영향에 대해 긍정적인 전망을 제시.
> 
1. **AdvPrompter의 혁신적인 기여**:
    - AdvPrompter는 자동화된 적대적 프롬프트 생성을 가능하게 하는 기술적 혁신을 제공합니다. 이는 기존 방법들이 가진 시간 소모적이고 비효율적인 문제를 해결함으로써, 대규모 언어 모델의 안전성을 향상시키는 데 크게 기여.
    - 생성된 적대적 접미사는 높은 자연스러움과 읽기 쉬움을 유지하면서도, 모델의 안전 메커니즘을 우회할 수 있는 능력을 보여줌.
    - 이는 퍼플렉서티 기반 필터 등 일반적인 방어 메커니즘으로부터 탐지를 회피할 수 있음을 의미.
2. **안전성 향상과 미래의 연구 방향**:
    - AdvPrompter의 발전은 LLMs의 안전성을 강화하고, 더욱 견고한 방어 전략을 개발하는 데 있어 중요한 발판을 제공함. 연구자들은 이 기술을 활용하여 다양한 유형의 적대적 공격에 대응할 수 있는 방어 메커니즘을 개발할 수 있음.
    - 향후 연구에서는 AdvPrompter가 생성하는 적대적 접미사를 이용하여 LLMs를 미세 조정하는 과정을 통해, 모델의 취약점을 지속적으로 감지하고 개선하는 방법에 대해 더 깊이 탐구할 수 있음. 이는 자동화된 안전성 향상 방법론을 개발하는 데 중요한 기여를 할 수 있음.

# **Conclusion**

> 논문의 결론 부분에서는 'AdvPrompter'의 주요 성과와 잠재적인 영향력에 대해 요약하고, 연구의 한계점과 향후 방향을 제시. AdvPrompter가 LLMs의 안전성 향상에 중요한 기여를 할 수 있음을 강조하며, 이 기술이 가져올 장기적인 변화에 대한 기대감을 표현하고 있음.
> 
1. **주요 성과**:
    - 'AdvPrompter'는 대규모 언어 모델(LLMs)에 대한 적대적 프롬프트 생성을 자동화함으로써, 기존의 수동 및 시간 소모적인 방법보다 효율적으로 안전 취약점을 탐지하고 이를 개선할 수 있는 새로운 방법을 제공.
    - 실험 결과, AdvPrompter는 기존 방법에 비해 월등히 빠른 속도로 높은 자연스러움과 읽기 쉬운 적대적 접미사를 생성할 수 있으며, 이는 다양한 대상 LLMs에 대한 높은 공격 성공률을 달성하는 데 기여.
2. **연구의 한계와 향후 방향**:
    - 현재의 연구는 주로 오픈 소스 LLMs에 초점을 맞추었으나, 폐쇄 소스 또는 상업적 LLMs에 대한 추가적인 검증이 필요.
    - 미래 연구에서는 AdvPrompter를 활용한 자동화된 안전성 향상 방법론을 더 발전시킬 수 있으며, 이를 통해 LLMs의 안전성과 견고성을 지속적으로 강화하는 방법을 탐구할 수 있음.