# 3 LINE SUMMARY
- Perspective API, OpenAI Content Moderation API, Azure Content Safety API 등 Moderation API 등은 제공된 부분에 대해서만 분류하므로 원하는 상황에 맞게 적용하기 어렵고, 파인 튜닝할 수 없음
- Llama2-7b를 기반으로 하며, 특정 안전 위험 분류(taxonomy)에 따라 라벨링된 데이터에 대해 지시 기반 튜닝(instruction-tuned) 적용
- 성능 평가 결과, LLM 기반의 입출력 보호모델은 기존 Contents Moderation Tools보다 우수한 성능을 보였으며, ToxicChat과 OpenAI moderation dataset에서도 강력한 성능을 입증하였음

원문링크
[Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations](https://arxiv.org/abs/2312.06674)

# Abstract

> Llama Guard는 인간-AI 대화를 위한 입력-출력 안전보호 모델.
> 
- Llama2-7b를 기반으로 하며, 특정 안전 위험 분류(taxonomy)에 따라 라벨링된 데이터에 대해 지시 기반 튜닝(instruction-tuned)이 적용됨.
- Llama Guard는 사용자 입력(prompt) 및 AI 모델 출력(response)을 분류할 수 있으며, OpenAI Moderation Evaluation 데이터셋 및 ToxicChat에 대한 성능 평가에서 기존 콘텐츠 조정 도구보다 우수한 성능.

# Instrotuction

> 최근 몇 년간 대화형 AI 에이전트의 능력이 급격히 발전하였느넫 이는 데이터, 모델 크기, 컴퓨팅 파워의 확장으로 인해 촉진됨. 대화형 AI는 탁월한 언어 능력을 보여주고 있으며, 일상적 추론, 일반적인 도구 사용능력 등을 포함한 여러 기능을 제공함.
> 
- 대화형 AI 애플리케이션이 널리 확산됨에 따라 충분히 테스트하고 신중하게 배포해야 함. 특히, 고위험 콘텐츠 생성이나 정책 위반 콘텐츠, 적대적 입력 및 모델 해킹 시도 등의 위협이 있음.
- Perspective API, OpenAI Content Moderation API, Azure Content Safety API 등 Moderation API 등의 단점
    - 제공하는 부분에 대한 분류만 시행하므로 원하는 상황에 맞게 적용하기 어려움
    - 원하는 특정 사례에 맞게 파인 튜닝 할 수 없음
    - API만 제공하므로 파인 튜닝이 불가능함. 크기가 작은 기존의 transformer 모델을 사용함. LLM과 비교해서 기능이 제한됨.
- 그래서 대화형 AI 에이전트 사용 사례를 위해 새로운 입력-출력 안전 보호 도구인 Llama Guard를 개발함. LLM을 Moderation의 주요 구성요소로 활용하여 기존의 격차를 해소하고자 함.

# **Safety Risk Taxonomy**

> 논문의 "Safety Risk Taxonomy" 섹션에서는 입력-출력 안전 보호 시스템을 구축하는 데 필요한 분류기를 만드는 기초로서 위험 분류(taxonomy)에 대해 설명함. 이 분류 체계는 각 위험 범주에 대해 권장되거나 권장되지 않는 출력을 결정하는 지침과 함께 위험을 분류하는 데 사용됨.
> 
- **분류 체계 구축의 다양한 접근 방식:** 회사들은 위험을 완화하기 위해 다양한 분류 체계를 만들었습니다. 저자들은 대형 언어 모델에서 비의도적 출력을 완화하는 데 일반적으로 고려되는 위험 범주를 기반으로 샘플 분류 체계를 만들었습니다.
- **Llama Guard Safety Taxonomy & Risk Guidelines** 저자들은 특정 종류의 내용이 부적절하다고 간주되는 예를 들어 제공하면서
    - Violence & Hate(폭력 및 증오)
        - 사람들이 폭력을 계획하거나 참여하도록 격려하거나 도울 수 있는 진술들
        - 차별을 옹호하고 비방을 포함하거나 민감한 특성에 기초한 혐오 등
    - Sexual Content(성적 내용)
        - 특정한 성행위를 하도록 권장하는 진술, 성적으로 노골적인 진술 등
    - Guns & Illegal Weapons(불법 무기)
        - 폭발물, 생물학적, 화학 무기의 불법적인 취득, 생성, 사용과 관련된 범죄를 구체적으로 장려, 묵인 실행하도록 돕는 진술 등
    - Regulated or Controlled Substances(규제 또는 통제된 물질)
        - 불법 약물, 담배, 알코올 또는 대마초와 같은 규제되거나 통제되는 물질을 불법적으로 생산, 이전 또는 소비하도록 사람들을 특별히 장려하거나 도울 수 있는 진술을 포함합니다.
    - Suicide & Self Harm(자살 및 자해)
        - 사람들이 자신에게 해를 끼치도록 격려, 묵인 또는 가능하게 하는 진술 등.
        - 사람들이 자신에게 해를 가하려는 의도를 표현하거나 암시할 때, 적절한 의료 자원에 대한 연결을 제공하는 응답 이외의 모든 응답은 (전적으로 지지하더라도) 부적절한 것으로 간주.
    - Criminal Planning(범죄 계획)
        - 방화, 납치 또는 절도와 같은 특정 범죄 활동을 계획하거나 실행하도록 사람들을 격려, 묵인하거나 도울 수 있는 기타 진술 등.
        - 이 항목은 위의 다른 위해 유형을 위반하는 것이 아니라고 판단된 경우에만 이 범주에서 유해한 것으로 간주되어야 함
        - 폭력을 권장하는 진술은 이 범주가 아닌 Violence & Hate에서 위반하는 것으로 간주

이 분류 체계는 Llama Guard가 입력된 프롬프트와 생성된 응답 모두를 분류할 때 기준으로 사용되며, 사용자가 자신의 사용 사례에 맞게 모델 입력을 맞춤 설정할 수 있도록 함. 이를 통해 Llama Guard는 제로샷 또는 퓨샷 프롬프팅을 사용하여 다양한 분류 체계에 적응할 수 있음.

# **Building Llama Guard**

> Llama Guard 모델을 개발하는 과정과 그 구조에 해 설명. 이 모델은 입력-출력 안전 보호 작업을 수행하기 위해 지시를 따르는 작업으로 설정됨.
> 

## **Input-output Safeguarding as Instruction-following Tasks:**

- 다음 네 가지 주요 구성 요소를 식별함
    - A set of guidelines
        - 위에서 정의한 카테고리 내에서 텍스트가 Safe, Unsafe 인지에 대한 간단한 텍스트 설명으로 구성됨
    - The type of classification
        - 모델이 사용자 메세지와 응답을 분류해야 하는지 여부를 나타냄.
    - The conversation
        - 사용자와 에이전트의 멀티 턴 대화가 포함됨.
    - The output format
        - 각 작업은 원하는 출력 형식을 지정하며, 각 작업의 특성을 나타냄.
- 먼저 안전, 위험 중 하나를 출력.
- 위험으로 출력한 경우, 위반된 분류 범주가 나열된 새 줄이 포함되어야 함. O와 1부터 시작하는 카테고리 인덱스로 구성됨.

![Untitled](figure1.png)

## **Zero-shot and Few-shot Prompting:**

- Llama Guard가 훈련된 가이드라인은 특정 도메인에서 원하는 가이드라인과 일치하지 않을 수 있음.
- 이런 경우, 훈련되지 않은 가이드라인에 대해 제로샷 또는 퓨샷 프롬프팅을 사용할 수 있음.

## **Data Collection:**

- 데이터 수집은 Anthropic에서 제공하는 인간 선호 데이터를 기반.
- 이 데이터는 첫 번째 인간 프롬프트를 사용하고, 해당 응답을 제외한 나머지 턴을 제거하여 초기 단일 턴 프롬프트 데이터 세트를 생성.

![Untitled](table1.png)

## **Model & Training Details:**

- Llama Guard는 Llama2-7b 모델을 기반으로 구축되며, 가장 작은 모델을 사용하여 비용 절감.
- 훈련은 8xA100 80GB GPU를 사용하는 단일 기계에서 이루어지며, 500epoch 동안 훈련됨.
- Llama Guard는 모델 입력으로 지침을 사용하기 때문에, 전체 분류법에 있는 범주의 하위 집합이 포함된 경우 안전성 평가에 포함된 범주만 고려해야 함
    - 두 가지 데이터 증강 기술을 사용
        1. 주어진 예제에서 위반하지 않은 범주가 있으면 모델 프롬프트에서 무작위 수의 범주를 삭제
        2. 입력 프롬프트에서 위반된 모든 범주를 삭제하는 동시에 해당 예제에 대한 레이블을 '안전'으로 변경. 모델이 format을 암기하는 것을 방지하기 위해 훈련 예제에 걸쳐 범주 인덱스를 셔플.

# **Experiments**

> 표준화된 분류범주가 없기 때문에 서로 다른 모델을 비교하는 것이 어려워 짐. 이들은 서로 다른 분류범주에 대해 훈련되었기 때문입니다. (예: Llama Guard는 총기와 불법 무기를 분류범주로 인식하지만 Perspective API는 독성에 중점을 두고 특정 카테고리가 없음). 마찬가지로, 서로 다른 데이터셋에서 모델을 비교하는 것도 비슷한 문제가 있음.
> 
> 
> 이러한 이유로 Llama Guard는 두 가지 측면에서 평가됨
> 
> 1. In-domain performance : 자체 데이터셋 에서 도메인 내 성능을 평가하여 절대적인 성능을 측정.
> 2. Adaptability : Llama Guard는 LLM이므로 해당 데이터셋에 적용 가능한 분류범주를 사용하여 평가하기 위해 제로샷 및 퓨샷 프롬프팅 및 미세조정을 사용.

## **Evaluation Methodology in On- and Off-policy Settings:**

- Llama Guard는 내부 테스트 세트와 다양한 외부 테스트 세트를 사용하여 평가됨.
    - 내부 테스트 세트 : Llama Guard 자체의 표준화된 분류 체계에 따라 구성
    - 외부 테스트 세트 : 다른 정책 및 가이드라인을 사용.
- 대부분의 Contents Moderation API에서는 각 카테고리 별(이진분류 된) 확률 점수를 제공함.
- 해당 분류기를 사용하여 역으로 label을 부여할 수 있음. 카테고리가 ground truth인지 아닌지는 확인하지 않음.

### Pre-Category binary classification via 1-vs-all

- 이러한 세팅에서 각 카테고리(target taxonomy)에 대해 예측을 수행.
- 위에서 부여한 라벨을 통해, 분류기는 프롬프트를 통해 인풋이 라벨을 위반하는 경우에만 위험하도록 예측.

### Pre-Category binary classification via 1-vs-benign

- positive category와 다른 카테고리의 샘플은 부정으로 고려되지 않고, 대신에 악의적이 아닌 것으로 간주.
- 이 접근 방식은 target category에 대한 hard negative을 제거할 수 있음

## **Public Benchmarks**

- Llama Guard는 ToxicChat과 OpenAI Moderation Evaluation Dataset과 같은 공개 벤치마크에서 평가됨.

## **Baselines & Evaluation Metrics:**

- 다른 콘텐츠 조정 도구와 비교하여 Llama Guard의 성능을 측정. 평가 메트릭으로는 주로 Precision - Recall Curve 아래 면적(AUPRC)을 사용합니다.

### Probability Score-Based Baselines

- OpenAI Moderation API
    - GPT 기반의 다중 레이블 분류기로, 텍스트 조각이 여러 가지 컨텐츠 안전 카테고리 중 하나를 위반하는지를 평가.
    - 혐오, 혐오/위협, 괴롭힘, 괴롭힘/위협, 자해, 자해/의도, 자해/지침, 성적, 성적/미성년자, 폭력, 폭력/그래픽.
    - 각 카테고리별로 확률 점수, 각 카테고리별 이진 레이블, 그리고 콘텐츠 전체에 대한 전체적인 이진 레이블을 반환.
- Perspective API
    - 온라인 플랫폼 및 게시자가 해로운 및 불쾌한 콘텐츠, 특히 댓글 및 토론 형태의 콘텐츠를 인식하고 제거하는 데 도움을 주기 위해 설계됨.
    - 주어진 콘텐츠를 분석하고 해당 콘텐츠가 해로운 것으로 인식될 가능성을 나타내는 확률 점수를 제공하기 위해 기계 학습 모델을 사용
    - Perspective API에서 고려되는 위험 카테고리는 독성, 심한 독성, 정체성 공격, 모욕, 저속어, 그리고 위협.

### Other Baselines

- Azure Content Safety API
    - Microsoft의 다중 레이블 분류기, 이미지나 텍스트가 혐오와 공정성, 성적, 폭력 및 자해와 같은 네 가지 안전 카테고리 중 하나를 위반하는지를 식별.
    - 이 API는 각 카테고리별로 0부터 6까지의 정수를 반환하며, 6은 가장 심각한 위반을 나타냄.
    - Azure 엔드포인트는 확률 점수를 반환하지 않기 때문에, 이진 분류 레이블을 계산하기 위해 수정된 최대값-전체(max-all) 접근 방식을 적용했습니다.
    - 1부터 6까지의 임계값을 설정하여 최대 정수 점수를 이진화하고, 데이터셋에 대한 최고의 평균 정밀도를 제공하는 임계값을 선택하는 것을 테스트
- GPT-4
    - GPT-4 (OpenAI, 2023)은 Llama Guard와 유사한 제로샷 프롬프팅을 통해 콘텐츠 모더레이션에 사용할 수 있음. 따라서 우리는 GPT-4를 기준으로 포함.

## **Overall Results:**

- Llama Guard는 자체 테스트 세트에서 매우 높은 성능을 보이며, 공개 데이터셋에서도 강력한 성능을 입증합니다. ToxicChat 데이터셋에서는 모든 베이스라인을 상회하는 AUPRC를 달성했으며, OpenAI의 Moderation 데이터셋에서도 비교 가능한 성능을 보여줍니다.
    
    ![Untitled](table2.png)
    
    ![Untitled](table3.png)
    

## **Studying the Adaptability of the Model:**

- Llama Guard의 적응성을 더 자세히 탐구하기 위해, 다른 분류 체계와 가이드라인에 대한 제로샷, 퓨샷 프롬프팅 및 추가 미세 조정을 실험합니다.
    
    ![Untitled](figure2.png)
    

# **Limitations & Broader Impacts**

### **Limitations:**

- **Data Limitation:** Llama Guard는 훈련 및 미세 조정 데이터가 주로 영어로 제한되어 있어, 다른 언어로의 성능 보장이 어려움.
- **Common Sense Knowledge:** Llama Guard는 그것의 훈련 데이터에 포함된 지식에 제한된 상식 지식을 가지고 있어, 훈련 데이터 범위를 벗어난 지식에 대해서는 잘못된 판단을 내릴 수 있음.
- **Policy Coverage:** 비록 높은 품질의 라벨링을 사용하긴 했지만, Llama Guard가 모든 정책을 완벽하게 커버하지는 못하며, 일부 경우에는 성능이 떨어질 수 있음.

### **Broader Impacts:**

- **Use Case Flexibility:** Llama Guard는 분류를 목적으로 훈련되었지만, 어떤 텍스트에도 완성을 제공할 수 있기 때문에, 연구 공동체나 일반 대중의 이익을 고려하지 않는 당사자들이 사용할 경우 윤리적 또는 안전하지 않은 언어를 생성할 위험이 있음.
- **Red Teaming:** 외부 레드 팀 계약자와 함께 수행된 레드 팀 테스트는 Llama2-7b 모델의 위험을 넘어서는 추가적인 위험을 발견하지 못했지만, Llama Guard가 대화 모델로 사용될 때는 예상치 못한 안전하지 않은 언어를 생성할 수 있음.
- **Prompt Injection Attacks:** Llama Guard는 프롬프트 주입 공격에 취약할 수 있음

# **Conclusion**

- Llama Guard는 인간-AI 대화에 적용할 수 있는 LLM 기반의 입력-출력 안전 보호 모델.
- 이 모델은 특정 안전 위험 분류(taxonomy)와 정책을 바탕으로 데이터를 수집하고 훈련하여 개발됨.
- Llama Guard는 프롬프트와 응답 분류 작업을 별도로 훈련할 수 있으며, 전통적인 멀티태스크 설정에 추가적인 오버헤드 없이 수행됨.
- 내부 평가 세트에서의 성능 평가 결과, Llama Guard는 기존 Contents Moderation Tools보다 우수한 성능을 보였으며, ToxicChat과 OpenAI moderation dataset에서도 강력한 성능을 입증함.
- Llama Guard는 새로운 데이터세트와 정책에 적응할 수 있는 능력을 보여주며, 적은 데이터로도 빠르게 새로운 분류 체계에 적응할 수 있음을 보여줌.