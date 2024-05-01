Large Language Models 관련 논문 정리

## Red Teaming

1. [Red Teaming Language Models to Reduce Harms:Methods, Scaling Behaviors, and Lessons Learned](papers/Red_Teaming_Language_Models_to_Reduce_Harms/contents.md)
    <details>
    <summary>3 LINE SUMMARY</summary>
    - AI가 편견, 공격적 출력, 개인정보 유출 등 다양한 해로운 행동을 생성할 수 있기 때문에 레드팀을 활용함<br>
    - Plane LM, Prompted LM, Rejection Sampling, Reinforcement Learning 등 모델 크기와 모델 형태에 따라 공격 성공률 측정<br>
    - 언어모델이 생성할 수 있는 해로운 출력을 식별하고, 완화하기 위하여 레드팀 활동은 중요함
    </details>

2. [Red Teaming Language Models with Language Models](papers/Red_Teaming_Language_Models_with_Language_Models/contents.md)
    <details>
    <summary>3 LINE SUMMARY</summary>
    - 인간 주석자는 비용이 많이 들고 다양성에서 제한이 있음<br>
    - Red LM을 검증하기 위하여 Zero-Shot, Few-Shot, Supervised Learning, Reinforcement Learinig 등 다양한 환경에서 테스트<br>
    - 레드 팀은 인간보다 먼저 테스트 할 수 있으며 Red LM은 LM를 안전하게 만들 수 있으며, 인간보다 여러 유형의 유해한 행동을 사전에 식별할 수 있음
    </details>
3. [Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations](papers/Llama%20Guard/contents.md)
    <details>
    <summary>3 LINE SUMMARY</summary>
    - Perspective API, OpenAI Content Moderation API, Azure Content Safety API 등 Moderation API 등은 제공된 부분에 대해서만 분류하므로 원하는 상황에 맞게 적용하기 어렵고, 파인 튜닝할 수 없음<br>
    - Llama2-7b를 기반으로 하며, 특정 안전 위험 분류(taxonomy)에 따라 라벨링된 데이터에 대해 지시 기반 튜닝(instruction-tuned) 적용<br>
    - 성능 평가 결과, LLM 기반의 입출력 보호모델은 기존 Contents Moderation Tools보다 우수한 성능을 보였으며, ToxicChat과 OpenAI moderation dataset에서도 강력한 성능을 입증하였음
    </details>
4. [AdvPrompter: Fast Adaptive Adversarial Prompting for LLMs]



## Prompt

1. [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](papers/CoT.md)
2. [Graph Prompting]

## Fine Tuning
1. [PEFT of LLaMA for the Clinical Domain](papers/PEFT%20of%20LLaMA%20for%20the%20Clinical%20Domain/contents.md)
    <details>
    <summary>3 LINE SUMMARY</summary>
    - 임상 의료와 같은 특수한 분야에서는 LLM의 성능이 떨어지는 경향이 있음. 이를 PEFT를 활용하여 해결하고자 함<br>
    - Downstram Task에 대해서 Adapter Tuning, LoRA, Prefix Tuning, P-Tuning, Prompt Tuning 등 비교<br>
    - Clinical LLaMA-LoRA와 Downstream LLaMA-LoRA를 활용한 두 단계 PEFT 프레임워크를 제안하여, 각 Downstram Task에서 높은 성능을 달성
    </details>


## Methods
1. [Retrieval-Augmented Generation for Knoledge-Intensive NLP Task](./papers/Retrieval-Augmented%20Generation%20for%20Knowledge-Intensive%20NLP%20Tasks/contents.md)
    <details>
    <summary>3 LINE SUMMARY</summary>
    - 학습되지 않은 데이터에 엑세스 할 수 있는 생성 모델을 제시함<br>
    - 사람들은 학습된 모델의 생성보다 RAG 생성을 더 선호하였음<br>
    - 검색 인덱스를 교체하여 재교육 없이 모델을 업데이트 할 수 있는 방법을 설명함
    </details>


2. [PEFT]
3. [Bytepair Encoding]
4. 

## Models

1. [LIMA](./papers/LIMA/contents.md)
    <details>
    <summary>3 LINE SUMMARY</summary>
    - 기존의 조정 방법은 상당한 계산 비용과 특수 데이터를 필요로 하며, GPT 수준의 성능을 달성하기 위해 많은 자원을 요구함<br>
    - 연구진은 65B 파라미터의 LLaMa 언어 모델을 1,000개의 세심하게 큐레이션된 프롬프트와 응답으로만 튜닝하여 성능 향상을 이뤄냄<br>
    - 지식의 대부분을 사전 훈련 과정에서 습득하고, 제한된 지시 학습 데이터만으로도 높은 품질의 출력을 생성할 수 있음을 시사함
    </details>
2. [OpenELM: An Efficient Language Model Family with Open-source Training and Inference Framework](./papers/OpenELM/contents.md)
    <details>
    <summary>3 LINE SUMMARY</summary>
    - OpenELM은 0.27B, 0.45B, 1.08B, 3.04B 모델을 오픈으로 공개<br>
    - Transformer 모델 내에서 파라미터가 균일하게 할당되어 비효율이 발생하는데, 이를 Layer-wise scaling 을 사용하여 파라미터를 효율적으로 할당함으로써 기존 모델보다 향상된 정확도를 달성.<br>
    - 사전학습 데이터를 절반 사용하면서도 OLMo에 비해 더 높은 정확도를 달성하였지만 토큰 생성 속도는 LayerNorm을 사용하는 OLMo 모델에 비하여 RMSNorm 을 사용하는 OpenELM가 느림.
    </details>

2. [LLaMA2]
3. [Qwen]
4. [FLAN T5]
6. [vicuna]
7. [GPT-1]
8. [GPT-2]
9. [HyperCLOVA X Technical Report]
10. [vicuna]
11. [Mistral 7B]
12. [BERT]

## Recommendation System

1. [A Survey on Large Language Models for Recommendation](papers/A_Survey_on_LLMs_for_Recommendation.md)

## Quantization

## Reviews

## Dataset