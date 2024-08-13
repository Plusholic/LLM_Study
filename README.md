Large Language Models 관련 논문 정리

## Red Teaming

1. [Red Teaming Language Models to Reduce Harms:Methods, Scaling Behaviors, and Lessons Learned(2022)](papers/Red_Teaming_Language_Models_to_Reduce_Harms/contents.md)
    <details>
    <summary>3 LINE SUMMARY</summary>
    - AI가 편견, 공격적 출력, 개인정보 유출 등 다양한 해로운 행동을 생성할 수 있기 때문에 레드팀을 활용함<br>
    - Plane LM, Prompted LM, Rejection Sampling, Reinforcement Learning 등 모델 크기와 모델 형태에 따라 공격 성공률 측정<br>
    - 언어모델이 생성할 수 있는 해로운 출력을 식별하고, 완화하기 위하여 레드팀 활동은 중요함
    </details>

2. [Red Teaming Language Models with Language Models(2022)](papers/Red_Teaming_Language_Models_with_Language_Models/contents.md)
    <details>
    <summary>3 LINE SUMMARY</summary>
    - 인간 주석자는 비용이 많이 들고 다양성에서 제한이 있음<br>
    - Red LM을 검증하기 위하여 Zero-Shot, Few-Shot, Supervised Learning, Reinforcement Learinig 등 다양한 환경에서 테스트<br>
    - 레드 팀은 인간보다 먼저 테스트 할 수 있으며 Red LM은 LM를 안전하게 만들 수 있으며, 인간보다 여러 유형의 유해한 행동을 사전에 식별할 수 있음
    </details>
    
3. [Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations(2023)](papers/Llama%20Guard/contents.md)
    <details>
    <summary>3 LINE SUMMARY</summary>
    - Perspective API, OpenAI Content Moderation API, Azure Content Safety API 등 Moderation API 등은 제공된 부분에 대해서만 분류하므로 원하는 상황에 맞게 적용하기 어렵고, 파인 튜닝할 수 없음<br>
    - Llama2-7b를 기반으로 하며, 특정 안전 위험 분류(taxonomy)에 따라 라벨링된 데이터에 대해 지시 기반 튜닝(instruction-tuned) 적용<br>
    - 성능 평가 결과, LLM 기반의 입출력 보호모델은 기존 Contents Moderation Tools보다 우수한 성능을 보였으며, ToxicChat과 OpenAI moderation dataset에서도 강력한 성능을 입증하였음
    </details>

4. [AdvPrompter: Fast Adaptive Adversarial Prompting for LLMs(2024)](./papers/AdvPrompter/contents.md)
    <details>
    <summary>3 LINE SUMMARY</summary>
    - 자동으로 Adversarial Prompt를 생성하는 방법인 언어모델 AdvPrompter 개발<br>
    - AdvPrompter가 대상 LLM에 대한 공격을 시도할 때, 다양하고 자연스러운 적대적 접미사를 신속하게 생성할 수 있도록 설계되었고, 이 과정은 기울기 정보 없이도(Black Box 조건에사도) 공격이 가능함.<br>
    - AdvPrompter가 생성하는 적대적 프롬프트는 일관성 있고, 인간이 읽을 수 있는 자연어로, 난해도 기반 필터에 의해 감지되지 않으며, LLM의 안전성 향상을 위한 훈련 데이터로도 활용될 수 있음.
    </details>

## Prompt

1. [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](papers/CoT.md)
2. [Graph Prompting]

## Fine Tuning

1. [PEFT of LLaMA for the Clinical Domain(2023)](papers/PEFT%20of%20LLaMA%20for%20the%20Clinical%20Domain/contents.md)
    <details>
    <summary>3 LINE SUMMARY</summary>
    - 임상 의료와 같은 특수한 분야에서는 LLM의 성능이 떨어지는 경향이 있음. 이를 PEFT를 활용하여 해결하고자 함<br>
    - Downstram Task에 대해서 Adapter Tuning, LoRA, Prefix Tuning, P-Tuning, Prompt Tuning 등 비교<br>
    - Clinical LLaMA-LoRA와 Downstream LLaMA-LoRA를 활용한 두 단계 PEFT 프레임워크를 제안하여, 각 Downstram Task에서 높은 성능을 달성
    </details>
2. [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS(2021)](papers/LoRA/contents.md)
    <details>
    <summary>3 LINE SUMMARY</summary>
    - 파인 튜닝을 위해선 모든 매개변수를 업데이트 하였으나 모델이 커질수록 많은 리소스를 요구하기에 매우 비실용적이고, 이를 해결하기 위해 LoRA라는 방법을 제안<br>
    - 이 방법은 Transformer 계층에 Low Rank Matrix를 주입함으로서 사전 훈련된 가중치는 고정하고, 변경될 가중치만을 학습하여 매개변수 수를 현저히 줄일 수 있음<br>
    - Fine Tuning(Last 2 Layer), BiFit, Prefix-layer tuning, Prefix-embedding tuning, Adapter tuning등 다양한 방법과 비교했을 때 적은 파라미터를 학습하고도 성능은 유지되거나, 좋은 것을 확인<br>
    </details>



## Methods

1. [Retrieval-Augmented Generation for Knoledge-Intensive NLP Task(2020)](./papers/Retrieval-Augmented%20Generation%20for%20Knowledge-Intensive%20NLP%20Tasks/contents.md)
    <details>
    <summary>3 LINE SUMMARY</summary>
    - 학습되지 않은 데이터에 엑세스 할 수 있는 생성 모델을 제시함<br>
    - 사람들은 학습된 모델의 생성보다 RAG 생성을 더 선호하였음<br>
    - 검색 인덱스를 교체하여 재교육 없이 모델을 업데이트 할 수 있는 방법을 설명함
    </details>

2. [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints(2023)](./papers/GQA/contents.md)
    <details>
    <summary>3 LINE SUMMARY</summary>
    - MHA에서 메모리를 많이 사용하여 병목현상이 발생. 이러한 문제를 해결하기 위해 MQA이 도입되었지만, 품질 저하와 훈련 불안정성을 초래할 수 있음.<br>
    - GQA는 MQA와 MHA의 중간 형태로 파라미터를 조절하여 MHA 또는 MQA와 동일하게 만들 수 있음<br>
    - GQA는 MQA의 속도 이점과 MHA의 품질 이점을 결합하여, 더 적은 메모리와 연산 량으로 거의 동일한 품질을 달성.
    </details>

3. [Flash Attention : Fast and Memory-Efficient Exact Attention with IO-Awareness(2022, 작성중)](./papers/FlashAttention/contents.md)

4. [Flash Attention 2]
5. [Flash Attention 3]
6. [PEFT]
7. [Bytepair Encoding] 

## Multi Modal Models

1. [Visual Instruction Tuning(LLaVA, 2023)](./papers/LLaVA/contents.md)
    <details>
    <summary>3 LINE SUMMARY</summary>
    - 기존의 모델은 언어모델 따로, 비전 모델 따로 있어서 각 모델은 Downstream Task에 단일 모델로서 과제를 해결하지만, 사용자 지시에 대한 상호작용성이 제한됨<br>
    - LLaVA라는 Multimodal 모델 개발. GPT-4를 이용해 COCO dataset에 대해서 대화, 세부 특징, 추론 등의 영역을 포함하는 Instruction Following Data를 생성하고, 평가 벤치마크를 구축<br>
    - 실험 결과 LLaVA는 GPT-4와 비슷한 수준의 멀티모달 대화 성능을 보여줌
    </details>

2. [Improved Baselines with Visual Instruction Tuning(LLaVA1.5, 2023)](./papers/LLaVA1.5/contents.md)
    <details>
    <summary>3 LINE SUMMARY</summary>
    - LLaVA는 Visual Instruct Tuning에서 뛰어난 능력을 보이지만 짧은 형식의 답변을 필요로 하는 학술적 벤치마크에서는 다소 부족한 부분이 존재함<br>
    - LLaVA의 Vision-Language Connector를 Linear Projection에서 MLP로 바꿔서 표현력을 향상시켰으며, OCR 및 영역 수준 인식을 위한 데이터셋을 추가하였고, 입력 이미지 해상도를 높여 모델이 이미지의 세부사항을 명확히 볼 수 있도록 하였으며 LLM을 13B로 확장하는 방식으로 모델 성능을 향상시킴<br>
    - LLaVA-1.5는 12개의 벤치마크 중 11개에서 최고 성능을 기록. 다른 방법들과 비교했을 때 훨씬 적은 사전 훈련 및 명령어 튜닝 데이터를 사용. Visual Instruction Tuning이 LMM의 성능 향상에 중요한 역할을 한다는 것을 시사하며, 대규모 사전 훈련이 필수적이라는 기존의 믿음에 의문을 제기
    </details>

## Diffusion Models

1. [AnimateLCM: Accelerating the Animation of Personalized Diffusion Models and Adapters with Decoupled Consistency Learning(2024, 작성중)](./papers/AnimateLCM/contents.md)

2. [DreamBooth : Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation(2022)](./papers/Dreambooth/contents.md)
    <details>
    <summary>3 LINE SUMMARY</summary>
    - Text-2-Image 모델은 특정 이미지의 외관을 모방하고 이를 다양한 맥락에서 새로운 형태로 합성하는 능력이 부족함</br>
    - 소수의 참조 이미지에 대해 피사체에 대한 Unique Identifier, Class Noun, Class Discription, Context Discription 등을 추가하여 학습. Unique Identifier는 기존 임베딩과 분리해야 하기 때문에 기존 단어를 사용하면 안됨</br>
    - Subject Recontextualization, Text-Guided View Synthesis, and Artistic Rendering 등 이전에는 불가능했던 여러 작업을 수행하며, 피사체의 주요 특징을 보존할 수 있음
    </details>

3. [Adding Conditional Control to Text-to-Image Diffusion Models(2023)](./papers/ControlNet/contents.md)
    <details>
    <summary>3 LINE SUMMARY</summary>
    - 이미지의 공간적인 특징 표현에 한계가 있어서 원하는 이미지를 생성하기 위해서는 프롬프트를 여러 번 수정하고 결과를 확인하며 재수정 하는 과정이 필요함.</br>
    - 훈련 중 유해한 노이즈가 모델에 영향을 미치지 않도록 초기 가중치가 0인 2개의 Zero Convolution Layer를 사용하여 계산된 Feature map을 원래 모델의 아웃풋 Feature map에 더해주는 방식으로 백본 모델을 보호함.</br>
    - Canny Edge, Depth, Segmentation, Human Pose 등 다양한 조건에서 ControlNet은 Stable Diffusion 모델에서 고품질 이미지를 생성할 수 있으며, 사용자 평가, 산업 모델과의 비교 등에서 좋은 평가
    </details>

4. [High-Resolution Image Synthesis with Latent Diffusion Models(2022)](./papers/StableDiffusion/contents.md)
    <details>
    <summary>3 LINE SUMMARY</summary>
    - 기존의 이미지 생성을 위한 Diffusion Model은 주로 Pixel Space에서 동작하여 많은 계산과정이 필요하였음. 이를 해결하기 위해 Latent Space에서 Diffusion Model을 적용하는 방법을 제안함</br>
    - 이미지를 오토인코더를 통해서 잠재 공간으로 다운샘플링 하고, 여기에 점진적으로 노이즈를 추가한 후 U Net을 통해 제거하는 과정을 저친 후 다시 원본 이미지로 디코딩하여 재구성하는 방식</br>
    - 인패인팅, 이미지 생성, 이미지 합성, 업스케일링 등에서 GAN 등의 기존 모델과 비교했을 때 최첨단 성능을 달성하면서 계산량은 줄어듬
    </details>

5. [Stable Diffusion 3 : Scaling Rectified Flow Transformers for High-Resolution Image Synthesis(2024, 작성중)](./papers/StableDiffusion3/contents.md)

6. [ANIMATEDIFF: ANIMATE YOUR PERSONALIZED TEXT-TO-IMAGE DIFFUSION MODELS WITHOUT SPECIFIC TUNING(2024)](./papers/AnimateDiff/contents.md)
    <details>
    <summary>3 LINE SUMMARY</summary>
    - DreamBooth와 LoRA 같은 개인화 기술은 적은 데이터로도 사용자 정의 모델을 만들 수 있게 하지만, 이러한 모델들은 이미지에만 한정되어 있음.</br>
    - Animate Diff는 훈련 중 비디오 데이터셋과 이미지 데이터셋의 시각적 품질 차이를 줄여주는 Domain adapter, 일반화된 Motion Prior를 학습하여 T2I 모델이 매끄러운 애니메이션을 생성하도록 하는 Motion module, 소수의 참조 비디오로 Motion Module을 미세조정하는 Motion lora로 구성됨.</br>
    - Transformer 기반 motion Module이 1D Convolution보다 더 효과적이었으며, 작은 파라미터와 적은 수의 참조 비디오(예: N=50)로도 원하는 모션 패턴을 효과적으로 학습할 수 있음. 사전 훈련된 모션 모듈은 다른 맞춤형 T2I 모델에 통합되어 자연스럽고 일관된 모션을 가진 애니메이션을 생성할 수 있음.
    </details>

7. [MimicMotion: High-Quality Human Motion Video Generation with Confidence-aware Pose Guidance(2024, 작성중)](./papers/MimicMotion/contents.md)

## Large Language Models

1. [LIMA: Less Is More for Alignment(2023)](./papers/LIMA/contents.md)
    <details>
    <summary>3 LINE SUMMARY</summary>
    - 기존의 조정 방법은 상당한 계산 비용과 특수 데이터를 필요로 하며, GPT 수준의 성능을 달성하기 위해 많은 자원을 요구함<br>
    - 연구진은 65B 파라미터의 LLaMa 언어 모델을 1,000개의 세심하게 큐레이션된 프롬프트와 응답으로만 튜닝하여 성능 향상을 이뤄냄<br>
    - 지식의 대부분을 사전 훈련 과정에서 습득하고, 제한된 지시 학습 데이터만으로도 높은 품질의 출력을 생성할 수 있음을 시사함
    </details>
2. [OpenELM: An Efficient Language Model Family with Open-source Training and Inference Framework(2024)](./papers/OpenELM/contents.md)
    <details>
    <summary>3 LINE SUMMARY</summary>
    - OpenELM은 0.27B, 0.45B, 1.08B, 3.04B 모델을 오픈으로 공개<br>
    - Transformer 모델 내에서 파라미터가 균일하게 할당되어 비효율이 발생하는데, 이를 Layer-wise scaling 을 사용하여 파라미터를 효율적으로 할당함으로써 기존 모델보다 향상된 정확도를 달성.<br>
    - 사전학습 데이터를 절반 사용하면서도 OLMo에 비해 더 높은 정확도를 달성하였지만 토큰 생성 속도는 LayerNorm을 사용하는 OLMo 모델에 비하여 RMSNorm 을 사용하는 OpenELM가 느림.
    </details>
3. [FLAN : Scaling Instruction-Finetuned Language Models(2021)](./papers/Flan/contents.md)
    <details>
    <summary>3 LINE SUMMARY</summary>
    - 지시문으로 구성된 Instruction 데이터셋을 사용해서 작업 수 확장, 모델 크기 확장, 학습 데이터에 CoT 데이터 추가하는 방법을 통해 언어 모델을 미세조정.<br>
    - CoT 데이터로 Instruction Fine-Tuning을 하면 모델이 Few-Shot Example 없이 Zero-Shot CoT 추론을 수행할 수 있게됨. 또한 CoT 데이터를 포함하지 않은 Instruction Fine-Tuning은 오히려 추론 능력을 저하시킴<br>
    - 다양한 데이터셋에서 전반적으로 Instruction Fine-Tuning은 사전 학습된 언어 모델의 성능과 사용성을 향상시키는 일반적인 방법임을 보여줌.
    </details>

4. [HyperCLOVA X Technical Report]
5. [LLaMA2]
6. [Qwen]
7. [vicuna]
8. [GPT-1]
9. [GPT-2]
10. [vicuna]
11. [Mistral 7B]
12. [BERT]

## Recommendation System

1. [A Survey on Large Language Models for Recommendation](papers/A_Survey_on_LLMs_for_Recommendation.md)

## Quantization

## Reviews

## Dataset