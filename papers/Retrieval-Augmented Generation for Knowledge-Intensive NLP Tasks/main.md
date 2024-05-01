[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)

# 3 Line SUMMARY

- <span style="color:yellow">학습되지 않은 데이터에 엑세스 할 수 있는 생성 모델을 제시함</span>
- <span style="color:yellow">사람들은 학습된 모델의 생성보다 RAG 생성을 더 선호하였음</span>
- <span style="color:yellow">검색 인덱스를 교체하여 재교육 없이 모델을 업데이트 할 수 있는 방법을 설명함</span>

---

# <span style="background-color:yellow">**Abstract**</span>

> 큰 사전 훈련된 언어 모델들이 파라미터에 사실 지식을 저장하고 있으며, downstream NLP task에 미세 조정될 때 최고의 성과를 냄. 그러나 이러한 모델들은 지식에 접근하고 정확하게 조작하는 능력이 제한되어 있어, 지식 집약적인 작업에서는 성능이 특정 아키텍처에 뒤처지는 경우가 많음. 또한, 결정에 대한 근거를 제공하고, 세계 지식을 업데이트하는 문제가 존재.
> 
- 사전 훈련된 seq2seq 모델(파라메트릭 메모리)과 위키피디아의 밀집 벡터 인덱스(비파라메트릭 메모리)를 결합한 새로운 RAG(검색-증강 생성) 모델을 소개.
- 다양한 지식 집약적 NLP 작업에 대해 평가되었고, 세 개의 오픈 도메인 QA 작업에서 최고의 성과를 보여줌.

# **1. Introduction**

> 사전 훈련된 신경 언어 모델이 데이터로부터 심층적인 지식을 학습할 수 있음을 설명. 이러한 모델들은 외부 메모리에 접근하지 않고도 매개변수화된 암묵적 지식 베이스로 작동할 수 있음. 그러나 이러한 모델들은 기억을 확장하거나 수정하기 어렵고, 예측에 대한 통찰을 제공하지 못하며, 때로는 '환각'을 일으킬 수 있음.
> 
- 이 문제를 해결하기 위해, 파라메트릭 메모리와 비파라메트릭(검색 기반) 메모리를 결합한 하이브리드 모델이 제안됨.
- 이러한 접근 방식은 지식을 직접 수정하고 확장할 수 있으며, 접근한 지식을 검사하고 해석할 수 있습니다.
- 논문은 REALM과 ORQA와 같이 최근에 도입된 모델들이 어떻게 다양한 검색 기반 모델을 활용하여 향상된 결과를 달성했는지를 언급하며, 이 논문에서 제안하는 RAG 모델은 시퀀스-투-시퀀스(seq2seq) 모델에 비파라메트릭 메모리를 통합하여 일반적인 목적의 미세 조정 접근 방식을 제시.

# **2. Methods**

> 입력 시퀀스를 사용하여 텍스트 문서를 검색하고, 이를 타겟 시퀀스 생성에 추가적인 컨텍스트로 사용하는 RAG 모델에 대해 설명. 이 모델은 $p_\eta$(retriever)와 $p_\theta$(generator) 두 가지 주요 구성 요소를 활용. 검색기는 주어진 쿼리에 대해 텍스트 패시지의 분포를 반환하며, 생성기는 검색된 패시지를 바탕으로 현재 토큰을 생성함.
> 

## 2.1 Models

**RAG-Sequence**

- 하나의 문서만을 사용하여 전체 시퀀스를 생성합니다. 검색된 최상위 K 문서를 사용하여 시퀀스의 가능성을 계산하고, 이를 평균화합니다.
- 하나의 문서만을 참조하기 z가 동일하여 우변이 성립

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/c78c2a8c-823d-47b3-a20e-d3cb6d6d505b/Untitled.png)

**RAG-Token**

- 각 타겟 토큰을 예측할 때 다른 문서를 사용. 이는 생성기가 응답을 생성할 때 여러 문서의 내용을 선택적으로 사용할 수 있게 합니다.
- 여러 문서를 참조하기 때문에 곱해서 더해줘야 함

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/f39d8d39-a62e-4b61-a05e-5d3a4657404f/Untitled.png)

## 2.2 Retriever : DPR

- Retriever $p_\eta(z|x)$는 DPR을 기반으로 함.
    - DPR(Dense Passage Retrieval)은 자연어 처리 분야에서 사용되는 문서 검색 기술.
    - 특히 질의 응답(Question Answering, QA) 시스템에서 주로 사용되며, 사용자의 질문에 가장 관련성 있는 문서 또는 문서의 일부(패시지)를 찾아내는 데 도움을 줍니다.
    - DPR은 이중 인코더(bi-encoder) 구조를 사용함.
    - 이 구조는 두 개의 BERT 모델을 기반으로 하는데, 하나는 문서를 인코딩하는 데 사용되고 다른 하나는 사용자 쿼리를 인코딩하는 데 사용됨.
    - 각 문서와 쿼리는 각각의 BERT 모델을 통해 밀집 벡터(dense vector) 형태의 표현으로 변환되며, 이 벡터들 사이의 유사도(내적을 사용)를 계산하여 가장 관련성 높은 문서를 검색합니다.
- TriviaQA, Natural Questions에 대한 질문에 대한 답변이 포함된 문서를 검색하도록 훈련되었음.
- 이러한 검색기는 사전 훈련된 이중 인코더를 사용하여 초기화되며, 문서 인덱스를 구축하는데 사용됨.
- 이 문서 인덱스는 "non-parametric memory"라고도 불리며, Retriever가 접근할 수 있는 문서의 데이터베이스 역할을함.

## 2.3 Generator : BART

- Generator  $p_\theta(y_i|x, z, y_{1:i-1})$는 BART-large를 사용
- 단순히 두 텍스트를 연결하여 모델의 입력으로 사용함으로써, 모델이 두 정보 소스에서 정보를 추출하고 통합할 수 있도록 함.
- 생성기 파라미터 *θ*를 *“parametric memory”* 라고 부릅니다.

## 2.4 Training

- 특정 문서가 검색되어야 한다는 직접적인 감독 없이 두 컴포넌트를 함께 학습

$**BERT_d$(문서 인코더)**

- 문서 인덱스는 주기적으로 업데이트해야 해야 함.
- 문서 인코더와 인덱스를 업데이트하는 것은 계산 비용이 높고, 이 과정은 자주 수행하기 어려움.
- 강력한 성능을 유지하기 위해 필수적이지 않다고 판단되어, 문서 인코더와 인덱스는 고정

$BERT_q$**(쿼리 인코더)**

- 쿼리 인코더와 BART 생성기만 미세 조정됨
- 효율성을 높이고 자원 사용을 최적화하는 선택.

## 2.5 Decoding

**RAG-Token**

- 표준 자동회귀 seq2seq 생성기로 볼 수 있으며, 전환 확률은 다음과 같이 정의됩니다:
- RAG-Token에서는 각 토큰에 대한 확률을 Standard Beam Decoder에 입력하여 디코딩을 수행.
- 이는 여러 문서의 정보를 종합하여 각 토큰의 최적의 확률을 계산하고, 이를 바탕으로 최종 텍스트를 생성.

**RAG-Sequence**

- 각 토큰의 확률을 개별적으로 계산할 수 없기 때문에, 전체 시퀀스에 대한 확률을 직접 계산해야 함.
- **Thorough Decoding**
    - 각 문서 $z$에 대해 beam search를 수행하고, 각 가설을  $p_\theta(y_i|x, z, y_{1:i-1})$를 사용하여 점수 부여
    - 가설 y가 빔에 나타나지 않는 문서 z에 대해 추가 포워드 패스를 실행하고 제너레이터 확률에 곱한 다음 빔 전체의 확률을 합산하여 한계값을 구함
    - 이로 인해 다양한 문서의 빔에서 나오지 않은 가설들도 생성될 수 있음
    - 특정 가설 $Y$에 대한 확률을 추정하기 위해, $Y$가 포함되지 않은 문서 $z$에 대해 추가적인 포워드 패스를 수행하고 제너레이터 확률에 $p_\eta(z|x)$를 곱한 후 이를 모든 빔에 대해 합산.
    - 계산량이 많지만 가장 정확함.
- **Fast Decoding**
    - 더 긴 Sequence의 경우 효율적인 디코딩을 위해 $p_\theta(y|x, z_i)\approx 0$ 으로 가정(빔 검색 중 $Y$가 생성되지 않은 경우).
    - 후보 세트 $Y$가 생성된 후 추가적인 포워드 패스를 실행할 필요가 없다는 것을 의미.
    - 빠르지만, 상대적으로 덜 정확함.

# 3. Experiments

> 다양한 지식 집약적 작업에서 RAG를 실험함. 모든 실험에서 비모수적 지식 소스로는
단일 Wikipedia 덤프를 사용. FAISS를 사용해 단일 MIPS 인덱스를 구축. 학습하는 동안
각 쿼리에 대해 상위 *k개의* 문서를 검색함.
> 

## 3.1 Open Domain QA

- 답변의 Negative Log Likelihood를 최소화하도록 훈련
- Parametric QA와 Non Parametric QA 성능 비교

## 3.2 Abstractive QA

- 단순히 문서에서 추출한 QA를 넘어 추상적인 텍스트 생성을 통해 질문에 답할 수 있음
- MSMARCO NLG 사용
    - 각 질문에 대해 검색 엔진에서 검색된 10개의 골드 구절, 검색된 구절에서 주석이 달린 전체 문장의 답변으로 구성됨.
    - 제공된 구절은 사용하지 않 고 질문과 답변만 사용.

## 3.3 Jeopardy Question Generation

- Non QA 환경에서 RAG의 생성능력을 평가하기 위함
- 제퍼디는 실체에 대한 사실로부터 실체를 추측하는 까다로운 형태의 질문 형식을 가짐
    - 질문 : 1986년 멕시코는 이 국제 스포츠 대회를 두 번 개최한 최초의 국가로 기록되었다
    - 답변 : 월드컵
- Q-BLUE-1 를 사용하여 평가
    - 엔티티 매칭에 더 높은 가중치를 부여한 BLEU의 변형
- 생성 사실성, 구체성 등 인적 평가
    - 사실성은 신뢰할 수 있는 외부 출처에 의해 진술이 확증될 수 있는지 여부
    - 구체성은 입력과 출력 간의 높은 상호 의존성
    - 평가자에게는 답변과 두 개의 생성된 질문이 표시되는데, 하나는 BART에서, 다른 하나는 RAG에서 제공됨

## 3.4 Fact Verification

- FEVER 데이터셋
    - 자연어 주장이 위키백과에 의해 지지 또는 반박되는지 또는 판단하기에 충분한 정보가 없는지 분류해야 함.
    - 검색 문제와 추론 과제가 결합된 문제

# **4. Results**

> RAG 모델이 다양한 지식 집약적 NLP 작업에서 어떻게 수행되었는지에 대한 분석. 특히, 오픈 도메인 질문 응답(QA), 추상적 질문 응답, 그리고 지퍼디 스타일의 질문 생성과 같은 다양한 작업에 대한 실험 결과 포함.
> 

## 4.1 Open Domain QA:

- RAG는 Natural Questions, WebQuestions, CuratedTrec 등의 데이터셋에서 최고의 성능을 보여주며, 기존의 Parametric, Non Parametric 접근 방식을 모두 능가.
- 특히 RAG-Sequence와 RAG-Token 모델은 기존 BERT 기반 시퀀스 모델과 비교하여 더 나은 정확도를 달성함.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/30b1b6b0-2fc6-4c0e-a55e-cc689cb4bf42/Untitled.png)

## 4.2 Abstractive QA

- RAG는 MS MARCO 데이터셋을 사용한 추상적 질문 응답 작업에서 또한 강력한 성능을 보였으며, BART 모델과 비교하여 더 정확하고 사실적인 응답을 생성했습니다.
- 참조 답을 생성하는 데 필요한 특정 정보가 있는 Gold Passage에 엑세스 하고, 해당 정보가 없으면 답을 구할 수 없는 문제가 많다는 점을 고려하면 인상적인 결과
    - 기존 모델을 모델이 답변을 생성하면 맞출 수도 있고, 못 맞출 수도 있는 상태에서 나온 점수
    - RAG는 참조를 하는 경우에만 답변을 함. 그럼에도 불구하고 SoTA에 근접하는 점수
    - RAG가 BART 보다 환각을 덜 일으키고 사실적으로 정확한 텍스트를 자주 생성함을 의미
    - 또한 RAG가 BART보다 더 다양하게 생성

## **4.3** Jeopardy Question Generation

- RAG-Tokne이 RAG-Sequence보다 더 나은 성능을 보임
    - 두 모델 모두 Q-BLEU-1에서 BART보다 더 나은 성능
- 인간 평가에서도 RAG는 BART보다 더 사실적이고 구체적인 질문을 생성했다는 평가.
- Table 3은 모델의 일반적인 Generation을 보여줌

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/02a645ce-6c70-40f2-a7b9-d2ee8180451c/Untitled.png)

- 제퍼디 문제는 종종 두 개의 개별 정보를 포함하는 경우가 많은데, RAG-Token은 여러 문서의 콘텐츠를 결합한 답변을 생성할 수 있기 때문에 더 좋은 성능을 발휘할 수 있음
- Sun을 생성한 다음엔 document 2의 posterior가 높아지고, A Farewell to Arms이 생성되었을 때는 documen 1의 posterior가 높아짐
- First Token이 생성된 후에는 document의 posterior가 flatten됨. Non-Parametric구성 요소는 Parametric Memory에 저장 된 특정 지식을 끌어내어 생성하는 데 도움을 줌.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/d870a3c0-24b4-44ef-971e-ef5bafabc848/Untitled.png)

## 4.4 Fact Verification

- 위의 Table 2 참조.

## 4.5 Additional Result

- **Generation Diversity**
    - 생성된 총 n-gram에 대한 고유한 n-gram의 비율을 계산하여 생성 다양성을 평가
- **Retrieval Ablations**
    - 훈련중에 리트리버를 정지시키는 절제 훈련을 실행함
    - BM25 리트리버와 비교했을 때 FEVER의 경우 BM25의 성능이 가장 좋음. 이는 FEVER가 엔티티 중심적이어서 단어 중첩 기반 검색에 적합하기 때문
- **Index hot-swapping**
    - Non Parametric Memory를 교체하는 것 만으로 RAG의 지식을 업데이트 할 수 있음을 보여줌
- **Effect of Retrieving more documents**
    - RAG-Sequence의 오픈 도메인 QA의 결과가 개선되지만 RAG-Token의 경우 10개일 때가 최고의 성능

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/fa7f548e-9893-45fe-918e-af35e2da5eb6/Untitled.png)