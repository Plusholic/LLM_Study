# Abstract

> Sequence transduction 모델들이 복잡한 순환 신경망(RNN) 또는 합성곱 신경망(CNN) 기반으로 구성되어 있으며, 이러한 모델들은 인코더와 디코더를 포함하고 있음. 이 모델들 중 최고 성능을 보이는 모델들은 주로 인코더와 디코더를 연결하는 어텐션 메커니즘을 사용. 이 논문에서는 어텐션 메커니즘만을 사용하는 새로운 간단한 네트워크 아키텍처인 "Transformer"를 제안.
> 
- sequence transduction은 입력 시퀀스를 받아서 다른 시퀀스로 변환하는 작업
- RNN과 CNN을 완전히 배제하고 있으며, 두 가지 기계 번역 과제에서 이 모델이 더 높은 품질을 제공하면서도 병렬화가 더 잘 되고 훈련 시간이 크게 단축된다는 것을 보여줌
- WMT 2014 영어-독일어 번역 작업에서 28.4 BLEU 점수를 기록하여 기존 최고 결과보다 2 BLEU 이상 개선된 성과를 보였고, 영어-프랑스어 번역 작업에서도 41.0의 BLEU 점수를 달성하여 단일 모델 상태에서 최고 기록을 세움

# Introduction

> 시퀀스 모델링과 변환 문제에서는 RNN, LSTM, 그리고 GRU가 대표적인 접근법. 이러한 모델들은 언어 모델링 및 기계 번역과 같은 문제에 특히 효과적임. 하지만, RNN 기반 모델들은 이전 hidden state $h_{t-1}$을 통해 hidden state $h_t$를 생성함. 각 위치에 대한 계산은 이전 위치의 상태와 현재 위치의 입력에 따라 결정되며 이러한 순차적인 계산 방식은 특히 긴 시퀀스의 경우에는 병렬화가 어려움.
> 
- Attentioin 메커니즘은 input 또는 output sequence 내의 거리와 관계없이 종속성을 모델링할 수 있게 해주며, 여러 시퀀스 모델링 및 변환 과제에서 중요한 부분이 되었습니다.
- 전적으로 Attention 메커니즘에 의존하여 입력 및 출력 간의 전역 종속성을 생성한다는 점을 강조합니다.

# Model Architecture

> Transformer는 Encoder-Decoder 구조를 따릅니다. Encoder는 심볼 표현의 입력 시퀀스 $(x_1, ..., x_n)$ 을 연속적인 표현의 시퀀스 $(z_1, ..., z_n)$ 으로 매핑하며, $z$가 주어지면 Decoder는 한 번에 한 요소씩 출력 시퀀스를 생성함. 각 단계에서 모델은 Auto-Regressive 하며, 이전에 생성된 심볼을 다음 생성 시 추가 입력으로 사용함. Transformer는 인코더와 디코더 모두에 스택된 Self-Attention Point-Wise Fully Connected Layer를 사용함.
> 

## 3.1 Encoder and Decoder Stacks

**Encoder**

- N = 6 레이어로 구성된 스택으로 이루어져 있음.
- 각 레이어는 두 개의 서브 레이어로 구성되어 있음
    - 첫 번째는 Multi-Head Self-Attention 메커니즘이고
    - 두 번째는 간단한 Position-Wise Fully Connected Feed-Forward Network
- 각 서브 레이어 주위에는 residual connection이 있고, Layer Normalization이 이어집니다.

**Decoder**

- N = 6 레이어로 구성된 스택으로 이루어져 있음
    - Encoder에 있는 두 개의 레이어 외에 인코더의 출력에 대해 Multi Head Attention을 수행하는 레이어가 존재함.
    - 추가적으로 Encoder 출력에 대한 멀티 헤드 어텐션 서브 레이어가 있음.
    - Decoder의 Self-Attention 서브 레이어에서는 각 위치가 이후 위치에 접근하지 못하도록 마스킹을 적용하여, 각 위치 i에 대한 예측이 이전 위치에만 의존하도록 함.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/a2326369-b168-49fd-bcd4-2e5af7bde3cd/Untitled.png)

## **3.2 Attention**

- Attention은 Query와 Key - Value 쌍을 출력에 매핑하는 것.
- Query, Key, Value, Output은 모두 벡터.
- 출력은 Value의 Weighted Sum으로 계산되며, 각 Value에 할당된 가중치는 Query와 해당 Key의 Compatibility Function에 의해 계산됨.

### **3.2.1 Scaled Dot-Product Attention**

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/3b940c33-c194-4ad6-afd8-9aa414cf8c7a/Untitled.png)

- 입력은 $d_v$ 차원과 $d_k$ 차원의 Query, Key로 구성
- 모든 Key로 Query의 Dot-Product를 계산하고 각각을 값으로 나눈 다음 softmax 함수를 적용하여 $d_k$에 대한 가중치를 구함.

$$
Attention(Q, K, V) = softmax({QK^T\over \sqrt{d_k}})
$$

- $d_k$가 작으면 additive attention과 scaled dot-product의 차이가 작음.
- $d_k$가 크면 Dot-Product의 크기가 커져 소프트맥스 함수가 매우 작은 기울기를 갖는 영역으로 밀려나는 것으로 추정됨. 이를 막기 위해 스케일에 $1 \over \sqrt{d_k}$ 곱해줌

### **3.2.2 Multi-Head Attention**

- $d_{model}$ 차원의 Key, Value, Query를 가지는 하나의 Attention을 사용하는 것 대신 학습된 서로 다른 linear projection을 통해 Query, Key, Value 값을 각각 $d_q, d_k, d_v$ 차원으로 linear projection 하는 것이 유리함.
- 이렇게 projection된 Query, Key, Value의 각 버전에 대해 Attention 함수를 병렬로 수행하여 $d_v$차원 출력 값을 산출함(그림 2)
- Multi-Head Attention은 모델이 서로 다른 위치에서 서로 다른 표현 하위 공간의 정보에 공동으로 주의를 기울일 수 있도록 함.

$$
MultiHead(Q, K, V) = Concat(head_1, ...head_h)W^O \\ where \ head_i = Attention(QW_{i}^Q, KW_{i}^K, VW_{i}^V)
$$

- 여기서는 8개의 Attention Layer 또는 Head를 사용한다.
- $d_k = d_v = d_{model} / h = 64$
- 각 Head의 크기가 줄어들었기 때문에 총 계산비용은 전체차원을 가진 Single Head Attention 와 비슷.

### **3.2.3 Applications of Attention in our Model**

> 트랜스포머는 세 가지 방식으로 Multi-Head Attention을 사용함.
> 

**Encoder-Decoder-Attention Layer**

- Encoder-Decoder-Attention Layer에서 메모리 Key와 Value는 Encoder의 Output임.
- 이를 통해 Decoder의 모든 위치가 입력 시퀀스의 모든 위치에 대해 주의를 기울일 수 있음

**Self-Attention Layer(Encoder)**

- 모든 Key, Value, Query가 인코더의 이전 Layer의 아웃풋 에서 나옴
- 인코더의 각 위치는 인코더의 이전 레이어에 있는 모든 위치를 확인할 수 있음

**Self-Attention Layer(Decoder)**

- Auto-Regressive 속성을 유지하려면 Deocder에서 왼쪽으로 정보가 흐르는 것을 방지해야 함.
- 이를 소프트맥스 입력에서 잘못된 연결 에 해당하는 모든 값을 마스킹 함으로써 Scaled-Dot Product Attention 내부에서 구현(그림 2)

## **3.3 Position-wise Feed-Forward Networks**

- 인코더와 디코더의 각 레이어에는 주의 하위레이어 외에도 Fully Connected Feed-Forward Network가 존재
- 두 개의 선형 변환과 ReLU를 사용.

$$
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
$$

## **3.4 Embeddings and Softmax**

- 학습된 임베딩을 사용해 입력 토큰과 출력 토큰을 차원 $d_{model}$의 벡터로 변환.
- 디코더의 출력은 선형 변환과 소프트맥스 함수로 다음 예측 토큰을 생성
- 두 임베딩 레이어와 소프트맥스 선형 변환 사이에 동일한 가중치 행렬을 공유. 임베딩 레이어에서는 이러한 가중치에 $d_{model}$을 곱합니다.
- 입력 토큰 및 출력 토큰은 512차원의 벡터로 임베딩됨.

## **3.5 Positional Encoding**

- Transforemr는 재귀와 컨볼루션이 없기 때문에, 모델이 시퀀스의 순서를 활용하기 위해서는 시퀀스 내 토큰의 상대적 또는 절대적 위치에 대한 정보가 필요함.
- 이를 위해 인코더와 디코더 스택의 하단에 있는 입력 임베딩에 Positional Encoding을 추가
- Positional Encoding은 임베딩과 동일한 차원 $d_{model}$을 가지므로 둘을 합산할 수 있음.
- 사인과 코사인 함수를 기반으로 하며, 이는 모델이 상대적 위치를 학습하는 데 유리함.

$$
PE_{(pos,2i)} = sin(pos/10000^{2i/d_{model}}) \\ PE_{(pos, 2i+1)} = cos(pos/10000^{2i/d_{model}})
$$

- $pos$는 위치이고, $i$는 차원임. 위치 인코딩의 각 차원은 정현파에 해당
- 이 함수는 고정된 오프셋 $k$에 대해 $PE_{pos+k}$는 $PE_{pos}$의 linear function으로 표현되기 때문에 상대적인 위치를 잘 학습할 것이라는 가설
- 학습된 Positional Embedding을 사용해 실험한 결과 두 버전이 거의 동일한 결과를 생성함.
- 정현파 버전을 선택한 이유는 모델이 훈련 중 발생하는 것 보다 더 긴 시퀀스 길이로 외삽할 수 있기 때문

# Results

> Transformer 모델의 성능은 WMT 2014 영어-독일어 번역 작업과 WMT 2014 영어-프랑스어 번역 작업에서 평가되었습니다. 큰 Transformer 모델(Transformer (big))은 영어-독일어 번역 작업에서 BLEU 점수 28.4를 기록해 기존에 보고된 최고 모델들을 2.0 이상 능가하며 새로운 최고 기록을 세웠습니다. 이 모델은 8개의 P100 GPU를 사용하여 3.5일 동안 훈련되었습니다. 기본 모델인 Transformer (base) 역시 기존에 발표된 모든 모델과 앙상블을 능가하며, 다른 경쟁 모델에 비해 훈련 비용이 훨씬 낮았습니다.
> 
- 영어-프랑스어 번역 작업에서는 Transformer (big) 모델이 BLEU 점수 41.0을 기록하여 이전에 발표된 단일 모델을 모두 능가했습니다.
- 이 성과는 이전 최고 기록 모델의 1/4 미만의 훈련 비용으로 달성되었습니다. 이 큰 모델은 드롭아웃 비율 0.3을 사용했습니다.
- Transformer 모델들은 개발 셋에서 선택한 하이퍼파라미터를 사용해 평가되었으며, 최종 출력 길이는 입력 길이 + 50으로 설정되어 조기 종료가 가능하게 되었습니다.

# Discussion

이 논문에서는 자기 어텐션 레이어를 순환 신경망(RNN) 및 합성곱 레이어와 비교하고 있습니다. 자기 어텐션 레이어는 상수 수의 순차적으로 실행된 연산으로 모든 위치를 연결하는 반면, 순환 레이어는 O(n) 순차적 연산이 필요합니다. 계산 복잡성 측면에서 자기 어텐션 레이어는 시퀀스 길이 n이 표현 차원 d보다 작을 때 순환 레이어보다 빠릅니다. 이것은 언어 모델링에서 사용하는 시퀀스의 대표적 특징을 고려할 때 자주 발생합니다.

자기 어텐션을 활용하면 더 많은 병렬화가 가능하고, 장거리 종속성에 대한 경로 길이도 짧게 유지됩니다. 따라서 자기 어텐션은 장거리 종속성을 학습하는 데 유리합니다. 반면, 단일 컨볼루션 레이어는 모든 위치 쌍을 연결하지 않으며, 이 때문에 여러 컨볼루션 레이어를 스택해야만 모든 위치 쌍을 연결할 수 있습니다.

자기 어텐션의 또 다른 이점은 모델의 해석 가능성이 향상될 수 있다는 점입니다. 논문에서는 자기 어텐션을 활용한 다양한 모델의 어텐션 분포를 분석하며, 개별 어텐션 헤드가 각기 다른 작업을 수행하고 구문 및 의미 구조와 관련된 행동을 보이는 것을 확인했습니다.

다음 섹션을 요약하고 싶으시면, 섹션 번호를 알려주세요.

# Conclusion

이 논문은 주목의 개념만을 활용하는 Transformer라는 새로운 시퀀스 변환 모델을 소개합니다. Transformer는 인코더-디코더 구조에서 자주 사용되는 순환 레이어를 멀티 헤드 자기 어텐션으로 대체했습니다. 번역 작업에 있어서 Transformer는 순환 레이어 또는 합성곱 레이어 기반 아키텍처보다 훈련 속도가 상당히 빠르며, 새로운 최고 수준의 번역 품질을 달성했습니다.

Transformer는 텍스트 이외의 입력 및 출력 양식을 포함하는 문제에도 적용할 수 있는 가능성을 가지고 있으며, 큰 입력 및 출력을 효율적으로 처리하기 위해 국지적, 제한된 어텐션 메커니즘을 탐구할 계획입니다. 또한, 생성 과정을 덜 순차적으로 만드는 것도 연구 목표 중 하나입니다.