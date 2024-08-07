# 3 LINE SUMMARY

# Abstract

> Transformer 모델은 긴 시퀀스에서 느리고 메모리를 많이 소모하는데, Self Attention 메커니즘의 시간 및 메모리 복잡도가 시퀀스 길이에 대해 제곱(quadratic)이기 때문. 기존의 근사적 주의 방법들은 계산 복잡도를 줄이기 위해 모델 품질을 희생하는 경향이 있지만, 실제 실행 시간 속도를 개선하지 못하는 경우가 많음. 이러한 문제를 해결하기 위해 Attention 알고리즘을 IO-aware 하게 만들어 Fast와 Slow 메모리 수준 간의 읽기/쓰기를 고려해야 함
> 
- 우리는 GPU 메모리 간의 읽기 및 쓰기 접근을 줄이기 위해 타일링(tiling)을 사용하는 IO-aware 정확 Attention 알고리즘인 FlashAttention을 제안.
- FlashAttention은 표준 주의보다 적은 HBM(고대역폭 메모리) 접근을 필요로 하며, GPU 온칩 SRAM 크기의 범위에 대해 최적. 또한, Block-Sparse Attention으로 확장하여 기존의 모든 근사적 주의 방법보다 빠른 근사적 주의 알고리즘을 제공.
- 실험 결과, FlashAttention은 BERT-large 모델에서 기존의 MLPerf 1.1 훈련 속도 기록보다 15% 빠른 종단간 벽 시계 시간(wall-clock speedup)을 기록했으며, GPT-2 모델에서는 3배, 장거리 아레나(long-range arena)에서는 2.4배 빠른 속도를 보였음.
- FlashAttention과 Block-Sparse FlashAttention은 더 긴 컨텍스트를 가능하게 하여 모델 품질을 향상시켰으며, 최초로 Path-X 및 Path-256 과제에서 확률보다 높은 성능을 달성한 Transformer 모델을 구현.

# Introduction

> Transformer 모델은 자연어 처리와 이미지 분류 등의 다양한 애플리케이션에서 널리 사용되는 구조. 최근에는 이러한 모델들이 더 커지고 깊어졌지만, 여전히 더 `긴 문맥을 다루는 데 어려움을 겪고 있음.` 이는 Transformer의 핵심인 `self-attention 모듈의 시간 및 메모리 복잡도가 시퀀스 길이에 대해 제곱으로 증가하기 때문.`
> 
- 여러 근사적 attention 방법들이 계산 및 메모리 요구 사항을 줄이기 위해 제안되었지만, 대부분 실제 실행 시간 속도 향상을 달성하지 못하고 있음.
    - 이는 주로 FLOP(부동 소수점 연산) 감소에 초점을 맞추고 메모리 접근(IO)에서 발생하는 오버헤드를 간과하기 때문.
- 이 논문에서는 attention 알고리즘을 IO-aware하게 만드는 것이 중요하다고 주장함.
- `GPU의 고속 메모리(SRAM)와 상대적으로 느린 고대역폭 메모리(HBM)` 간의 읽기 및 쓰기를 줄이는 타일링 기법을 사용하는 FlashAttention이라는 IO-aware attention 알고리즘을 제안.
- FlashAttention은 기존의 attention 방법보다 더 적은 HBM 접근을 필요로 하며, 다양한 SRAM 크기에 대해 최적화되어 있음
- 또한, FlashAttention을 block-sparse attention으로 확장하여, 기존의 모든 근사적 attention 방법보다 더 빠른 근사 attention 알고리즘을 제공.
- FlashAttention은 BERT-large와 GPT-2 모델에서 기존 기록보다 빠른 훈련 속도를 보여주었으며, 더 긴 문맥을 처리할 수 있게 함으로써 모델의 품질을 향상시켰음.
- 이로써 최초로 Path-X 및 Path-256 과제에서 확률보다 높은 성능을 달성한 Transformer 모델을 구현할 수 있었음.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/47dc6e8b-1e11-47ad-bc4a-dbde39001eb5/Untitled.png)

# Background

### 2.1 Hardware Performance

> GPU에서의 성능 특성에 대해 논의. GPU 메모리 계층 구조는 여러 형태의 메모리로 구성되며, 크기가 작을수록 빠릅니다. 예를 들어, A100 GPU는 40-80GB의 고대역폭 메모리(HBM)와 각 스트리밍 멀티프로세서 당 192KB의 온칩 SRAM을 가지고 있음. 온칩 SRAM은 HBM보다 훨씬 빠르지만 크기는 훨씬 작음.
> 

**GPU 메모리 계층 구조**

- HBM: 1.5-2.0TB/s의 대역폭, 40-80GB 용량.
- 온칩 SRAM: 19TB/s의 대역폭, 192KB 용량 (각 스트리밍 멀티프로세서당).

**실행 모델**

- GPU는 많은 스레드를 사용하여 커널이라는 연산을 수행.
- 각 커널은 HBM에서 입력을 읽어와서 레지스터와 SRAM에 저장한 후 계산을 수행하고, 다시 HBM에 출력을 기록.

**성능 특성**

- 연산은 계산 집약적(compute-bound) 또는 메모리 집약적(memory-bound)으로 분류될 수 있음.
- 계산 집약적
    - 연산 시간이 주로 산술 연산의 수에 의해 결정
    - ex) 큰 내적 차원을 가진 행렬 곱셈.
- 메모리 집약적
    - 연산 시간이 주로 메모리 접근의 수에 의해 결정
    - ex) 활성화 함수, 드롭아웃, 소프트맥스, 배치 정규화 등.

**커널 퓨전**

- 메모리 집약적 연산을 가속화하는 가장 일반적인 방법.
- `여러 연산이 동일한 입력에 적용될 때, 입력을 한 번만 HBM에서 읽어오는 방법.`
    - 예를 들어, 입력 데이터 'A'에 대해서 연산 'A1'과 'A2'가 연속적으로 필요할 때, 전통적인 방식에서는 'A'를 HBM에서 불러와서 'A1'을 수행하고, 다시 'A'를 불러와서 'A2'를 수행.
    - 반면 커널 융합을 사용하면 'A'를 한 번만 불러와서 'A1'과 'A2'를 연속적으로 즉시 수행.
- 많은 요소별 연산을 자동으로 융합할 수 있음
- `하지만 Forward Pass를 진행하며 얻은 파라미터들 Back Propagation에서 다시 사용해야 함.`
    - 이 값들은 HBM에 저장되어야 하고, 여러 번의 메모리 접근이 필요해 짐.
    - 이로 인해 커널 퓨전의 최적화 효과가 줄어듦

### 2.2 Standard Attention Implementation

- 기본적인 attention 구현에서는 입력 시퀀스 $K, V \in \mathbb{R}^{N \times d}$를 사용하여 attention 출력 $O \in \mathbb{R}^{N \times d}$를 계산. 여기서 $N$은 시퀀스 길이, $d$는 헤드 차원

$$
S = QK^T \in \mathbb{R}^{N \times N} \quad P = \text{softmax}(S) \in \mathbb{R}^{N \times N} \quad O = PV \in \mathbb{R}^{N \times d}
$$

- softmax는 행 단위로 적용됨. 표준 attention 구현에서는 행렬 $S$와 $P$를 HBM(고대역폭 메모리)에 저장하며, 이는 $O(N^2)$ 메모리가 필요함.

- **`Algorithm 0`**
    1. HBM에서 $Q, K$를 블록 단위로 로드하고 $S=QK^T$를 계산하여 HBM에 Write.
        - 블록 단위란, 큰 데이터 세트를 작은 조각(블록)으로 나누어 처리하는 방법을 의미함.
        - 입력 데이터 매트릭스 $Q, K, V$는 매우 크므로 한 번에 GPU의 SRAM에 로드하기 어려움
        - 이 매트릭스를 여러 작은 블록으로 나누어 한 번에 하나의 블록씩 처리.
    2. HBM에서 S를 읽어와 $P=\text{softmax}(S)$를 계산하고, P를 HBM에 Write.
    3. HBM에서 $P$와 $V$를 블록 단위로 로드하고 $O=PV$를 계산하여 HBM에 Write.
    4. 최종적으로 $O$를 Return.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/ff9b7a74-a464-458f-af24-ddd532198b79/Untitled.png)

# FlashAttention: Algorithm, Analysis, and Extensions

> `더 적은 수의 HBM 읽기/쓰기로 Backward Pass를 위한 대용량 중간 행렬을 저장하지 않고도 정확한 Attention을 계산하는 방법을 제시.` 이를 통해 메모리 효율성과 월 클럭 시간 모두에서 더 빠른 주의 알고리즘을 얻을 수 있음. IO 복잡성을 분석해 본 결과, 이 방법은 표준 주의에 비해 훨씬 적은 수의 HBM 액세스를 필요로 하는 것으로 나타남. 또한  Flash Attention이 블록 희소 주의 처리를 위해 확장되어 유용한 프리미티브 역할을 할 수 있음을 보여줌. 여기서는 설명의 편의를 위해 Forward Pass에 초점을 맞추고, 부록 B에는 Backward Pass에 대한 자세한 내용이 포함되어 있음.
> 

### 3.1 An Efficient Attention Algorithm With Tiling and Recomputation

- FlashAttention은 시퀀스 길이에 대한 HBM(고대역폭 메모리) 접근 횟수를 줄여 효율적으로 정확한 attention을 계산하는 알고리즘.
- 이를 위해 입력 $Q, K, V \in \mathbb{R}^{N \times d}$를 블록으로 나누어, 느린 HBM에서 빠른 SRAM(온칩 메모리)으로 로드한 후, 각 블록에 대해 attention 출력을 계산. 각 블록의 출력을 올바르게 정규화하여 최종 결과를 얻음.
- **타일링(Tiling)**
    - Attention을 block 단위로 계산함.
    - Softmax는 K의 컬럼을 결합함. 큰 Softmax를 분해해서 수치적 안전성을 위해서 Softmax 벡터를 다음과 같이 계산함.
    - 수치적 안정성을 위해 벡터의 최대값을 빼고 지수 함수로 변환한 후, 정규화.
    - 벡터 $x \in \mathbb{R}^B$에 대해
    
    $$
    \begin{align}
    𝑚(𝑥) &:= \text{max}_i x_i \\
    f(x) &:= [e^{x_1-m(x)}, \cdots, e^{x_B-m(x)}], \\
    l(x) &:= \Sigma_i f(x)_i, \\ \text{softmax}(x) &:= {f(x) \over l(x)} 
    \end{align}
    $$
    
    - 두 벡터 $x^{(1)}, x^{(2)} \in \mathbb{R}^B$ 에 대해:
        
        $$
        \begin{align}
        m(x) &= m([x^{(1)} \quad x^{(2)}]) = \text{max}(m(x^{(1)}), m(x^{(2)}))\\
        f(x) &= [e^{m(x^{(1)}) - m(x)}f(x^{(1)}) \quad e^{m(x^{(2)})-m(x)}f(x^{(2)})] \\
        l(x) &= l([x^{(1)} \quad x^{(2)}]) = e^{m(x^{(1)}) - m(x)}l(x^{(1)}) + e^{m(x^{(2)})-m(x)}l(x^{(2)}) \\
        \text{softmax} &= {f(x) \over l(x)}
        \end{align}
        $$
        
    - 이렇게 통계값을 유지하면 블록 단위로 소프트맥스를 계산 가능.

- **재계산(Recomputation)**
    - 중간 행렬을 저장하지 않고 필요한 경우 다시 계산.
    - 이는 메모리 접근을 최소화하여 효율성을 높임.

- 타일링 덕분에 하나의 CUDA 커널에서 모든 계산을 수행 가능.
- HBM에서 입력 로드 -> on-chip에서 계산 -> HBM으로 결과 쓰기.

- **`Algorithm 1`**
- 행렬 𝑄,𝐾,𝑉가 HBM에 저장되어 있어야 함. 온칩 SRAM의 크기 𝑀.
1. 블록 크기 설정 : $𝐵_𝑐=[{M \over 4d}], 𝐵_𝑟=\text{min}([{M \over 4d}],𝑑)$
2. 초기화
    
    $O = (0)_{N\times d} \in \mathbb{R}^{N \times d} \\ l = (0)_N \in \mathbb{R}^N \\ m = (-\infty)_N \in \mathbb{R}^N \quad \text{in} \quad \text{HBM}$
    
3. $Q$를  $T_r=[{N \over B_r}]$ 개로 분할. 블록 $Q_1, \cdots Q_{T_r}$의 각 사이즈는 $B_r \times d$
$K, V$를 $T_c = [{N \over B_c}]$ 개로 분할. 블록 $K_1, \cdots, K_{T_c} \quad \text{and} \quad V_1, \cdots, V_{T_c}$의 각 사이즈는 
4. $O$를  $T_r=[{N \over B_r}]$ 개로 분할. 블록 $O_1, \cdots, O_{T_r}$의 각 사이즈는 $B_r \times d$
$l$을 $T_r$개로 분할. 블록 $l_1, \cdots. l_{T_r}$의 각 사이즈는 $B_r$
$m$을 $T_r$개로 분할. 블록 $m_1, \cdots m_{T_r}$의 각 사이즈는 $B_r$
5. $\text{for} \quad 1 \le j \le T_r \quad \text{do}$
    
    **$K, V$ 블록 로드**
    
    1. $\text{for} \quad 1 \le i \le T_r \quad \text{do}$
        - HBM에서 On-Chip SRAM으로  $Q_i, O_i, ℓ_i, m_i$ 블록을 로드
        - On Chip에서 $S_{ij} = Q_iK_j^T \in \mathbb{R}^{B_r \times B_c}$ 계산
        - On Chip에서
            - $\tilde{m}_{ij} = \text{rowmax}(S_{ij}) \in \mathbb{R}^{B_r}$
            - $\tilde{P}_{ij} = \exp(S_{ij} - \tilde{m}_{ij}) \in \mathbb{R}^{B_r \times B_c}$
            - $\tilde{\ell}_{ij} = \text{rowsum}(\tilde{P}_{ij}) \in \mathbb{R}^{B_r}$
        - On Chip에서 새로운 ℓ와 m 계산 및 업데이트
            - $𝑚_𝑖^{new}=max(𝑚_𝑖,\tilde{𝑚}_{𝑖𝑗})$
            - $\ell_i^{\text{new}} = e^{m_i - m_i^{\text{new}}} \ell_i + e^{\tilde{m}_{ij} - m_i^{\text{new}}} \tilde{\ell}_{ij}$
        - $O$ 업데이트
            - $O_i = \text{diag}(\ell_i^{\text{new}})^{-1}\left(\text{diag}(\ell_i) e^{m_i - m_i^{\text{new}}} O_i\right) + e^{\tilde{m}_{ij} - m_i^{\text{new}}} \tilde{P}_{ij} V_j$
        - 새로운 ℓ와 m 값을 HBM에 Write

16. Return $O$

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/226283bd-5cbf-4520-a04c-d83bb2c51efc/Untitled.png)

- **Theorem 1**:
    - Algorithm 1의 return은 다음과 같음
    - $𝑂=\text{softmax}(𝑄𝐾^\top)𝑉$
    - 계산 복잡도(FLOPs) : $O(N^2d)$
    - 요구되는 추가 메모리 : $O(N)$

### 3.2 FlashAttention의 IO 복잡도 분석

- FlashAttention는 HBM 접근 횟수를 줄여 성능을 개선함
- **GFLOPs**: FlashAttention는 표준 Attention보다 FLOP 수가 높음.
    - 표준 Attention: 66.6
    - FlashAttention: 75.2
- **HBM R/W (GB)**: FlashAttention는 HBM 접근 횟수가 현저히 적음.
    - 표준 Attention: 40.3 GB
    - FlashAttention: 4.4 GB
- **Runtime (ms)**: FlashAttention는 실행 속도가 훨씬 빠름.
    - 표준 Attention: 41.7 ms
    - FlashAttention: 7.3 ms
- HBM 접근이 주된 요인이라는 점이 중요.

- **Theorem 2**
    - 𝑁이 Sequence Length이고, 𝑑가 Head Dimension이고, $d ≤ M ≤ Nd$ 인 SRAM 사이즈 $M$을 가질 때, 스탠다드 어텐션은(알고리즘0) $\Theta(Nd+N^2)$ HBM access를 요구하는 반면 Flash Attention(알고리즘1)은 $\Theta({N^2d^2 \over M})$ HBM acess를 요구함
    - $d^2$은 일반적으로 $M$보다 훨씬 작기 때문에, FlashAttention은 Standard Attention보다 훨씬 적은 HBM 접근을 필요로 함.
    - 이는 더 빠른 실행과 낮은 메모리 사용을 의미.
    - SRAM 사이즈 𝑀이 주어지면, 크기 $Θ(𝑀)$의 $K$와 $V$블록을 로드할 수 있습니다.
    - 모든 $Q$ 블록을 반복하면서 중간 값을 계산하는 방식입니다.
- **하한 증명**:
    - 𝑀 값의 모든 범위에 대해 정확한 Attention을 계산할 때, 이러한 수의 HBM 접근을 비대칭적으로 줄이는 것은 불가능.
    - 이는 시퀀스 길이 𝑁과 헤드 차원 𝑑의 함수로 증명되었습니다.

- **Proposition 3**
    - N이 Sequence Length이고, d가 Head Dimension, M이 SRAM의 사이즈 with $d ≤ M ≤ Nd$ 일 때, 모든 $M$에 대해서 $o(N^2d^2M^{-1})$ HBM access를 가지는 정확한 어텐션을 계산하는 알고리즘은 존재하지 않는다.
    - HBM 접근 수가 주된 실행 시간 결정 요인임을 확인했다.
    - **Fig. 2 (left)**
        - FlashAttention은 표준 Attention보다 더 많은 FLOP를 사용하지만, HBM 접근 수가 훨씬 적어 더 빠른 런타임을 가진다. GPT-2 Medium의 Forward + Backward 실행 시간.
    - **Fig. 2 (middle)**
        - FlashAttention의 블록 크기 $B_c$를 변경하여 다양한 HBM 접근 양을 생성하고, 전방 패스의 런타임을 측정했다.
    - **Fig. 2 (right)**
        - Block-Sparse FlashAttention의 실행 시간.
    - 블록 크기가 커질수록, HBM 접근 수가 줄어들어 런타임이 감소한다. (입력에 대한 패스 수 감소)
    - 블록 크기가 충분히 커지면, 런타임은 다른 요인 (예: 산술 연산)으로 인해 병목 현상이 발생한다.
    - 너무 큰 블록 크기는 작은 SRAM 크기에 맞지 않음.
    - $Ω(N^2d^2M^{−1})=Ω(Nd)$
        - 어떤 알고리즘도 최소 $Ω(N^2d^2M^{−1})$ HBM 접근이 필요
        - 이를 통해 $Ω(Nd)$ HBM 접근이 본질적으로 필요하다는 것을 증명

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/19fd67b1-0aa7-45f9-9b6d-a8b441f60733/cdd7e20f-0cc9-4310-95a5-30c0bae8baa6/Untitled.png)

### 3.3 블록 희소 FlashAttention 확장

- FlashAttention을 블록 희소 attention으로 확장하여, 더 적은 HBM 접근으로 근사 attention을 계산할 수 있습니다. 이 방법은 블록 형태의 마스크 행렬을 사용하여 불필요한 블록을 건너뜁니다. 이로 인해 IO 복잡도는 희소성 비율에 비례하여 개선됩니다.
- FlashAttention과 블록 희소 FlashAttention은 긴 시퀀스를 처리하는 Transformer 모델의 성능을 크게 향상시키며, 더 빠른 훈련 시간과 더 나은 모델 품질을 제공합니다 .

- **입력: $𝑄,𝐾,𝑉 \in \mathbb{R}^{𝑁×𝑑}$**
- **마스크 행렬: $\tilde{M} \in 0,1^{𝑁×𝑁}$**

1. **행렬 곱셈: $𝑆=𝑄𝐾^𝑇 \in \mathbb{𝑅}^{𝑁×𝑁}$o**
2. **마스킹 적용: 

$𝑃=\text{softmax}(𝑆⊙𝑀~) \in \mathbb{𝑅}^{𝑁×𝑁}$**

3. 여기서 는 요소별 곱셈.

$$
\tilde{M} \odot S_{kl} = S_{kl} \quad if \quad \tilde{M}_{kl} = 1, −∞ \quad if \quad \tilde{M}_{kl} = 0
$$

1. **출력 계산:$𝑂=𝑃𝑉 \in \mathbb{R}^{𝑁×𝑑}$**

### 블록 마스킹 요구조건

- 𝑀~은 블록 형태를 가져야 한다.
- 블록 크기 와 에 대해, \( \tilde{M}*{k, l} = M*{ij} \) with ,  for some .
    
    𝐵𝑟
    
    𝐵𝑐
    
    𝑖=⌊𝑘𝐵𝑟⌋
    
    𝑗=⌊𝑙𝐵𝑐⌋
    
    𝑀∈0,1𝑁𝐵𝑟×𝑁𝐵𝑐
    

### IO 복잡도

- **블록 희소 FlashAttention의 HBM 접근 횟수:**
    
    Θ(𝑁𝑑+𝑁2𝑑2𝑀𝑠)
    
    - 𝑁: 시퀀스 길이
    - 𝑑: 헤드 차원
    - 𝑀: SRAM 크기 (  )
        
        𝑑≤𝑀≤𝑁𝑑
        
    - 𝑠: 블록 희소성의 비율

### 성능 향상

- **블록 희소성 적용 시 직접적인 개선점:**
    - 블록 희소성을 적용하면 에 비례하여 IO 복잡도가 직접 개선됨.
        
        𝑠
        
    - 큰 시퀀스 길이 에 대해, 는 종종  또는 로 설정됨.
        
        𝑁
        
        𝑠
        
        𝑁−1/2
        
        (log⁡𝑁)−1
        
    - 이는 IO 복잡도를  또는 로 만듦.
        
        Θ(𝑁𝑁)
        
        Θ(𝑁log⁡𝑁)
        

### 실험 결과

- **LRA 벤치마크:** 블록 희소 FlashAttention이 2.8배 속도 향상을 달성하면서 성능 저하 없이 표준 Attention과 동등한 성능을 보임.

# Experiments

### 4.1 Faster Models with FlashAttention

FlashAttention을 이용하여 Transformer 모델의 훈련 시간을 크게 단축시켰습니다.

- **BERT**: FlashAttention은 BERT 모델 훈련 속도를 15% 개선하여, MLPerf 1.1에서 기록된 Nvidia의 속도를 초과했습니다.
- **GPT-2**: FlashAttention은 HuggingFace와 Megatron-LM보다 GPT-2 훈련 속도를 각각 최대 3배, 1.8배 빠르게 했습니다.
- **Long-Range Arena (LRA)**: FlashAttention은 LRA 벤치마크에서 2.4배 빠른 속도를 보여주었고, 블록 희소 FlashAttention은 모든 기존 근사 attention 방법들보다 더 빠른 성능을 보였습니다.

### 4.2 Better Models with Longer Sequences

긴 시퀀스를 처리할 수 있는 모델을 통해 모델 성능을 향상시켰습니다.

- **GPT-2 with Long Context**: FlashAttention은 GPT-2의 컨텍스트 길이를 4배 늘리면서도 여전히 Megatron-LM보다 30% 빠른 속도로 실행되었으며, 더 낮은 퍼플렉서티(더 좋은 성능)를 달성했습니다.
- **Long Document Classification**: 긴 시퀀스 처리가 필요한 문서 분류 작업에서 FlashAttention을 사용하여 MIMIC-III와 ECtHR 데이터셋에서 성능을 향상시켰습니다.

### 4.3 Benchmarking Attention

- FlashAttention과 블록 희소 FlashAttention의 실행 시간과 메모리 사용량을 시퀀스 길이에 따라 측정했습니다.
- **Runtime and Memory Usage**: FlashAttention은 시퀀스 길이에 비례하여 메모리 사용량이 선형적으로 증가하며, 일반적인 시퀀스 길이(최대 2K)에서 표준 attention보다 최대 3배 빠른 속도를 보였습니다.
- **Block-Sparse FlashAttention**: 희소성을 높임에 따라 실행 시간이 비례적으로 개선되었으며, 기존의 모든 근사 attention 방법들보다 더 빠른 성능을 보였습니다.
- 이 실험 결과들은 FlashAttention이 Transformer 모델의 훈련 시간과 성능을 크게 향상시킬 수 있음을 보여줍니다(FlashAttention _ Fast a…).

# Limitations and Future Directions

- **CUDA로 컴파일**: 현재 IO-aware attention 구현은 각 새로운 attention 구현마다 새로운 CUDA 커널을 작성해야 합니다. 이는 PyTorch보다 낮은 수준의 언어로 알고리즘을 작성해야 하므로, 상당한 엔지니어링 노력이 필요합니다. 또한, 이러한 구현은 GPU 아키텍처 간에 이전할 수 없습니다. 따라서, Halide 이미지 처리와 유사하게, 고수준 언어(예: PyTorch)로 attention 알고리즘을 작성하고 이를 CUDA로 컴파일할 수 있는 방법이 필요합니다.
- **IO-인식 딥러닝**: IO-aware 접근 방식은 attention을 넘어 확장될 수 있습니다. attention은 Transformer에서 가장 메모리 집약적인 연산이지만, 딥 네트워크의 모든 레이어가 GPU HBM을 사용합니다. 우리의 연구는 추가 모듈의 IO-aware 구현을 촉진할 수 있기를 희망합니다.
- **멀티-GPU IO-aware 방법**: 우리의 IO-aware attention 구현은 단일 GPU에서 attention을 계산할 때 상수 내에서 최적입니다. 그러나 attention 계산은 여러 GPU에 걸쳐 병렬화될 수 있습니다. 여러 GPU를 사용하면 GPU 간 데이터 전송을 고려한 추가 IO 분석이 필요합니다. 우리는 이 방향의 미래 연구를 기대합니다.

- **다중 GPU 사용**: 대규모 언어 모델은 수백 또는 수천 개의 GPU에서 훈련됩니다. 일반적으로 동일한 노드의 4-8개의 GPU 간에 attention 계산을 분할합니다. 이는 메모리 계층의 비대칭성을 고려하여 매우 긴 시퀀스의 attention 계산에서 협력할 수 있습니다.
- **희소 MLP 레이어**: 일반적인 밀집 MLP 레이어는 계산 집약적이지만 메모리 집약적이지 않습니다. 희소 가중치 행렬을 가진 MLP 레이어를 사용하면 효율성을 개선할 수 있습니다. 그러나 많은 희소 MLP 레이어는 메모리 집약적이 되며, 속도 향상이 희소성에 비례하지 않는 경우가 많습니다. IO-인식 구현은 이러한 문제를 완화하고 희소성의 이점을 실현할 수 있을 것으로 기대합니다.
- **커널 머신 러닝**: FlashAttention 접근 방식은 $N \times N$ attention 행렬이 저차원 행렬 $QK^\top$의 함수라는 사실을 이용합니다. 이로 인해 필요한 attention 행렬 블록을 다시 계산하여 HBM 접근을 크게 줄일 수 있습니다. 유사한 시나리오는 커널 머신 러닝에서도 발생합니다. KeOps 라이브러리는 메모리 읽기/쓰기를 줄여 커널 연산 속도를 높인 성공적인 예입니다. 우리는 FLOP를 줄이는 것뿐만 아니라 IO를 줄이는 것에 초점을 맞춘 커널 방법이 개발되기를 기대합니다.