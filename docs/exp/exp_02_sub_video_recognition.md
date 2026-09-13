# Exp-02-sub. CLEVRER ResNet-34 비디오 인식 실험

Wan2.1 픽셀 생성(Exp-02) 전에 CLEVRER 클립에서 객체·궤적·충돌을 읽는 인식기를 학습한다. 본 실험은 확률적 상태 생성이나 제약 샘플링이 아니라, **사전학습된 ResNet-34의 공간 특징과 학습 가능한 시간·객체 헤드로 상태 \(S\)를 직접 예측하는 지도학습 실험**이다.

Exp-02의 방법 비교는 FlowMatch baseline과 Y-Flow를 중심으로 둔다. HardFlow, SafeFlow, UniConFlow, GuideFlow는 제외한다. 이 문서는 해당 방법들의 속도장 비교를 더 이상 계획하지 않는다. Wan VAE/DiT도 이 인식기의 입력 인코더로 사용하지 않는다.

주요 목표는 다음 세 가지다.

1. CLEVRER 비디오에서 객체 속성·개수·가시성을 인식한다.
2. 객체별 world 위치와 속도를 프레임마다 추정하고, 클립 안의 충돌 이벤트를 찾는다.
3. 예측된 상태를 Exp-02의 상태 스키마와 물리 타당성 평가에 연결한다.

---

## 1. 과제 정의와 예측 상태

입력은 한 CLEVRER 클립 \(V\), 출력은 최대 \(K=6\)개 객체에 대한 고정 슬롯 상태 \(\hat S\)다. 상태의 좌표계는 CLEVRER 시뮬레이터 world 좌표이며, 이미지 픽셀 좌표를 world 좌표로 간주하지 않는다.

클립 길이는 \(T=33\) 프레임이다. 속성은 클립 단위, 위치·속도·가시성은 프레임 단위, 충돌은 프레임 및 비순서 슬롯 쌍 단위로 예측한다.

| 출력 | 형상 | 표현 |
| :--- | :--- | :--- |
| 색상 | \([K,8]\) | 8개 CLEVRER 색 클래스 로짓 |
| 재질 | \([K,2]\) | metal / rubber 로짓 |
| 모양 | \([K,3]\) | cube / cylinder / sphere 로짓 |
| 객체 존재 | \([K]\) | 클립 안에 해당 객체 슬롯이 사용되는지에 대한 로짓 |
| 가시성 | \([K,T]\) | 객체별 프레임 가시성 로짓 |
| 위치 | \([K,T,3]\) | 시뮬레이터 world 좌표 \((x,y,z)\) |
| 속도 | \([K,T,3]\) | 시뮬레이터 world 속도 \((u_x,u_y,u_z)\) |
| 충돌 | \([T,\binom K2]\) | 각 비순서 객체 쌍의 프레임별 충돌 로짓 |

비활성 슬롯은 객체 존재 타깃 0으로 학습한다. 속성·궤적 타깃은 활성 슬롯에만 적용한다. 충돌 헤드는 \(i<j\) 쌍만 출력하므로 대칭과 자기 충돌 금지는 구조적으로 보장된다.

---

## 2. 비디오 입력과 라벨 생성

### 2.1 입력 텐서

기본 입력 해상도는 현재 실험 설정에 맞춰 \(H=160,\ W=240\)으로 둔다. 원본 CLEVRER 프레임 \(320\times480\)을 종횡비를 유지해 절반 크기로 리사이즈한다. 이 비율에서는 padding이 필요 없다.

- 데이터 로더 출력: \([B,3,T,H,W]=[B,3,33,160,240]\), RGB, 값 범위 \([-1,1]\).
- ResNet 프레임 입력: \([BT,3,H,W]=[33B,3,160,240]\).
- 시간 격자: \(s_t=t/16,\ t=0,\ldots,32\). 원본 25fps 클립에서 각 요청 시각에 가장 가까운 프레임과 같은 frame_id의 simulator annotation을 사용한다.
- frame_indices, 요청 시각, 실제 원본 시각을 함께 보존해 영상과 라벨 정렬을 검사한다.
- 기본 설정에서 valid_region은 전부 유효하다. 다른 입력 해상도를 쓰거나 padding이 생기면 padding 영역을 마스킹한다.

CLEVRER 로더는 영상 텐서를 \([-1,1]\)로 반환한다. ResNet-34 ImageNet 사전학습 입력을 위해 먼저 \([0,1]\)로 변환한 뒤, 채널별 ImageNet mean/std로 정규화한다. 정규화 상수는 모델 가중치와 함께 설정에 기록한다.

### 2.2 상태 타깃

기존 CLEVRER annotation parser를 사용해 색·재질·모양, 가시성, world 위치·속도, 충돌을 \(K=6,T=33\) 상태로 패킹한다. 객체 슬롯 순서는 annotation의 object_id 오름차순으로 만든다. 학습 시 예측 슬롯 순서는 고정되어 있지 않으므로, 손실 계산에서 클립별 Hungarian matching으로 GT 객체와 예측 슬롯을 일대일 대응한다. 프레임마다 따로 matching하지 않는다. 이 전역 매칭이 객체 ID가 시간에 따라 바뀌는 것을 막는다.

world 위치·속도 손실의 정규화 통계는 train split에서만, 활성 객체의 유효 타깃을 사용해 계산한다. 빈 슬롯의 0 패딩이나 보이지 않는 프레임의 채움 값을 통계에 포함하지 않는다. 지표 및 제약 검사를 위해 출력은 원래 단위로 복원한다.

충돌 손실은 두 객체가 해당 프레임에 모두 보이는 쌍을 학습 대상으로 삼는다. 한 객체라도 보이지 않는 충돌 이벤트는 영상에서 관측 불가능할 수 있으므로 학습 손실과 관측 가능 충돌 지표에서 제외하고 개수를 별도로 보고한다.

### 2.3 데이터 분할

- 학습 및 개발 데이터: CLEVRER train scene 0–9999. scene_id를 고정 seed로 분할해 90% 학습, 10% 개발에 사용한다.
- 최종 평가: 기존 고정 eval 100 manifest의 scene 10000–10099.
- 체크포인트와 임계값은 개발 데이터에서만 고른다. eval 100으로 하이퍼파라미터나 임계값을 조정하지 않는다.
- 공식 visual-mask JSON은 본 실험의 입력과 평가 정답으로 사용하지 않는다. 정답은 simulator annotation에서 만든다.

---

## 3. ResNet-34 시각 인코더

### 3.1 사용할 가중치와 절단 지점

Torchvision의 ImageNet-1K 사전학습 ResNet-34 가중치 ResNet34_Weights.IMAGENET1K_V1을 사용한다. 실험 설정에 정확한 가중치 이름을 기록한다.

**유지:** stem(conv1, bn1, relu, maxpool), layer1, layer2, layer3.

**제거:** layer4, 전역 avgpool, ImageNet 분류용 fc.

layer4부터는 32픽셀 간격의 저해상도 특징만 남아 여러 작은 도형의 위치를 구분하는 데 불리하다. 분류용 전역 pooling과 fc도 객체별 공간 위치를 없애므로 사용하지 않는다.

### 3.2 텐서 크기

ResNet은 시간을 접어 각 프레임을 독립 처리하고, 프레임별 특징을 다시 시간축으로 묶는다. 입력 크기 \(160\times240\)에서의 크기는 다음과 같다.

| 지점 | 채널 | 공간 크기 | Reshape 전 형상 |
| :--- | :---: | :---: | :--- |
| 입력 | 3 | \(160\times240\) | \([BT,3,160,240]\) |
| stem + maxpool | 64 | \(40\times60\) | \([BT,64,40,60]\) |
| layer1 | 64 | \(40\times60\) | \([BT,64,40,60]\) |
| layer2 | 128 | \(20\times30\) | \([BT,128,20,30]\) |
| layer3 | 256 | \(10\times15\) | \([BT,256,10,15]\) |

layer2의 공간 세부 정보와 layer3의 큰 문맥을 함께 쓴다. 각각을 256채널로 1×1 투영한 후 layer3를 2배 bilinear upsample하고 layer2 특징과 더한다. 3×3 convolution으로 융합해 프레임당 \(F_t\in\mathbb{R}^{256\times20\times30}\)을 얻는다. 이후 위치 인코딩을 더해 600개 위치 토큰, 각 256차원으로 만든다.

전체 형상:

\[
F\in\mathbb{R}^{B\times T\times 600\times256}.
\]

이 공간 특징을 유지해야 객체 위치와 슬롯별 시각 단서를 헤드가 읽을 수 있다. 프레임 전체 평균 하나로 축약하지 않는다.

### 3.3 학습 방식

첫 학습 단계에서는 ResNet 전체를 고정하고 특징 융합층과 인식 헤드만 학습한다. 개발 성능이 정체되면 layer3만 낮은 learning rate로 풀어 fine-tuning한다. 주 비교는 다음 두 설정이다.

1. ImageNet ResNet-34 고정 + 학습 가능한 융합층·헤드
2. ImageNet ResNet-34의 layer3 미세조정 + 동일한 융합층·헤드

layer4, avgpool, fc는 두 설정 모두 사용하지 않는다. 첫 기준선은 사전학습 특징만으로도 결과를 내야 하므로, 처음부터 전체 backbone을 풀지 않는다.

---

## 4. 시간·객체 인식 헤드

헤드는 프레임별 공간 특징에서 객체별 표현을 뽑고, 각 객체의 표현을 시간축으로 연결한다.

### 4.1 프레임별 객체 슬롯 추출

각 프레임의 600개 공간 토큰을 key/value로 사용하고, \(K=6\)개의 학습 가능한 슬롯 query가 cross-attention한다. 두 층의 슬롯 decoder를 사용한다.

초기 헤드 설정:

- 차원 \(D=256\)
- 슬롯 수 \(K=6\)
- cross-attention decoder 2층
- 각 층 8 attention head, FFN 차원 1024
- 고정 2D sine-cosine 공간 위치 인코딩 + 학습 가능한 33-step 프레임 위치 인코딩
- attention residual 뒤 LayerNorm, FFN은 256 → 1024 → 256, GELU

출력은 프레임별 객체 토큰

\[
O\in\mathbb{R}^{B\times T\times K\times256}.
\]

슬롯 query는 모든 프레임에서 공유한다. 슬롯 번호 자체가 object_id라고 가정하지 않고, 학습 손실의 클립별 전역 matching으로 ID를 정렬한다.

### 4.2 객체별 시간 인코더

각 슬롯의 \(T=33\)개 토큰에 시간 위치 인코딩을 더하고, 같은 슬롯 안에서 시간 정보를 섞는 temporal Transformer에 넣는다. 다른 슬롯의 정보는 충돌 head의 쌍 표현에서 결합한다.

초기 설정:

- temporal Transformer 4층
- 차원 256, attention head 8, FFN 차원 1024
- 각 층은 Pre-LayerNorm self-attention 및 256 → 1024 → 256 GELU FFN
- 입력/출력 형상 모두 \([B,T,K,256]\)

이를 통해 속도 추정과 충돌 분류가 단일 프레임의 외형만이 아니라 시간 변화도 이용한다.

### 4.3 출력 head

시간 인코더 출력 \(O'\)에서 다음을 예측한다.

- **존재:** 객체별 시간 토큰을 learned-query attention pooling한 뒤 MLP → \([B,K]\)
- **속성:** 같은 clip-level pooled token에 각각 독립 MLP → 색상 \([B,K,8]\), 재질 \([B,K,2]\), 모양 \([B,K,3]\)
- **가시성:** 프레임별 token MLP → \([B,K,T]\)
- **위치·속도:** 프레임별 token에서 별도 MLP → 각각 \([B,K,T,3]\)
- **충돌:** 각 \(i<j\) 쌍에 대해 \(O'_i+O'_j,\ |O'_i-O'_j|,\ O'_i\odot O'_j\)와 예측 상대 위치·속도를 결합한 pair MLP → \([B,T,15]\)

충돌은 비순서 쌍 15개만 직접 출력해 \(c_{ij}=c_{ji}\), \(c_{ii}=0\)을 구조적으로 만족시킨다. 존재와 가시성은 별도 출력이다. 클립에 없는 슬롯의 속성·궤적은 지표 및 물리 제약 집계에서 제외한다.

각 출력용 MLP의 기본 형태는 256 → 256 → 출력 차원이며, 은닉층은 GELU와 dropout 0.1을 쓴다. 충돌 pair MLP의 입력 차원은 슬롯 결합 특징 768과 상대 위치·속도 6을 합친 774, 은닉층은 512, 출력은 충돌 로짓 1이다.

---

## 5. 지도학습 손실

모델은 정규분포 노이즈에서 상태를 생성하지 않는다. 예측 \(\hat S=f_\theta(V)\)와 simulator annotation 상태 \(S\) 사이의 직접 지도 손실을 학습한다. 각 항은 유효 슬롯·프레임·쌍 수로 나누어 평균한다.

### 5.1 클립별 슬롯 대응

각 예측 슬롯 \(k\)와 GT 객체 \(j\)의 matching 비용은 다음 항의 가중합이다.

- 객체 존재 점수
- 색·재질·모양 분류 비용
- GT 가시 프레임에서의 위치 거리

비용 행렬에 Hungarian assignment를 한 번 적용해 클립 전체 permutation \(\pi\)를 정한다. unmatched prediction은 빈 슬롯(no-object), unmatched GT는 누락 객체로 계산한다. GT 객체가 클립 전체에서 보이지 않아 위치 항의 유효 프레임이 없으면 그 항은 matching 비용에서 생략한다. matching은 학습의 손실 계산용이며, 평가 지표에서도 같은 속성/궤적 기반 Hungarian 기준을 명시해 사용한다.

초기 matching 비용은

\[
C_{k,j}=\lambda_{\mathrm{obj}}(-\log \sigma(o_k))
+\lambda_{\mathrm{attr}}\sum_{r\in\{a,m,q\}}\mathrm{CE}(\hat y^r_k,y^r_j)
+\lambda_{\mathrm{pos}}\,\mathrm{mean}_{t:\nu_{j,t}=1}\|\hat p_{k,t}-p_{j,t}\|_1
\]

로 둔다. 객체 존재와 속성 항은 clip 단위, 위치 항은 GT 가시 프레임만 사용한다. matching 비용의 계수는 손실 계수와 같은 초기값에서 시작하고 개발 split에서 고정한다.

### 5.2 손실 항

\[
\mathcal L =
\lambda_{\mathrm{obj}}\mathcal L_{\mathrm{obj}}+
\lambda_{\mathrm{attr}}\mathcal L_{\mathrm{attr}}+
\lambda_{\mathrm{vis}}\mathcal L_{\mathrm{vis}}+
\lambda_{\mathrm{pos}}\mathcal L_{\mathrm{pos}}+
\lambda_{\mathrm{vel}}\mathcal L_{\mathrm{vel}}+
\lambda_{\mathrm{col}}\mathcal L_{\mathrm{col}}+
\lambda_{\mathrm{kin}}\mathcal L_{\mathrm{kin}}.
\]

- **존재 \(\mathcal L_{\mathrm{obj}}\):** 활성/빈 슬롯에 대한 binary cross-entropy. 빈 슬롯이 6개 중 일부뿐이므로 batch 평균 후 positive/negative imbalance를 기록하고 필요 시 train 기준 가중치를 쓴다.
- **속성 \(\mathcal L_{\mathrm{attr}}\):** matching된 활성 객체에 대해 색·재질·모양별 categorical cross-entropy를 합산한다. 속성은 클립 단위 타깃이다.
- **가시성 \(\mathcal L_{\mathrm{vis}}\):** matching된 객체의 프레임별 binary cross-entropy. 객체가 화면 밖으로 나가는 경우도 정상 타깃으로 포함한다.
- **위치 \(\mathcal L_{\mathrm{pos}}\):** matching된 활성 객체의 GT 가시 프레임에 대해 정규화 좌표에서 Smooth L1. 비가시 프레임은 시각 입력만으로 위치를 복원할 수 없으므로 주 회귀 손실에서 제외한다.
- **속도 \(\mathcal L_{\mathrm{vel}}\):** GT 가시 프레임에 대한 정규화 world 속도 Smooth L1.
- **충돌 \(\mathcal L_{\mathrm{col}}\):** 두 객체가 모두 보이는 프레임/쌍에 대한 binary cross-entropy. 양성 충돌이 드물기 때문에 positive weight를 train split 빈도에서 계산하고 20 이하로 제한한다. 최종 threshold는 개발 split에서 고정한다. 충돌 허용 오차를 반영한 event F1도 함께 보고한다.
- **운동학 정합 \(\mathcal L_{\mathrm{kin}}\):** 연속 두 프레임 모두 보이는 구간에서 예측 위치 차분과 예측 속도를 약하게 정합한다.

\[
\mathcal L_{\mathrm{kin}} =
\operatorname{SmoothL1}\left(
\frac{\hat p_{k,t+1}-\hat p_{k,t}}{\Delta s},\hat u_{k,t}
\right), \qquad \Delta s=1/16.
\]

이 항은 충돌 전후 실제 속도 변화 자체를 평탄화하지 않는다. 가시 연속 구간의 위치와 속도 단위가 서로 맞도록 돕는 보조 손실이다.

초기 손실 가중치 후보는 \(\lambda_{\mathrm{obj}}=1,\lambda_{\mathrm{attr}}=1,\lambda_{\mathrm{vis}}=1,\lambda_{\mathrm{pos}}=5,\lambda_{\mathrm{vel}}=2,\lambda_{\mathrm{col}}=2,\lambda_{\mathrm{kin}}=0.1\)이다. 좌표/속도 정규화 후 각 항의 크기를 개발 split에서 확인하고, 최종 값은 eval을 보기 전에 고정한다.

### 5.3 물리 제약 사용 원칙

물리 제약은 우선 학습 손실로 넣지 않고 **검증·평가 진단값**으로 사용한다. 우선순위는 GT 상태 복원이며, 작은 위반률을 얻기 위해 객체를 지우거나 궤적을 과도하게 평탄화하는 것을 막는다.

보고할 constraint 그룹과 시작값은 다음과 같다. 현재 설정 파일의 값이며, GT sanity check 후 개발 split에서만 확정한다.

| 그룹 | 검사 항목 | 시작값 |
| :--- | :--- | :--- |
| 객체 | vocab, count, 시간 불변 속성, 빈 슬롯 궤적 | \(K=6\), 색 8 / 재질 2 / 모양 3 |
| 궤적 | table, plane, step, 위치-속도 정합, 비충돌 가속도 | \(R_{xy}=12, z_0=0.20, \tau_z=0.05, v_{\max}=3.2, \tau_{kin}=0.20, \tau_{acc}=0.60\) |
| 충돌 | 대칭·대각, 가시성, 접촉 거리, 관통, 충돌 속도 변화 | \(d_0=0.45, d_{\min}=0.38, \tau_{col}=0.20, \delta_v=0.15\) |

각 위반률은 원래 단위의 예측 상태에서 계산한다. 예측 속성을 강제로 투영하거나 예측 궤적을 보정한 값은 기본 결과에 섞지 않는다.

GT 자체가 제약을 만족하는지 먼저 검사한다. GT에서 위반이 발견되면 임계값 또는 frame sampling/annotation 정렬을 수정하고, 그 뒤 예측 상태의 경계 이탈·텔레포트·관통·충돌 정합 위반을 보고한다. 추후 물리 보조 손실을 시험한다면 이 supervised baseline이 고정된 뒤 별도 ablation으로 추가한다.

---

## 6. 학습과 평가 프로토콜

학습은 배치 비디오를 ResNet에 프레임별로 넣고, 위 손실의 합으로 end-to-end 수행한다. 기본 설정은 배치 4, 최대 30 epoch, AdamW head learning rate (3\times10^{-4}), weight decay (10^{-2}), gradient norm clip 1.0이다. ResNet backbone은 처음에는 고정하며, `backbone_mode: layer3` 비교에서는 layer3에만 (10^{-5}) learning rate를 쓴다.

실행 전 `python scripts/setup_resnet34.py`로 고정된 `ResNet34_Weights.IMAGENET1K_V1` state dict를 로컬에 준비한다. 학습 명령은 `python main.py recognition --mode train --run_name exp_02_sub_video_recognition --config configs/exp_02_sub_video_recognition.yaml`, 최종 평가는 `python main.py recognition --mode eval --run_name exp_02_sub_video_recognition --config configs/exp_02_sub_video_recognition.yaml`이다. 개발 split은 train scene의 10%를 seed 42로 고정해 사용한다. epoch별 손실과 optimizer/scheduler state를 저장하고 dev total loss 기준으로 early stopping 및 `best.pt` 선택을 한다. 종료 시 dev 예측으로 objectness, visibility, collision의 F1 threshold를 각각 보정해 best checkpoint에 저장한다. `last.pt`는 마지막 epoch, `best.pt`는 dev에서 선택된 모델이며 `model.local_dir/last.pt`에는 best 모델을 게시한다.

기본 손실 계수는 설정 파일에서 ((1,1,1,5,2,2,0.1))로 고정한다. matching은 detached 비용 행렬에 SciPy Hungarian assignment를 적용하며, 객체 존재 BCE는 전체 슬롯, 속성은 매칭 객체, 위치·속도는 가시 프레임, 운동학은 연속 가시 프레임, 충돌은 양 객체 가시 프레임에서 계산한다. 위치/속도 정규화 통계와 collision positive weight는 train subset에서만 산출해 checkpoint에 기록한다.

### 주요 지표

| 지표 | 계산 |
| :--- | :--- |
| Object precision / recall / F1 | 존재 threshold 적용 후 clip-level Hungarian matching으로 GT 객체 검출 평가 |
| Attribute set F1 | 활성 객체 속성의 색·재질·모양 분류 |
| Count MAE | 예측 존재 슬롯 수와 GT 객체 수 차이 |
| Visibility F1 | 프레임별 가시성 |
| ADE / FDE | 가시 프레임 world 위치 오차 |
| Velocity MAE | 가시 프레임 world 속도 오차 |
| Collision event P/R/F1 | 객체 쌍 matching 후 허용 프레임 오차 적용 |
| Constraint violation rate | 예측 상태의 물리 제약별 위반률, 진단 지표 |
| Runtime | 초/클립, 프레임당 처리량, GPU 메모리 |

평가 때 모델은 deterministic direct forward 한 번으로 \(\hat S\)를 출력한다. 속도장 NFE, ODE 적분 스텝 수, training-free constraint sampling은 보고하지 않는다. 픽셀 FVD와 CLIP은 인식 실험 지표가 아니다.

공식 PropNet 출력이 있으면 참고 결과로 별도 표기할 수 있지만, 학습 타깃이나 동일 조건 모델 비교로 간주하지 않는다.

---

## 7. Exp-02 생성 실험과 연결

\[
V_{\mathrm{CLEVRER}}\rightarrow
\text{ResNet-34 spatial features}\rightarrow
\text{temporal/object head}\rightarrow
\hat S
\]

Exp-02에서는 생성 영상 \(V_{\mathrm{gen}}\)에도 같은 추론 경로를 적용해 generated_tracks를 만들 수 있다. 출력 좌표·속성·충돌 형식은 본 문서의 \(S\)와 일치시킨다.

다만 CLEVRER validation 성능은 Wan 생성 영상에 대한 일반화 보증이 아니다. 생성 영상에서 이 인식기를 평가할 때는 도메인 이동 결과를 별도로 기록하고, 이 인식기 자체가 Y-Flow의 물리 제약 판정을 유리하게 만드는 순환 평가가 되지 않도록 독립 오라클/수동 표본 검토를 병행한다.

Wan VAE는 본 인식 모델에서 쓰지 않는다. 이 실험은 CLEVRER가 Wan의 생성 조건 입력으로 가능한지를 판정하는 실험도 아니다.

---

## 8. 구현 단계와 완료 조건

1. **입력·상태 정렬 검사**
   train/dev/eval 샘플에서 영상 33프레임, frame_id, object_id, 위치·속도·충돌 시각이 맞는지 시각화한다. GT 상태에 대해 제약 오라클을 실행한다.
2. **ResNet 공간 특징 검사**
   ImageNet 가중치와 전처리를 고정하고, 입력·layer2·layer3·융합 특징의 실제 형상이 위 표와 맞는지 확인한다. 특징맵 overlay로 객체 위치 정보가 보존되는지 표본 검토한다.
3. **시간 헤드 지도학습**
   고정 backbone으로 학습하고 개발 결과를 기록한다. 먼저 시간 헤드를 제거한 per-frame 기준선과 비교한다.
4. **제한적 fine-tuning ablation**
   layer3를 풀어 동일 설정으로 재학습한다. 정확도, 안정성, 시간, 메모리를 비교해 고정 backbone 또는 미세조정 중 하나를 선택한다.
5. **최종 eval 및 Exp-02 연결**
   선택된 checkpoint를 한 번만 eval 100에서 평가한다. 같은 상태 스키마로 결과를 export하고, Wan 생성 영상 검증은 별도 도메인 이동 단계로 둔다.

첫 구현 완료 조건:

- GT 변환 결과에서 객체 수 및 annotation 대응이 원본 JSON과 일치한다.
- 모든 분류/회귀 출력 형상과 slot matching이 작은 batch에서 검증된다.
- 시간 헤드가 없는 기준선과 비교해 궤적 ADE/FDE 또는 충돌 event F1 중 적어도 시간 의존 과제에서 개선되는지 확인한다.
- 최종 표에는 인식 지표, 제약 위반 진단값, 추론 시간 및 사용한 ResNet 가중치를 함께 기록한다.

---

## 9. 한 줄

Exp-02-sub는 ImageNet 사전학습 ResNet-34의 layer2/layer3 공간 특징에 객체 슬롯 및 시간 Transformer 헤드를 붙여, CLEVRER 비디오에서 객체·궤적·충돌 상태를 직접 지도학습하는 인식 실험이다.
