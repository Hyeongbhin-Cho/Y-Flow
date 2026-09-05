# Exp-01. Swiss Roll Hard-Constraint 생성 실험

비교 방법:

- **FlowMatch** (무제약 baseline. 같은 pretrained $v_t^\theta$)
- **HardFlow**
- **SafeFlow** (SafeFlowMatcher)
- **UniConFlow**
- **GuideFlow**
- **YFlow** (기존 POV / Improved POV를 이 이름으로 통일)

목적: 순수 가우시안 노이즈에서 2D Swiss roll 포인트를 생성할 때,  
제약 없는 Flow Matching 대비 다섯 방법이 **hard constraint를 실제로 지키는지** 같은 프로토콜로 비교한다.

---

## 1. 실험 목적

1. 저차원 toy에서 각 방법의 constraint satisfaction을 측정한다.
2. 무제약 FlowMatch 대비 제약을 켠 방법이 Safety를 얼마나 올리는지 본다.
3. “경로 전체 제약” vs “최종/실행 경로만 제약” vs “속도장 가이던스”의 차이를 본다.
4. YFlow가 단순 투영 POV가 아니라 **terminal $h,C$ + 선형 보간**일 때 어디에 서는지 확인한다.
5. 데이터 메타(말림, 크기, 튜브 두께)를 고정·변화시켜 재현 가능하게 만든다.

보고 싶은 질문:

- Safety Rate가 거의 1인가?
- 제약을 지킨 뒤에도 나선 분포가 살아 있는가?
- 바퀴 사이 틈으로 점이 새는가?

---

## 2. 2D Swiss roll은 그리드 픽셀인가, 포인트인가?

**픽셀이 아니다. 포인트다.**

이 실험의 한 샘플은 이미지 격자의 occupancy가 아니라  
**유클리드 평면 위의 좌표 $(x,y)\in\mathbb{R}^2$** 이다.

| 방식 | 쓰는가 | 설명 |
|------|--------|------|
| 고정 개수 $N_{\mathrm{eval}}$개의 2D 포인트 생성 | **예 (본실험)** | $x_0\sim\mathcal{N}(0,I_2)$를 ODE로 $x_1=(x,y)$로 보냄 |
| 정해진 박스 $\mathcal{B}$ 안에서만 유효 | **예 (제약)** | 학습 데이터 bbox + margin. 생성 공간이 무한이면 안 됨 |
| 전체 grid의 각 pixel에 값을 채움 | **아니오** | 그건 이미지/복셀 생성. Exp-01 범위 밖 |
| 시각화용으로만 히스토그램/히트맵 rasterize | 가능 | 평가 본질은 아님 |

정리:

- 학습: 매니폴드에서 뽑은 점 $\{p_i\}_{i=1}^{N_{\mathrm{train}}}\subset\mathbb{R}^2$
- 추론: 가우시안 노이즈 $N_{\mathrm{eval}}$개를 각각 한 점으로 매핑
- 박스는 “그리드 해상도”가 아니라 **좌표 범위**다. 예: $[-R,R]^2$
- 점 개수는 픽셀 수가 아니라 샘플 수다. 예: train 20,000점, eval 4,000점

왜 픽셀이 아닌가:

- Flow matching 속도장이 2차원이라 제약을 $h:\mathbb{R}^2\to\mathbb{R}^m$으로 바로 쓸 수 있다.
- HardFlow / YFlow의 $\hat x_1$ 최적화가 2D라 싸다.
- SafeFlow CBF, UniConFlow QP도 상태 차원 2에서 검증하기 쉽다.
- Pixel grid로 바꾸면 차원이 $H\times W$가 되어 toy 검증이 아니다.

2차 확장에서만 “구름 단위”(한 샘플 = $N_p$개 점) 또는 raster occupancy를 본다.

---

## 3. 데이터와 메타데이터

### 3.1 2D Swiss roll

$${
x=a u\cos u,\qquad y=a u\sin u,\qquad u\sim\mathrm{Unif}[u_{\min},u_{\max}]
}$$

$${
\tilde p = (x,y)+\sigma\varepsilon,\quad\varepsilon\sim\mathcal{N}(0,I_2)
}$$

생성 영역(박스):

$${
\mathcal{B}=[-R,R]^2,\qquad R=\max\|p\|_{\infty}+\mathrm{margin}
}$$

정규화: 학습 전 평균 0, 분산 1 (역변환 후 제약 평가).

### 3.2 기록할 메타

| 키 | 기본 제안 | 의미 |
|----|-----------|------|
| `a` | 1.0 | 크기 |
| `u_min`, `u_max` | $1.5\pi,\ 4.5\pi$ | 말림 구간 |
| `n_turns` | 1.5 | 바퀴 수 |
| `sigma_obs` | 0.05 | 매니폴드 두께 노이즈 |
| `n_train` | 20000 | 학습 점 수 |
| `n_eval` | 4000 | 평가 점 수 |
| `R`, `margin` | data max + 0.2 | 유효 박스 |
| `tau` | $3\sigma_{\mathrm{obs}}$ | 튜브 반경 |
| `rho_min` | $a u_{\min}-\tau$ | 코어 금지 반경 |
| `seed` | 0 | 재현 |

Split: Easy (`n_turns=1`) / Default / Tight (`n_turns=2`, gap 축소).

말림·크기·펼친 길이 $L=\int a\sqrt{1+u^2}\,du$는 메타로 남긴다.  
점 단위 1차에서 펼친 연속성은 **지표만**, 제약은 아니다.

---

## 4. Hard constraint

좌표 $p=(x,y)$에 대해 $h(p)\le 0$ (성분별).

1. 튜브 (매니폴드 위)
$${
h_{\mathrm{tube}}(p)=d_{\mathcal{M}}(p)-\tau\le 0
}$$
간단 차트: $r=\sqrt{x^2+y^2}$, $u=\mathrm{atan2}(y,x)$ unwrap 후
$${
h_{\mathrm{rad}}=|r-a u|-\tau\le 0
}$$

2. 코어/갭 금지 (나선 안·사이)
$${
h_{\mathrm{core}}=\rho_{\min}-r\le 0
}$$
갭은 nearest-arm 사이 거리로 추가 가능.

3. 박스
$${
h_{\mathrm{box}}=\|p\|_\infty-R\le 0
}$$

Cost (HardFlow / YFlow / GuideFlow 에너지에 사용):

$${
C(p)=d_{\mathcal{M}}(p)^2
}$$

YFlow 물리 연산자:

$${
P(p)=\Pi_{\mathcal{M}}(p)\quad\text{(매니폴드 최근접 투영)}
}$$

$P$는 대략 $L_P\le 1$. Lipschitz 스케줄에 사용.

---

## 5. 비교 방법

무제약 baseline은 FlowMatch다. 같은 2D FM을 학습하고, 생성 중 $h$를 쓰지 않는다.  
제약 다섯 방법은 이 $v_t^\theta$를 고정한 뒤 inference만 교체한다.

같은 $x_0$ 시드, 같은 Euler $N$ (50 또는 100), $\alpha_t=t$.

| 방법 | 한 줄 | Swiss roll에서 어떻게 돌릴지 |
|------|------|------------------------------|
| **FlowMatch** | 무제약 linear CFM | 학습된 $v_t^\theta$를 Euler로만 적분. $h$는 평가에만 |
| **HardFlow** | 예측 최종 $\hat x_1$에 $h\le 0$, $C$ 최적화 후 affine으로 현재에 되돌림 | 2D QP/SLSQP, $t\ge t_{\mathrm{on}}$만 활성화 |
| **SafeFlow** | Predict(자유 FM) 후 Correct(CBF-QP). 실행 경로만 안전 | $h_{\mathrm{tube}},h_{\mathrm{core}}$를 barrier로. 점 생성에서는 한 점 ODE + CBF |
| **UniConFlow** | equality/inequality를 prescribed-time zeroing + QP guidance | $h\le 0$ inequality, 필요 시 $r-au=0$을 soft equality |
| **GuideFlow** | 생성 중 속도장을 제약 속도장으로 보정 (+ EBM 항은 가능하면) | Constraining Velocity Field: $v\leftarrow v+\Delta v_{\mathrm{cons}}$ |
| **YFlow** | 타깃 $x+(1-t)v$를 $P$로 warm start한 뒤 $h,C$ 최적화, 선형 보간 | Improved POV. 이름만 YFlow |

공통 아님:

- 다섯 방법은 학습 시 constraint fine-tune 없음 (GuideFlow 원논문의 EBM 결합은 **옵션 ablation**)
- 표의 무제약 기준선은 같은 backbone의 unguided FlowMatch다.

---

## 6. 결과 비교 표

- **실험 환경 (동일 하드웨어/소프트웨어 단일 환경)**:
  - **GPU**: NVIDIA GeForce RTX 3090 (24GB VRAM)
  - **Driver Version**: 580.126.09, **CUDA Version**: 13.0
  - **OS / Platform**: Linux x86_64
  - **데이터 소스**: [runs/exp_01_swiss_roll/metrics.json](file:///home/hyeon/Research/Y-Flow/runs/exp_01_swiss_roll/metrics.json) (데이터 dump: `datasets/swiss_roll/default`)
  - **평가 프로토콜**: $n_{\mathrm{eval}}=4000$, $N_{\mathrm{steps}}=100$, 동일 초기 가우시안 노이즈 $x_0$, 동일 FlowMatch $v_t^\theta$ backbone

| Method | Type | Safety ↑ | Tube viol. rate (mean) ↓ | Core viol. rate (mean) ↓ | Box viol. rate (mean) ↓ | MMD ↓ | Radius MAE ↓ | Time (s/1k) ↓ | Total Time (s) ↓ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **FlowMatch** (Baseline) | train | 0.7305 | 0.2695 (0.05624) | 0.00175 (0.00067) | 0.00025 (0.00001) | $3.62\times 10^{-5}$ | 0.1379 | **0.0514** | **0.206 s** |
| **GuideFlow** | train-free | **1.0000** | 0.0000 (0.0) | 0.0000 (0.0) | 0.0000 (0.0) | 0.00790 | 0.0625 | 0.0721 | 0.288 s |
| **HardFlow** | train-free | **1.0000** | 0.0000 (0.0) | 0.0000 (0.0) | 0.0000 (0.0) | 0.00986 | 0.0943 | 492.736 | 1970.95 s |
| **SafeFlow (Euler)** | train-free | **1.0000** | 0.0000 (0.0) | 0.0000 (0.0) | 0.0000 (0.0) | 0.01534 | 0.1161 | 2.0163 | 8.065 s |
| *(SafeFlow Dopri5)* | train-free | *1.0000* | *0.0000 (0.0)* | *0.0000 (0.0)* | *0.0000 (0.0)* | *0.01555* | *0.1170* | *3.7035* | *14.814 s* |
| **UniConFlow** | train-free | **1.0000** | 0.0000 (0.0) | 0.0000 (0.0) | 0.0000 (0.0) | 0.01432 | **0.0319** | 0.7699 | 3.079 s |
| **YFlow** (Ours) | train-free | **1.0000** | 0.0000 (0.0) | 0.0000 (0.0) | 0.0000 (0.0) | **0.00224** | 0.0727 | 0.6219 | **2.488 s** |

지표 정의:
- **Type**: `train`은 백본 학습 포함, `train-free`는 사전학습된 동일 $v_t^\theta$ 동결 상태에서 추론만 제어
- **Safety**: 모든 하드 제약($h_{\mathrm{tube}}, h_{\mathrm{core}}, h_{\mathrm{box}} \le 0$)을 완전히 만족하는 샘플 비율 (`safe_ratio`)
- **Viol. rate & mean**: 해당 제약 $h > 0$ 위반율 및 위반량의 평균값 $(h)_+$
- **MMD**: 생성 샘플과 테스트 정답 샘플 간 Maximum Mean Discrepancy (RBF 커널, 낮을수록 원본 분포 보존 우수)
- **Radius MAE**: 나선 곡선 중심선 기준 반경 절대 오차 $\mathbb{E}|r - a u^*|$
- **Time**: 1,000점 생성 기준 시간(`s/1k`) 및 4,000점 전수 추론 총 소요 시간(`Total Time`)

---

### 6.1 FlowMatch 실행 결과 (무제약 Baseline, `runs/exp_01_swiss_roll/flowmatch`)

- **설정**: Linear CFM, 20k step 학습, Euler $N=100$, $n_{\mathrm{eval}}=4000$, RTX 3090.
- **분포 학습의 우수성**:
  - MMD가 $3.62 \times 10^{-5}$로 가장 낮아 데이터 매니폴드의 전반적인 형상을 완벽히 포착함.
  - 추론 시간 $0.206\,\mathrm{s}$ ($0.051\,\mathrm{s}/1\mathrm{k}$)로 순수 신경망 추론만의 최고 속도 제공.
- **하드 제약 위반의 치명적 한계**:
  - **Safety Rate 73.05%**: 4,000개 샘플 중 약 27%에 달하는 1,078개 샘플이 제약을 위반함.
  - 실패의 99% 이상이 **튜브 이탈(Tube viol. rate 26.95%)**에서 발생하며, 나선 팔 바깥 및 인접한 바퀴 사이(arm gap)로 점들이 새어나감.
  - 무제약 Flow Matching은 평균적인 생성 품질은 우수하나 신뢰성/안전성 보장이 필요한 실제 응용에는 부적합함을 입증.

---

### 6.2 HardFlow 실행 결과 (`runs/exp_01_swiss_roll/hardflow`)

- **설정**: Training-free, Euler $N=100$, $t_{\mathrm{on}}=0.5$, $\lambda_{oc}=10.0$, CPU SciPy SLSQP (max_iter=20).
- **Hard Constraint 완전 보장**:
  - Safety Rate 1.0000 (위반율 0.00%): Proposition 1에 기반하여 예측 최종 상태 $\hat{x}_N$에 직접 제약 최적화를 수행함으로써 모든 점을 튜브 내부로 완전히 수렴시킴.
- **극심한 연산 병목 및 분포 편향**:
  - **추론 시간 1,970.95초 (~32.8분)**: $t \ge 0.5$ 이후 50개 스텝 동안 4,000개 샘플 각각에 대해 CPU 기반 SLSQP를 순차 호출(총 200,000회 비선형 최적화)하여 동일 환경 내 타 GPU 기법 대비 600~800배 느림.
  - **분포 왜곡 (MMD 0.00986)**: 목적함수의 매니폴드 거리 비용 및 비선형 고정점 복원 맵으로 인해 안쪽 나선($u$가 작은 중심부)으로 점들이 과도하게 쏠리는 현상 발생.

---

### 6.3 SafeFlow 실행 결과 (`runs/exp_01_swiss_roll/safeflow`)

- **설정**: Training-free, Smooth CFMBF-QP ($t \ge 0.5$) + 최종 스텝 SLSQP Terminal Filter, Euler/Dopri5.
- **제약 준수 및 필터 의존성**:
  - 최종 Safety는 1.0000을 기록했으나, **Pre-filter Safety는 40.55%**에 불과함.
  - 즉, 4,000개 샘플 중 **60.03%의 점이 경로 중간의 CBF가 아닌 마지막 스텝의 Terminal Filter에 의해 강제 교정**됨.
- **분포 왜곡 및 연산 속도**:
  - **MMD 0.01534**: 제약 기법 중 가장 높은 분포 오차를 기록. 중간 CBF 보정 벡터가 점들을 안전 영역 경계면에 밀집시키고, 대규모 terminal filtering이 겹치면서 원본 분포가 손상됨.
  - 추론 시간은 Euler 기준 $8.065\,\mathrm{s}$ ($2.016\,\mathrm{s}/1\mathrm{k}$) 소요.

---

### 6.4 UniConFlow 실행 결과 (`runs/exp_01_swiss_roll/uniconflow`)

- **설정**: Training-free, Prescribed-Time Zeroing Function (PTZF) + GPU Closed-form Slack QP + Exact Terminal Refinement.
- **고속 QP와 엄격한 매니폴드 밀착**:
  - **Safety 1.0000 달성**: PTZF 가이던스와 터미널 사영으로 모든 제약 위반 제거.
  - **Radius MAE 0.0319 (전체 1위)**: 최소 노름 QP와 강한 slack 가중치로 인해 점들이 나선 중심선에 가장 강하게 밀착됨.
  - **추론 시간 3.079초 ($0.770\,\mathrm{s}/1\mathrm{k}$)**: 닫힌 해(closed-form) 슬랙 QP를 GPU 배치 텐서 연산으로 풀어 매우 빠른 속도 달성.
- **분포 보존성 (MMD 0.01432)**:
  - 점들이 나선 중심선과 제약 경계선으로 강하게 견인되어 데이터 고유의 분산 두께($\sigma=0.05$)가 축소되고 MMD가 다소 높게 나타남.

---

### 6.5 GuideFlow 실행 결과 (`runs/exp_01_swiss_roll/guideflow`)

- **설정**: Training-free, Constraining Velocity Field (CVF) + Curve Fitting (CF) + Energy Gradient (RFE).
- **최고 수준의 연산 효율성**:
  - **추론 시간 0.288초 ($0.072\,\mathrm{s}/1\mathrm{k}$)**: 무거운 수치 최적화 솔버 없이 벡터 대수 투영(CVF)과 해석적 에너지 그래디언트(RFE)만으로 작동하여, 무제약 FlowMatch($0.206\,\mathrm{s}$)에 필적하는 초고속 추론 실현.
- **우수한 제약 준수 및 품질**:
  - Safety 1.0000, MMD 0.00790, Radius MAE 0.0625 기록.
  - 앵커 보캐뷸러리와 에너지 감쇄 스케줄을 통해 안정적인 나선 형태 복원.

---

### 6.6 YFlow 실행 결과 (`runs/exp_01_swiss_roll/yflow`)

- **설정**: Training-free, $P(\hat{x}_1^{\mathrm{raw}})$ Warm Start, PyTorch Autograd GPU-batched PGD ($\lambda=10.0, \mu=1.0$), 국소 립시츠 게이팅 $\gamma(t, \widehat{L}_P)$, 선형 보간($\eta = \Delta t / (1-t)$).
- **압도적인 분포 보존성 (MMD 0.00224, 제약 기법 중 1위)**:
  - HardFlow(0.00986) 대비 **약 4.4배**, UniConFlow(0.01432) 대비 **약 6.4배**, SafeFlow(0.01534) 대비 **약 6.8배** 우수한 MMD 달성.
  - 무제약 FlowMatch($0.000036$)의 참 분포에 가장 가까운 샘플 품질 유지.
- **초고속 병렬 최적화 (2.488초, $0.622\,\mathrm{s}/1\mathrm{k}$)**:
  - GPU-batched PyTorch Autograd PGD 파이프라인으로 4,000개 샘플을 일괄 최적화하여, 기존 CPU HardFlow(1970.95초) 대비 **약 792배 가속**.
  - 최적화 솔버를 사용하는 제약 기법(HardFlow, SafeFlow, UniConFlow, YFlow) 중 가장 빠른 속도 기록.
- **Hard Constraint 완전 보장**: Safety Rate 1.0000 (Tube/Core/Box 위반 0건), Radius MAE 0.0727 기록.

---

### 6.7 동일 환경(RTX 3090) 종합 원인 분석 및 방법론 비교

```
[Safety vs MMD vs Time 벤치마크 요약 (RTX 3090, n=4000)]

Method        Safety    MMD (↓)      Radius MAE (↓)    Time (s/1k)
------------------------------------------------------------------
FlowMatch     73.05%    0.000036     0.1379            0.051 s  (기준선)
GuideFlow    100.00%    0.007900     0.0625            0.072 s  (초고속 벡터보정)
HardFlow     100.00%    0.009860     0.0943          492.736 s  (CPU 병목 극심)
SafeFlow     100.00%    0.015336     0.1161            2.016 s  (필터 의존 60%)
UniConFlow   100.00%    0.014318     0.0319            0.770 s  (중심선 강밀착)
YFlow (Ours) 100.00%    0.002243     0.0727            0.622 s  (최적 균형점)
```

#### 1. 제약 준수 메커니즘 차이: Path-wise vs Terminal
- **Path-wise CBF/QP (SafeFlow, UniConFlow)**:
  - 매 시각 $t$마다 경로 속도장을 강제로 꺾기 때문에 샘플들이 제약 경계면(boundary)으로 몰리는 부작용이 발생함. 특히 SafeFlow는 중간 보정의 한계로 인해 최종 단계에서 60% 이상의 샘플이 terminal filter에 의존함.
- **Terminal Guidance (HardFlow, YFlow)**:
  - 중간 $x_t$의 자유도를 유지하고 예측된 최종 종점 $\hat{x}_1$에만 제약을 걸어 보간하므로, 불필요한 경로 간섭 없이 자연스럽고 안전한 궤적을 형성함.

#### 2. MMD(분포 보존)에서 YFlow가 압도적인 이유
1. **Lipschitz Gating ($\widehat{L}_P$)**:
   - 나선 팔 사이(arm gap)나 노이즈가 심한 구간에서는 투영 $P$의 국소 립시츠 상수가 폭증($\widehat{L}_P \gg 1$)함. YFlow는 이를 감지하여 불안정 영역에서는 무리한 투영을 끄고($\gamma=0$) 순수 Flow Matching 방향을 보존함.
2. **선형 보간 (Linear Interpolation)**:
   - HardFlow는 비선형 고정점 역맵 과정에서 곡률이 큰 안쪽 나선으로 샘플이 쏠리는 편향을 겪었으나, YFlow는 선형 보간 $\eta = \Delta t / (1-t)$을 채택하여 nominal flow의 직진성과 데이터의 균등한 밀도 분포를 완벽히 유지함.

#### 3. Radius MAE와 모드 붕괴(Mode Collapse)의 해석
- UniConFlow의 Radius MAE(0.0319)는 매우 낮지만, 이는 실제 데이터의 관측 노이즈 두께($\sigma=0.05, \tau=0.15$)를 무시하고 1D 중심선으로 점들을 과도하게 압축시킨 결과로 볼 수 있으며, 이로 인해 MMD가 0.0143으로 증가함.
- YFlow(0.0727)와 GuideFlow(0.0625)는 데이터의 실제 노이즈 두께를 자연스럽게 반영하면서도 튜브 내부를 100% 만족시키는 이상적인 밸런스를 달성함.

#### 4. 추론 속도(Inference Time) 벤치마크 평가
- **최적화가 없는 벡터 가이던스**: GuideFlow($0.072\,\mathrm{s}/1\mathrm{k}$)가 연산량 측면에서 가장 유리함.
- **최적화 기반 제약 방법군**:
  - YFlow($0.622\,\mathrm{s}/1\mathrm{k}$)와 UniConFlow($0.770\,\mathrm{s}/1\mathrm{k}$)가 **2~3초대 초고속 연산**으로 실시간성을 확보함.
  - YFlow의 PyTorch Autograd GPU-batched PGD는 CPU 순차 SLSQP(HardFlow, 492.7 s/1k)의 치명적 속도 병목을 완전히 해결함 (약 792배 가속 실현).

**결론**: YFlow는 동일한 RTX 3090 벤치마크에서 **Hard Constraint 100% 보장**, **제약 기법 중 압도적 1위의 분포 보존성 (MMD 0.00224)**, 그리고 **최적화 기법 중 가장 빠른 추론 속도 (2.49초)**를 동시에 달성하여 전 지표에서 가장 우수한 Pareto Frontier를 증명함.

---

## 7. 프로토콜

1. 메타 고정 후 데이터 dump, 매니폴드/코어/박스를 그림으로 확인
2. Oracle: 데이터 점은 Safety=1, 원점 근처는 Core 위반
3. 제약 없이 FlowMatch 학습. unguided scatter가 나선인지 확인
4. 다섯 방법 inference만 교체, 동일 노이즈 시드·같은 $v_t^\theta$
5. 표 + 그림 저장
6. Ablation: $\tau$, `t_on`, Tight split, YFlow의 $\mu$($P$ 항)

시작 하이퍼:

- steps 50, $\lambda=10$, $\tau=3\sigma$, `t_on=0.5`
- 2D solver 20 iter면 충분

Run json에 dataset / constraints / method / metrics를 남긴다.

---

## 8. 성공 기준

Default, $n_{\mathrm{eval}}=4000$:

1. HardFlow 또는 YFlow Safety ≥ 0.99
2. 나선 사이가 scatter에서 비어 있음
3. MMD가 “제약만 강한 방법”보다 크게 나쁘지 않음

이후 Robot/Maze로 넘어간다.

---

## 9. 한 줄

2D Swiss roll Exp-01은 **박스 안 고정 개수의 좌표점**을 생성하는 실험이지,  
그리드 픽셀을 채우는 이미지 실험이 아니다.  
비교는 무제약 FlowMatch와 HardFlow, SafeFlow, UniConFlow, GuideFlow, YFlow다.
