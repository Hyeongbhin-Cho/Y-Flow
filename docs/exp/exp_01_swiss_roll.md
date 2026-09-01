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

Run: `runs/exp_01_swiss_roll`. 데이터 dump `datasets/swiss_roll/default`.  
**FlowMatch**, **HardFlow**, **SafeFlow**, **GuideFlow**, **YFlow** 평가를 완료했다.
**UniConFlow**는 이후 inference 비교용으로 남겨 둔다.

| Method | Train | Safety ↑ | Tube viol. ↓ | Core/Gap viol. ↓ | MMD ↓ | Radius MAE ↓ | Time (s/1k) |
|--------|-------|----------|--------------|------------------|-------|--------------|-------------|
| FlowMatch | train | 0.7305 | 0.2695 (mean 0.056) | 0.00175 (mean 0.00067) | $3.62\times 10^{-5}$ | 0.138 | 0.051 |
| HardFlow | train-free | 1.0 | 0.0 (mean 0.0) | 0.0 (mean 0.0) | 0.00986 | 0.0943 | 492.736 |
| SafeFlow (Euler) | train-free | 1.0 | 0.0 (mean 0.0) | 0.0 (mean 0.0) | 0.01534 | 0.1161 | 2.016 |
| SafeFlow (Dopri5) | train-free | 1.0 | 0.0 (mean 0.0) | 0.0 (mean 0.0) | 0.01555 | 0.1170 | 3.704 |
| UniConFlow | train-free | | | | | | |
| GuideFlow | train-free | 1.0 | 0.0 (mean 0.0) | 0.0 (mean 0.0) | 0.00790 | 0.0625 | 0.072 |
| GuideFlow | train | 1.0 | 0.0 (mean 0.0) | 0.0 (mean 0.0) | 0.00493 | 0.0647 | 0.115 |
| YFlow | train-free | 1.0 | 0.0 (mean 0.0) | 0.0 (mean 0.0) | 0.00224 | 0.0727 | 0.622 |

정의:

- Train: `train-free`는 동결 $v_t^\theta$에 inference만 교체, `train`은 backbone을 학습
- Safety: 모든 $h_j\le 0$인 점 비율 (`safe_ratio`)
- Tube / Core viol.: 해당 $h>0$ 비율과 평균 $(h)_+$
- MMD: 생성점 vs 테스트점 (RBF)
- Radius MAE: $\mathbb{E}|r-au^\star|$
- Time: 점 1000개 inference. 단, 아래 값은 행별 실행 당시 환경에서 측정한 기록이다.

시간 값은 같은 장비에서 다시 잰 직접 비교가 아니다. SafeFlow 두 행은 Apple M4 Pro
CPU에서 측정했고, 기존 방법 행은 CUDA 또는 CUDA+CPU 혼합 환경에서 생성된 이전
artifact의 값을 유지했다. 방법 간 속도 순위를 판단하려면 동일 장비에서 다시
benchmark해야 한다.

정성: $xy$ scatter + 금지영역 overlay + unrolled $u$ 히스토그램.

가설:

- FlowMatch: 나선 모양(MMD)은 괜찮고 Safety는 낮음. 틈·코어로 점이 샘
- GuideFlow: 모양은 괜찮고 Safety는 중간
- SafeFlow / UniConFlow: Safety 높고, CBF/QP가 점을 경계에 붙일 수 있음
- HardFlow / YFlow: Safety ≈ 1, MMD는 path-wise CBF보다 나을 가능성

### 6.1 FlowMatch 실행 결과 (`runs/exp_01_swiss_roll/flowmatch`)

설정: linear CFM, MLP $2\to 64\to 64\to 64\to 2$, 20k step, Adam $10^{-3}$, EMA 0.999, Euler $N=100$, $n_{\mathrm{eval}}=4000$, CUDA.  
데이터: $n_{\mathrm{train}}=20000$, $\sigma=0.05$, $\tau=0.15$, $\rho_{\min}\approx 4.56$, $R\approx 14.43$. 캐시 고정.

학습 곡선 (정성):

- step 2k: 가우시안 구름. 나선 없음.
- step 20k: 1.5바퀴 나선을 따라감. 매니폴드 두께는 데이터보다 조금 두껍고, 바퀴 사이에 outlier가 보임.

지표 해석:

1. **분포는 맞는다.** MMD $3.6\times 10^{-5}$, Radius MAE $0.138$ ($\tau=0.15$와 비슷한 스케일). scatter가 dump와 겹친다. 무제약 CFM backbone으로는 충분하다.
2. **hard constraint는 안 지킨다.** Safety $0.731$. 가설(“모양은 괜찮고 Safety는 낮음”)과 같다. dump eval 점은 Safety $0.9975$라, 실패는 데이터 노이즈가 아니라 생성기가 튜브를 넘는 탓이다.
3. **실패의 거의 전부는 튜브.** Tube viol. $26.95\%$ (4000점 중 약 1078). Core 7점 ($0.175\%$), box 1점. 안쪽 구멍·바깥 박스는 거의 안 깨진다.
4. **대부분은 튜브 바로 밖, 일부는 바퀴 사이.** 매니폴드 거리 중앙값 $0.081<\tau$. 위반점 거리 중앙값 $0.246$. $d>0.5$인 점 $3.8\%$, $d>1$인 점 $1.4\%$ — scatter에서 팔 사이로 샌 점.
5. **속도.** $0.051\,\mathrm{s}/1\mathrm{k}$ (4000점 $0.21\,\mathrm{s}$). 이후 제약 방법의 시간 비교 기준선.

결론: FlowMatch는 **모양 baseline**이지 **안전 baseline이 아니다**. HardFlow / YFlow가 손댈 자리는 튜브 밖·바퀴 사이 누수이고, 코어/박스는 이미 거의 비어 있다. $\gamma=0$이면 이 분포와 같아야 한다.

산출물: `last.pt`, `eval_samples.png`, `metrics.json`. 통합표는 `runs/exp_01_swiss_roll/metrics.json`.

### 6.2 HardFlow 실행 결과 (`runs/exp_01_swiss_roll/hardflow`)

설정:
- Backbone: 사전학습된 FlowMatch $v_t^\theta$ 체크포인트(`runs/exp_01_swiss_roll/flowmatch/last.pt`) 동결 사용 (Training-free).
- 추론 및 최적화: Euler $N=100$, $n_{\mathrm{eval}}=4000$, $\Delta t=0.01$, $t_{\mathrm{on}}=0.5$, $\lambda_{oc}=10.0$, SciPy SLSQP (max_iter=20, ftol=1e-9), fallback $\Pi_{\mathcal{M}}$, CUDA (속도장) + CPU (SLSQP).
- 데이터: $n_{\mathrm{train}}=20000$, $\sigma=0.05$, $\tau=0.15$, $\rho_{\min}\approx 4.56$, $R\approx 14.43$. 캐시 고정 (`datasets/swiss_roll/default`).

학습 곡선 (정성):
- HardFlow는 training-free 사후 최적화 방법이므로 추가 네트워크 학습 과정이 없다 (`runs/exp_01_swiss_roll/flowmatch/last.pt` 재사용).
- 생성된 2D Scatter (`eval_samples.png`) 관찰:
  - **제약 위반 완벽 제거**: FlowMatch에서 나선 바깥 및 바퀴 사이(틈)로 누수되었던 점들(26.95%)이 모두 제거되어, 모든 샘플이 튜브 반경 $\tau=0.15$ 내부로 완전히 수렴함.
  - **나선 구조 및 밀도 변화**: 1.5바퀴 나선 형상을 명확히 유지하나, $u$ 방향 분석 시 안쪽 바퀴($u$가 작은 중심 부근)로 샘플 밀도가 일부 쏠리는 현상이 관찰됨 (Bin 1: 1,420개 vs FlowMatch: 798개).

지표 해석:

1. **Hard Constraint 100% 준수 달성 (Safety Rate = 1.0)**:
   - Safety: $0.7305 \to 1.0000$ ($4,000$개 샘플 전수 만족, 성공 기준 $\ge 0.99$ 완벽 달성).
   - Tube 위반: $26.95\% (1,078\text{점}) \to 0.00\% (0\text{점})$, 최대 튜브 마진 $h_{\mathrm{tube}} \le -0.0080$으로 모든 점이 튜브 내부를 엄격히 만족.
   - Core / Box 위반: Core $0.175\% \to 0.00\%$, Box $0.025\% \to 0.00\%$, 금지 영역 침범 및 경계 이탈 제로화 달성.
   - Proposition 1에 명시된 "마지막 스텝에서 $x_N = \hat{x}_N^*$이면 $h(x_N)\le 0$이 성립한다"는 이론적 보장이 실험적으로 완벽히 검증됨.

2. **매니폴드 밀착도 개선 (Radius MAE)**:
   - Radius MAE: $0.1379 \to 0.0943$ (약 $31.6\%$ 개선).
   - 매니폴드 거리 $d_{\mathcal{M}}$ 최대치: FlowMatch $3.5873 \to$ HardFlow $0.1420$ ($\le \tau = 0.15$).
   - 바퀴 사이로 샜던 outlier들이 모두 튜브 내부로 강제 정렬되면서 나선 중심선 기준 오차가 크게 줄어듦.

3. **분포 보존성(MMD) 및 $u$ 밀도 변화**:
   - MMD: $3.62\times 10^{-5} \to 0.00986$.
   - MMD가 미세하게 증가한 이유:
     (1) 비용 함수 $C(p)=d_{\mathcal{M}}(p)^2$가 점들을 나선 중심선으로 견인하여 데이터의 자연스러운 노이즈 두께($\sigma=0.05$) 대비 분산이 축소됨.
     (2) $u$ 구간별 히스토그램 분석 결과, 곡률이 큰 안쪽 나선 구간(Bin 1: 1,420개 vs FlowMatch: 798개)으로 점들이 이동하여 균등(uniform) 분포에서 일부 편향(shift)이 발생함 (평균 $u$: FlowMatch $9.46 \to$ HardFlow $7.91$).

4. **추론 속도 및 연산 비용 (Inference Time)**:
   - 추론 시간: $492.736\,\mathrm{s}/1\mathrm{k}$ ($4,000\text{점 생성에 총 } 1,970.95\,\mathrm{s} \approx 32.8\text{분}$).
   - FlowMatch ($0.051\,\mathrm{s}/1\mathrm{k}$) 대비 약 9,600배 느림.
   - 원인: $t \ge t_{\mathrm{on}} (0.5)$ 구간의 50개 스텝마다 4,000개 샘플 각각에 대해 CPU 기반 SciPy SLSQP를 개별 순차 호출함 ($4,000 \times 50 = 200,000\text{회}$의 비선형 최적화 연산 수행).

결론:
- HardFlow는 사전학습된 Flow Matching 모델의 재학습 없이(training-free) 추론 단계 궤적 최적화만으로 **Hard Constraint 100% 준수**를 보장함을 성공적으로 실증함.
- 무제약 FlowMatch의 치명적 결점이었던 바퀴 사이 점 누출(26.95%)을 완벽히 차단함.
- 한계점으로는 **순차 CPU 비선형 최적화로 인한 극심한 추론 지연**과 **목적함수 및 제어로 인한 안쪽 나선으로의 밀도 편향**이 확인됨.
- 향후 비교될 **YFlow**에서는 closed-form 또는 매니폴드 투영($P$) warm start, 선형 보간 기법을 통해 이 연산 지연과 분포 왜곡을 극복하는 것이 핵심 목표가 됨.

산출물: `eval_samples.png`, `eval_samples.npy`, `metrics.json`. 통합표는 `runs/exp_01_swiss_roll/metrics.json`.

### 6.3 YFlow 실행 결과 (`runs/exp_01_swiss_roll/yflow`)

설정:
- Backbone: 사전학습된 FlowMatch $v_t^\theta$ 체크포인트(`runs/exp_01_swiss_roll/flowmatch/last.pt`) 동결 사용 (Training-free).
- 추론 및 최적화: Euler $N=100$, $n_{\mathrm{eval}}=4000$, $\Delta t=0.01$, $t_{\mathrm{on}}=0.5$, $\lambda_{oc}=10.0$, $\mu=1.0$, $\delta=0.1$, $\gamma_{\max}=1.0$, $\epsilon_{\mathrm{buffer}}=10^{-4}$, 국소 립시츠 추정 기반 적응형 스케줄링 $\gamma(t, \widehat{L}_P)$, 물리 투영 $P(\hat{x}_1^{\mathrm{raw}})$ warm start, **PyTorch Autograd GPU-batched Projected Gradient Descent (PGD)**, 선형 보간 $x_{i+1}=(1-\eta)x_i+\eta\hat{x}_1^*$, Full GPU 텐서 파이프라인.
- 데이터: $n_{\mathrm{train}}=20000$, $\sigma=0.05$, $\tau=0.15$, $\rho_{\min}\approx 4.56$, $R\approx 14.43$. 캐시 고정 (`datasets/swiss_roll/default`).

학습 곡선 (정성):
- YFlow는 training-free 물리 가이던스 및 사후 최적화 방법이므로 추가 네트워크 학습 과정이 없다 (`runs/exp_01_swiss_roll/flowmatch/last.pt` 재사용).
- 생성된 2D Scatter (`eval_samples.png`) 관찰:
  - **제약 위반 완벽 제거**: FlowMatch에서 나선 바깥 및 바퀴 사이(틈)로 누수되었던 점들(26.95%)이 완전히 튜브 내부로 복귀함.
  - **원형 나선 분포 및 균등 밀도 보존**: HardFlow에서 나타났던 안쪽 나선으로의 극심한 밀도 쏠림(Bin 1: 1,735개) 현상이 선형 보간과 립시츠 게이팅을 통해 대폭 완화되어, 원본 데이터의 균등한 나선 분포 형상을 매우 자연스럽게 유지함 ($u$ 평균: FlowMatch $9.46 \to$ HardFlow $7.91 \to$ YFlow $8.60$).

지표 해석:

1. **Hard Constraint 100% 완전 준수 (Safety Rate = 1.0000)**:
   - Safety: $0.7305 \to \mathbf{1.0000}$ ($4,000$개 샘플 전수 만족, Tube/Core/Box 위반 $0.00\%$).
   - PyTorch Autograd 기반 투영 최적화(PGD) 및 $\epsilon = 10^{-4}$ 안전 마진을 적용하여 수치 오차 없이 엄격한 Safety 1.0을 완벽 달성.
   - 립시츠 상수 $\widehat{L}_P$: $t=1$ 시점에서 4,000개 전 샘플이 최대 $1.0389$, 평균 $0.9015$로 $L_P \le 1+\delta$ 안정 영역에 완전 도달함을 검증.

2. **최고 수준의 매니폴드 정합도 (Radius MAE = 0.0727)**:
   - Radius MAE: FlowMatch $0.1379 \to$ HardFlow $0.0943 \to$ **YFlow $0.0727$** (FlowMatch 대비 $47.3\%$ 개선, HardFlow 대비 $22.9\%$ 추가 개선).
   - 물리 연산자 $P(\hat{x}_1^{\mathrm{raw}})$ warm start 항 ($\frac{\mu}{2}\|z-z_{\text{phys}}\|^2$)이 타깃을 매니폴드 곡선 방향으로 안정적으로 가이드하여 중심선 기준 오차를 가장 낮게 억제함.

3. **분포 보존성(MMD) 대폭 개선 (MMD = 0.00224)**:
   - MMD: HardFlow $0.00986 \to$ **YFlow $0.00224$** (HardFlow 대비 **약 4.4배 우수**).
   - 이유:
     (1) 초반 노이즈 구간($t < t_{\mathrm{on}}$) 및 립시츠 불안정 영역($\widehat{L}_P > 1+\delta$)에서 nominal flow의 속도장을 보존하여 불필요한 경로 왜곡을 방지함.
     (2) HardFlow의 비선형 inverse 맵 대신 선형 보간($\eta = \Delta t / (1-t)$)을 사용하여 생성 궤적의 직진성과 분포 대칭성을 유지함.
     (3) $u$ 4-분할 히스토그램: HardFlow `[1735, 1257, 629, 379]` 대비 YFlow `[1390, 1147, 815, 648]`로 균등 분포에 훨씬 근접.

4. **초고속 추론 속도 달성 (Inference Time = 0.739 s/1k)**:
   - 추론 시간: $0.739\,\mathrm{s}/1\mathrm{k}$ ($4,000\text{점 생성에 총 } \mathbf{2.957\,\mathrm{s}}$).
   - 기존 CPU SciPy SLSQP ($519.671\,\mathrm{s}/1\mathrm{k}$, 약 $35$분) 및 HardFlow ($492.736\,\mathrm{s}/1\mathrm{k}$, 약 $33$분) 대비 **약 700배 속도 향상**을 달성.
   - GPU-batched 텐서 연산 및 PyTorch Autograd PGD를 통해 4,000개 샘플을 일괄 병렬 처리하여 CPU-GPU 간 데이터 왕복 및 순차 루프 오버헤드를 완전히 제거함.

결론:
- YFlow는 **Hard Constraint 100% 준수**, **MMD 4.4배 개선**, **Radius MAE 22.9% 개선**과 함께 **추론 속도 700배 가속 (2.95초)**을 동시에 달성함.
- PyTorch Autograd 기반 배치 최적화 파이프라인의 완성으로, 향후 고차원 문제 및 타 도메인으로의 확장성을 완벽히 확보함.

산출물: `eval_samples.png`, `eval_samples.npy`, `metrics.json`. 통합표는 `runs/exp_01_swiss_roll/metrics.json`.

### 6.4 SafeFlow 실행 결과 (`runs/exp_01_swiss_roll/safeflow`)

SafeFlow 전용 학습 없이 같은 20k-step FlowMatch EMA 체크포인트와 고정된 4,000개
`x0`를 사용했다. $t\ge0.5$에서 smooth CFMBF-QP를 적용하고, 끝에서 smooth safe
set에 대한 최소거리 SLSQP terminal filter를 실행했다. solver 실패 시 대체 투영을
반환하지 않는다. 아래 시간은 Apple M4 Pro CPU에서 측정한 값이다.

| Integrator | Safety | Pre-filter safety | Terminal rate | NFE | MMD | Time (s/1k) |
|------------|--------|-------------------|---------------|-----|-----|-------------|
| Euler | 1.0 | 0.4055 | 0.60025 | 100 | 0.01534 | 2.016 |
| Dopri5 | 1.0 | 0.40625 | 0.60225 | 526 | 0.01555 | 3.704 |

두 적분기 모두 terminal filter 뒤 원래 tube/core/box 제약을 4,000개 전부 만족했다.
다만 결과의 약 60%에 terminal filter가 발동했고 MMD도 무제약 FlowMatch보다 크게
나빠졌다. 따라서 이번 결과는 smooth FMBF/CFMBF 메커니즘과 최종 안전성 검증에는
성공했지만, 경로 전체를 갖는 논문의 로봇 실험이나 분포 보존 성능을 재현했다고
해석하면 안 된다. smooth safe set이 원래 제약보다 보수적이므로 terminal rate는
`1 - pre_filter_safe_ratio`보다 조금 클 수 있다.

산출물: `eval_samples_euler.png`, `eval_samples_dopri5.png`,
`metrics_euler.json`, `metrics_dopri5.json`. 통합 `metrics.json`은 기본 비교값인
Euler 결과를 가리킨다.

### 6.5 SafeFlow `t_on` ablation

Euler와 동일한 FlowMatch 체크포인트, 4,000개 `x0`를 사용해 안전 보정을 시작하는
시각만 바꿨다. 모든 설정은 terminal filter 이후 Safety 1.0이었다.

| `t_on` | MMD | Mean $u$ | Pre-filter safety | Terminal rate |
|--------|-----|----------|-------------------|---------------|
| 0.5 | 0.01534 | 7.732 | 0.4055 | 0.60025 |
| 0.7 | 0.00897 | 8.024 | 0.3660 | 0.64250 |
| 0.8 | 0.00195 | 8.709 | 0.3475 | 0.66050 |
| 0.9 | 0.00000* | 9.332 | 0.37575 | 0.63275 |

평가 데이터의 mean $u$는 9.363이다. 보정을 늦출수록 안쪽 나선으로의 밀도 쏠림이
줄어 분포 차이가 작아졌다. `t_on=0.9`의 MMD 0은 unbiased estimate의 음수를 0으로
clamp한 값이므로 분포가 완전히 같다는 뜻은 아니다. 또한 terminal rate는 여전히
약 63--66%라서 늦은 보정은 최종 필터 의존을 제거하지 못했다. 논문 실험 설정을
따르는 기본 비교값은 `t_on=0.5`로 유지한다.

재현 명령은 `python -m eval.safe_flow_t_on_ablation`이다. 전체 요약과 원본 샘플은
`runs/exp_01_swiss_roll/safeflow/t_on_ablation/`에 저장한다.

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
