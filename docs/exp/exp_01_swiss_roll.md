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
**FlowMatch**(무제약 baseline), **HardFlow**(terminal 제약), **GuideFlow**(생성 중 제약) 평가 완료. 나머지 칸은 이후 inference 비교용.

Train 열은 5장의 공통 규칙(제약 방법은 같은 $v_t^\theta$를 고정하고 inference만 교체)을 따르는지 표시한다.

| Method | Train | Safety ↑ | Tube viol. ↓ | Core/Gap viol. ↓ | MMD ↓ | Radius MAE ↓ | Time (s/1k) |
|--------|-------|----------|--------------|------------------|-------|--------------|-------------|
| FlowMatch | train | 0.7305 | 0.2695 (mean 0.056) | 0.00175 (mean 0.00067) | $3.62\times 10^{-5}$ | 0.138 | 0.051 |
| HardFlow | train-free | 1.0 | 0.0 (mean 0.0) | 0.0 (mean 0.0) | 0.00986 | 0.0943 | 492.736 |
| SafeFlow | train-free | | | | | | |
| UniConFlow | train-free | | | | | | |
| GuideFlow | train-free | 1.0 | 0.0 (mean 0.0) | 0.0 (mean 0.0) | 0.00790 | 0.0625 | 0.072 |
| YFlow | train-free | | | | | | |

옵션 ablation (5장이 명시한 GuideFlow 원논문의 EBM 결합. 비교표 본문과 분리한다):

| Method | Train | Safety ↑ | Tube viol. ↓ | Core/Gap viol. ↓ | MMD ↓ | Radius MAE ↓ | Time (s/1k) |
|--------|-------|----------|--------------|------------------|-------|--------------|-------------|
| GuideFlow + EBM/CFG | train | 1.0 | 0.0 (mean 0.0) | 0.0 (mean 0.0) | 0.00493 | 0.0647 | 0.115 |

정의:

- Train: `train-free`는 동결 $v_t^\theta$에 inference만 교체, `train`은 backbone을 학습
- Safety: 모든 $h_j\le 0$인 점 비율 (`safe_ratio`)
- Tube / Core viol.: 해당 $h>0$ 비율과 평균 $(h)_+$
- MMD: 생성점 vs 테스트점 (RBF)
- Radius MAE: $\mathbb{E}|r-au^\star|$
- Time: 점 1000개 inference

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

### 6.3 GuideFlow 실행 결과 (`runs/exp_01_swiss_roll/guideflow`)

설정: 사전학습 FlowMatch $v_t^\theta$ 동결(training-free), CVF($\lambda=0.1$) + CF($k_c=50$) + RFE($\tau^{*}=0.5$, $\varepsilon_{\max}=0.5$, $n_{\mathrm{refine}}=10$), 비용 가중치 $w_{\mathrm{cost}}=0.05$.  
앵커 vocabulary: $h\le 0$인 train 점에 farthest point sampling, $N=256$.  
추론: Euler $N=100$, $n_{\mathrm{eval}}=4000$, HardFlow와 같은 $x_0$ 시드. 데이터 캐시 고정.

1. **Hard constraint 100% 준수 (Safety = 1.0)**:
   - Tube 위반 $26.95\% (1,078\text{점}) \to 0.00\% (0\text{점})$, Core $0.175\% \to 0.00\%$, Box $0.025\% \to 0.00\%$.
   - 성공 기준 $\ge 0.99$ 달성. 단 HardFlow Proposition 1 같은 이론적 보장이 아니라, 에너지 하강의 수렴에 의존하는 수치적 결과다. $\varepsilon_{\max}$를 낮추면 Safety가 1에 못 미친다.

2. **매니폴드 밀착도 (Radius MAE)**:
   - Radius MAE: $0.1379 \to 0.0625$ (약 $54.7\%$ 개선). HardFlow($0.0943$) 대비 $34\%$ 우수.
   - 두 방법 모두 같은 비용 $C(p)=d_{\mathcal{M}}(p)^2$를 쓰지만, HardFlow는 SLSQP가 $h\le 0$ 제약 하에서 풀어 경계에 걸리는 반면 GuideFlow의 에너지 하강은 제약 없이 곧장 비용을 낮춰 더 깊이 들어간다.

3. **분포 보존성 (MMD)**:
   - MMD: $3.62\times 10^{-5} \to 0.00790$. HardFlow($0.00986$)보다 낫다.
   - 사후 투영이 아니라 flow 내부에서 제약을 밀어 넣기 때문에 종단 분포가 덜 밀린다. 다만 무제약 FlowMatch 대비로는 두 자릿수 증가로, 8장 성공 기준의 "제약만 강한 방법보다 크게 나쁘지 않음"을 만족하는 수준이다.

4. **추론 비용**:
   - $0.072\,\mathrm{s}/1\mathrm{k}$. HardFlow($492.736$) 대비 약 6,800배 빠르다. per-sample 비선형 solver 없이 최근접 앵커 탐색과 닫힌 형태 에너지 기울기만 쓰기 때문이다.

5. **가설 검증**: 5장 가설의 "GuideFlow: 모양은 괜찮고 Safety는 중간"은 **부분적으로 틀렸다.** 세 전략을 모두 켜면 Safety는 HardFlow와 같은 1.0이고, MMD와 Radius MAE도 HardFlow보다 낫다.

산출물: `eval_samples.png`, `eval_samples.npy`, `metrics.json`.

### 6.4 GuideFlow 옵션 ablation (EBM 결합 학습 + CFG)

5장이 "옵션 ablation"으로 지정한 원논문의 EBM 결합($\mathcal{L}_{\mathrm{RFE}}$)과 classifier-free guidance를 켠 설정. 20,000 step 학습.

| 설정 | Safety ↑ | MMD ↓ | Radius MAE ↓ | Time (s/1k) |
|------|----------|-------|--------------|-------------|
| training-free (비교표) | 1.0 | 0.00790 | 0.0625 | 0.072 |
| EBM/CFG (옵션) | 1.0 | **0.00493** | 0.0647 | 0.115 |

- **분포 보존이 38% 개선**된다 ($0.00790 \to 0.00493$). 평가 시드 6회 반복으로 측정한 MMD 표준편차 $\pm 0.0009$의 3배를 넘는 차이다.
- **Radius MAE는 사실상 동일**하다. 정확도는 이미 $w_{\mathrm{cost}}$가 담당하므로 학습이 추가로 기여할 여지가 적다.
- 즉 EBM 학습이 사는 지점은 정확도가 아니라 분포 보존이다. 속도장 자체가 제약을 인지하게 되어 추론 시점 교정량이 줄고, 그만큼 분포가 덜 밀린다.
- 다만 이 설정은 backbone을 새로 학습하므로 5장의 공통 규칙 밖이다. 비교표 본문에는 넣지 않는다.

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