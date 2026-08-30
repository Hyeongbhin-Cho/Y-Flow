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

| Method | Safety ↑ | Tube viol. ↓ | Core/Gap viol. ↓ | MMD ↓ | Radius MAE ↓ | Time (s/1k) |
|--------|----------|--------------|------------------|-------|--------------|-------------|
| FlowMatch | | | | | | |
| HardFlow | | | | | | |
| SafeFlow | | | | | | |
| UniConFlow | | | | | | |
| GuideFlow | | | | | | |
| YFlow | | | | | | |

정의:

- Safety: 모든 $h_j\le 0$인 점 비율
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