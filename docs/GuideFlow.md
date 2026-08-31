# GuideFlow 구현 연구 정리

Constraint-Guided Flow Matching for Planning in End-to-End Autonomous Driving  
(Lin Liu, Caiyan Jia, Guanyi Yu, Ziying Song et al., BJTU / Qcraft)

본 문서는 논문 구현을 위한 **연구 목적, 이론, 알고리즘**을 정리한다.

---

## 1. 연구 목적

Flow Matching은 사전분포 \(p_0\)에서 데이터 분포 \(p_1\)로 샘플을 수송한다.  
자율주행 planner에서 생성된 궤적은 **반드시** 지켜야 하는 조건이 있다.

- 주행 궤적은 다른 차량과 충돌하면 안 된다.
- 궤적은 주행 가능 영역(drivable area)을 벗어나면 안 된다.
- 운동학적으로 실행 불가능한 궤적을 내면 안 된다.

기존 방법의 한계:

| 계열 | 한계 |
|------|------|
| Imitative E2E planner (UniAD, VAD, SparseDrive) | 장면마다 GT가 하나뿐이라 multimodal 출력이 하나로 붕괴 (**mode collapse**) |
| Generative E2E planner (DiffusionDrive, GoalFlow) | 분포는 다양하나 샘플링이 확률적이라 **hard constraint를 보장하지 못함** |
| 사후 최적화 단계 추가 | 생성과 제약이 분리되어 **추가 단계 비용**과 분포 이동이 큼 |
| Soft guidance (에너지 penalty만) | 제약을 권장할 뿐 **보장하지 못함** |

GuideFlow의 목표:

1. **Flow matching 명시적 모델링**: 랜덤 prior에서 출발해 mode collapse를 완화하고 다양한 조건 신호로 유도.
2. **생성 과정 내부 제약**: 사후 최적화 단계로 미루지 않고 생성 중에 제약을 직접 강제.
3. **EBM 결합**: 에너지 지형을 학습해 모델이 제약 만족 영역을 스스로 탐색.
4. **Style 제어**: 보상(EP)을 조건 신호로 두어 추론 시점에 aggressive/conservative 전환.

핵심은 세 가지 제약 주입 전략 CVF, CF, RFE다.

---

## 2. 배경: Flow Matching과 Energy Matching

시간 \(t\in[0,1]\)에서 상태 \(x_t\in\mathbb{R}^d\), 속도장 \(v_\theta\). ODE

$${
\frac{dx_t}{dt}=v_\theta(x_t,t),\qquad t\in[0,1],\quad x_0\sim\pi_0
}$$

를 따른다. Rectified flow는 prior \(\pi_0\)와 target \(\pi_1\) 사이에 선형 경로를 놓는다.

$${
x_t=(1-t)x_0+t x_1
}$$

학습 목적:

$${
\mathcal{L}_{\mathrm{RF}}=\mathbb{E}_{t,x_0\sim\pi_0,x_1\sim\pi_1}\big\|v_\theta(x_t,t)-(x_1-x_0)\big\|^2
}$$

추론은 수치적분:

$${
x^{(k+1)}=x^{(k)}+v_\theta(x^{(k)},t_k)\Delta t,\qquad t_k=\frac{k}{K}
}$$

이 직선 수송은 빠르고 안정적이지만 본질적으로 **mode-seeking**이라 지배적 패턴으로 붕괴한다.

Energy Matching은 에너지 함수 \(E_\theta\)를 도입해 여러 feasible mode를 복원한다. 동적 형태의 최적성 조건은

$${
\frac{x_{t+\Delta t}-x_t}{\Delta t}+\nabla_{x_t}v_\theta(x_t)+\varepsilon(t)\nabla_{x_t}\log\phi_t(x_t)=0
}$$

이고, 에너지 가중 스케줄은

$${
\varepsilon(t)=
\begin{cases}
0, & 0\le t<\tau^{*},\\[2pt]
\varepsilon_{\max}\dfrac{t-\tau^{*}}{1-\tau^{*}}, & \tau^{*}\le t\le 1,\\[6pt]
\varepsilon_{\max}, & t\ge 1.
\end{cases}
}$$

데이터 매니폴드 근방에서 수송항이 사라지면 종단 분포는 Boltzmann 형태가 된다.

$${
\pi_1(x)\propto\exp(-\beta E_\theta(x)),\qquad \beta=\varepsilon_{\max}^{-1}>0
}$$

따라서 \(E_\theta\)는 매니폴드를 여러 저에너지 basin으로 조각내고, 각 basin이 하나의 feasible mode(yield, merge 등)에 대응한다. 이산화된 샘플링 업데이트는

$${
x^{(k+1)}=x^{(k)}+v_\theta(x^{(k)},t_k)\Delta t-\eta(t_k)\nabla_x E_\theta(x^{(k)})
}$$

\(0<t<1\)에서는 flow 항이 샘플을 매니폴드로 수송하고, \(t\ge\tau^{*}\)부터 에너지 항이 켜져 저에너지 mode로 밀어 넣는다.

---

## 3. 문제 정의

### 3.1 Nominal distribution

고정된 pretrained \(v_t^\theta\)와 초기분포 \(\pi_0\)에 대해, 제약 없이 적분한 최종 분포를 nominal distribution \(\bar\mu\)라 한다.

$${
\bar\mu=(\Phi^\theta_{0\to 1})_{\#}\pi_0
}$$

### 3.2 Hard constraint와 cost

- Hard constraint: \(h(x)\le 0\). 반드시 만족. 벡터면 성분별 만족.
- Cost \(C(x)\): 작을수록 좋지만 제약보다 우선순위가 낮음.
- Exp-01에서는 \(h=(h_{\mathrm{tube}},h_{\mathrm{core}},h_{\mathrm{box}})\), \(C(p)=d_{\mathcal{M}}(p)^2\).

### 3.3 조건 신호와 앵커

속도장에 제어를 더하는 대신, GuideFlow는 세 가지를 놓는다.

- **앵커** \(x_1^{c}\): 제약을 만족하는 종단 후보. 궤적 vocabulary \(\mathcal{V}_a\)(\(N=256\), farthest point sampling)에서 고른다.
- **조건 신호** \(c\): plan anchor \(C_p\), goal point \(C_g\), driving command \(C_d\), reward \(C_r\). 앞 셋은 의미가 겹쳐 동시에 쓰지 않는다.
- **에너지** \(E_\theta\): 제약 만족도를 평가하는 스칼라 지형.

중간 상태 \(x_t\)가 feasible일 필요는 없다. 최종 \(x_1\)만 \(h(x_1)\le 0\)이면 된다.

---

## 4. 이론: 세 가지 제약 주입

매 업데이트 \(x^{(k+1)}\)는 (1) 속도장 \(v_\theta\), (2) 직전 상태 \(x^{(k)}\), (3) 정련 구간의 에너지 \(E_\theta\)에 의존한다. 세 전략은 각각을 건드린다.

### 4.1 Strategy 1 — CVF (Constraining the Velocity Field)

앵커 \(x_1^{c}\)에 대응하는 제약 만족 속도장은

$${
v_t^{c}=\frac{x_1^{c}-x_0}{1-0}
}$$

이다. 이 방향은 종단에서의 제약 만족을 보장하지만 그 자체로는 최적이 아니다. 그래서 예측 속도장을 **반사**시켜 보정한다.

$${
v_t^{*}=v_t-\frac{2\lambda\,v_t\cdot v_t^{c}}{\|v_t^{c}\|^2}v_t^{c}
}$$

목적은 \(v_t\)의 **방향만 조정하고 크기는 최소한으로 건드리는** 것이다. 기하학적 의미는 §6.1에서 정리한다.

### 4.2 Strategy 2 — CF (Constraining the Flow States)

속도장 보정만으로는 적분 중 flow 경로가 제약 매니폴드에서 이탈할 수 있다. 연속 flow \(\phi_t\)를 이산화하면

$${
\phi'_t=\{x^{(0)},\dots,x^{(k)},\dots,x^{(K)}\},\qquad x^{(K)}\sim\pi_1
}$$

이고, 생성 궤적이 제약을 어기면 \(\phi'_t\)가 이상 flow에서 벗어난 것으로 본다.

매 스텝 \(x^{(k)}\)를 손보는 방식(projected diffusion)은 샘플링 과정을 심하게 훼손하고 비효율적이다. GuideFlow는 **truncation 전략**으로 \(k=k_c\)에서 한 번만 개입한 뒤 계속 적분한다.

$${
x^{(k_c)}=x_1^{c},\qquad
x^{(k+1)}=x^{(k)}+v_\theta(x^{(k)},t_k)\Delta t,\quad k=k_c,\dots,K
}$$

DiffusionDrive가 학습 시 truncation을 쓰는 것과 달리 GuideFlow는 **추론에서만** 켠다. 학습은 매끄러운 조건부 flow를 배우고, 적응력은 테스트 시점에 남긴다.

### 4.3 Strategy 3 — RFE (Refining the Flow by EBM)

제약 자체를 에너지 지형에 심는다. \(t>1\) 구간에서 flow matching 모델을 EBM으로 해석하고, 에너지 대리함수를 다음과 같이 둔다.

$${
E_\theta(x_t)=\big\|\jmath(f_{t>1}(x_t))-\jmath(x_t)\big\|^2
}$$

여기서 \(f_{t>1}\)은 샘플링 연산자, \(\jmath(\cdot)\)는 제약 만족도(도로 준수, 충돌 페널티)를 평가한다. \(E_\theta\)는 feasible 궤적에 낮은 에너지를, 위반 궤적에 높은 에너지를 부여한다.

학습 목적은

$${
\mathcal{L}_{\mathrm{RFE}}=E_\theta(x^{(1)})-E_\theta(x_1)
}$$

로, 제약을 어긴 샘플의 에너지를 올리고 만족한 샘플의 에너지를 내려 속도장이 제약을 인지하게 만든다.

### 4.4 Classifier-Free Intent and Reward Guidance

조건 입력을 확률 \(p=0.2\)로 마스킹해 학습한다.

$${
h^{c}_t\leftarrow F_\theta\big(h_t,\ \mathcal{M}(C_p\oplus C_g\oplus C_d),\ \mathcal{M}(C_r)\big)
}$$

샘플링에서는 guidance scale \(\gamma\)로 조건의 영향력을 조절한다.

$${
v^{\mathrm{guide}}_\theta(x_t,t,c,\gamma)=(1-\gamma)v_\theta(x_t,t)+\gamma v_\theta(x_t,t,c)
}$$

\(\gamma=0\)이면 완전 무조건, \(\gamma=1\)이면 완전 조건부다.

### 4.5 Reward as Style Condition

aggressiveness 점수 EP(단위 시간당 차선 중심선 진행 거리, \([0,1]\))를 조건 입력으로 넣는다. 추론에서 EP를 1에 가깝게 두면 공격적 주행이 나온다. 논문 ablation에서 EP는 \(79.6\to 82.3\)으로 올랐지만 EPDMS는 0.8 하락했다. **공격성 장려는 안전 제약과 상충한다.**

---

## 5. 구현 알고리즘

### 5.1 입력

- 초기분포 \(\pi_0\), pretrained \(v_t^\theta\)
- 비용 \(C(\cdot)\), 제약 \(h(\cdot)\le 0\)
- 앵커 vocabulary \(\mathcal{V}_a\) (\(N=256\), farthest point sampling)
- CVF 계수 \(\lambda\), CF 시점 \(k_c\), 에너지 스케줄 \((\tau^{*},\varepsilon_{\max})\)
- 스텝 수 \(K\), 시간격자 \(\{t_k\}\)

### 5.2 GuideFlow 루프

1. \(x^{(0)}\sim\pi_0\)
2. for \(k=0,\dots,K-1\):
   1. \(t=t_k\), \(\Delta t=1/K\)
   2. **CF**: \(k=k_c\)이면 예측 종단의 최근접 앵커로 flow 상태 재설정
      $${
      x^{(k)}\leftarrow(1-t)x^{(0)}+t\,x_1^{c}
      }$$
   3. 속도장 계산:
      $${
      v\leftarrow v_\theta(x^{(k)},t)
      }$$
   4. **CVF**: 앵커 선택 후 속도장 보정
      $${
      x_1^{c}=\mathrm{NN}_{\mathcal{V}_a}\!\big(x^{(k)}+(1-t)v\big),\qquad
      v\leftarrow v-\frac{2\lambda\,v\cdot v^{c}}{\|v^{c}\|^2}v^{c}
      }$$
   5. Euler 전진:
      $${
      x^{(k+1)}=x^{(k)}+v\,\Delta t
      }$$
   6. **RFE**: \(\eta\leftarrow\varepsilon(t_{k+1})\), \(\eta>0\)이면
      $${
      x^{(k+1)}\leftarrow x^{(k+1)}-\eta\,\nabla_x E(x^{(k+1)})
      }$$
3. \(t\ge 1\) 정련: \(n_{\mathrm{refine}}\)회 \(x\leftarrow x-\varepsilon_{\max}\nabla_x E(x)\)
4. 출력 \(x^{(K)}\)

per-sample 비선형 solver가 필요 없다. 앵커 탐색은 \(\mathcal{V}_a\)에 대한 최근접 이웃, 에너지 기울기는 닫힌 형태다.

### 5.3 \(\alpha_t=t\), \(\beta_t=1-t\), uniform Euler일 때 단순화

\(\Delta t=1/K\)로 두면 구현이 짧아진다. 예측 종단은 posterior mean과 같다.

$${
\hat x_1=x^{(k)}+(1-t_k)v_\theta(x^{(k)},t_k)
}$$

앵커 선택과 CVF 보정:

$${
x_1^{c}=\mathrm{NN}_{\mathcal{V}_a}(\hat x_1),\qquad
v^{c}=x_1^{c}-x^{(0)},\qquad
v^{*}=v-\frac{2\lambda\,v\cdot v^{c}}{\|v^{c}\|^2}v^{c}
}$$

CF는 \(k=k_c\)에서 flow 상태를 앵커로 끝나는 경로 위로 되돌린다.

$${
x^{(k_c)}=(1-t_{k_c})x^{(0)}+t_{k_c}x_1^{c}
}$$

논문 Eq. (16)은 \(x^{(k_c)}=x_1^{c}\)로 쓰지만, 선형 보간 경로를 그대로 학습한 backbone에는 종단 스케일 상태가 분포 밖 입력이 된다. 보간 형태가 같은 앵커로 끝나면서도 경로 위에 머문다.

RFE는 \(\varepsilon(t)\) 스케줄을 따라 에너지를 하강시킨다.

$${
x^{(k+1)}\leftarrow x^{(k+1)}-\varepsilon(t_{k+1})\nabla_x E(x^{(k+1)})
}$$

마지막 스텝 이후 \(t\ge 1\)에서는 \(\varepsilon=\varepsilon_{\max}\)로 고정되므로, \(n_{\mathrm{refine}}\)회 추가 하강이 자연스럽게 이어진다.

### 5.4 에너지 설계와 기울기

\(\jmath(\cdot)\)를 제약의 제곱 hinge로 두면 에너지와 기울기가 닫힌 형태로 나온다.

$${
E(p)=\sum_j w_j\big(h_j(p)+s\big)_+^2+w_{\mathrm{cost}}\,C(p)
}$$

$${
\nabla_p E=\sum_j 2w_j\big(h_j(p)+s\big)_+\nabla_p h_j(p)+w_{\mathrm{cost}}\nabla_p C(p)
}$$

\(s\)는 경계 위 부동소수점 오차를 피하는 슬랙이다.

| 제약/비용 구조 | 추천 |
|----------------|------|
| 닫힌 형태 \(\nabla h\)가 있는 기하 제약 | 해석적 기울기. solver 불필요 |
| 미분 가능한 신경망 제약 | autograd로 \(\nabla E\) 계산 |
| 미분 불가능한 규칙 기반 제약 | 대리함수로 완화하거나 앵커 선택(CVF/CF)에만 사용 |

비용 항 \(C\)를 끄면 hinge 경계까지만 밀고, 크게 켜면 모든 점이 비용 최소점에 붙어 데이터의 자연스러운 분산이 사라진다. 데이터 노이즈 스케일 대비 작은 값이 적정이다.

하강은 정규화 좌표가 아니라 **원좌표에서** 수행한다. 정규화 스케일이 대각 metric으로 흡수되어 \(\varepsilon_{\max}\)가 데이터 스케일과 무관해진다.

### 5.5 실용 heuristic

- CVF는 매 스텝 개입하므로 확률 경로의 매끄러움을 해칠 수 있다. \(\lambda\)는 작게(0.1) 둔다.
- CF는 단 한 번만 개입한다. \(k_c\)가 너무 이르면 누적 편차를 못 잡고, 너무 늦으면 모델이 적응할 스텝이 부족하다. 중간 지점이 무난하다.
- 에너지는 종단 예측이 정확해지는 후반에만 켠다. 초반에는 posterior mean이 실제 종단과 크게 다르다.
- \(w_{\mathrm{cost}}\)가 크면 모든 점이 비용 최소점에 붙어 데이터의 분산이 사라진다. 노이즈 스케일 대비 작게 둔다.
- CFG와 EBM 결합 학습은 backbone을 새로 학습해야 하므로, training-free 비교에서는 끈다.

---

## 6. 이론적 보장 (구현 시 알아둘 점)

### 6.1 Proposition 1 — CVF의 크기 보존

\(v^{*}=v-\dfrac{2\lambda\,v\cdot v^{c}}{\|v^{c}\|^2}v^{c}\)를 \(v^{c}\) 방향 성분과 직교 성분으로 분해하면

$${
v=v_\parallel+v_\perp,\qquad v^{*}=(1-2\lambda)v_\parallel+v_\perp
}$$

따라서

$${
\|v^{*}\|^2=(1-2\lambda)^2\|v_\parallel\|^2+\|v_\perp\|^2
}$$

- \(\lambda=0\): 항등. \(v^{*}=v\)
- \(\lambda=0.5\): \(v_\parallel\) 제거. \(v^{c}\)에 직교하는 성분만 남음
- \(\lambda=1\): \(v_\parallel\) 부호 반전. **반사이므로 \(\|v^{*}\|=\|v\|\)**
- 일반적으로 크기 변화율은 \(2\lambda\)로 유계다.

논문 기본값 \(\lambda=0.1\)에서 크기 변화는 최대 20%이며, 이것이 "방향만 조정하고 크기는 최소한으로 건드린다"는 서술의 근거다.

### 6.2 Proposition 2 — Hinge 에너지 한 스텝 투영

제약이 활성인 점에서 tube 항만 보면 \(E(p)=(d-\tau+s)_+^2\), \(\nabla_p E=2(d-\tau+s)_+\hat n\) (\(\hat n\)은 매니폴드 바깥 방향 단위벡터)이다. 하강 한 스텝 후 거리는

$${
d\leftarrow d-2\eta(d-\tau+s)
}$$

이므로 잔차 \((d-\tau+s)\)는 스텝마다 \((1-2\eta)\)배가 된다.

- \(\eta=0.5\): 한 스텝에 정확히 제약 경계로 투영
- \(\eta<0.5\): 선형 수렴. \(n_{\mathrm{refine}}\)회 후 잔차는 \((1-2\eta)^{n}\)배
- \(\eta>0.5\): 진동. \(\eta\ge 1\)이면 발산

이것이 \(\varepsilon_{\max}=0.5\)를 기본값으로 두는 이유다.

### 6.3 종단 feasibility의 성격

세 전략 중 종단에서 \(h\le 0\)을 실제로 성립시키는 것은 RFE뿐이다.

- **CVF**는 방향만 조정한다. \(v^{c}\)가 feasible 종단을 향하더라도 \(\lambda<0.5\)에서는 그 성분을 줄일 뿐이고, 매 스텝 누적된 결과가 feasible set에 들어간다는 보장이 없다.
- **CF**는 \(k_c\)에서 한 번 경로를 되돌리지만, 이후 \(K-k_c\)스텝의 자유 적분이 남아 다시 이탈할 수 있다.
- **RFE**만이 종단에서 직접 \(\nabla E\)를 따라 하강하므로, §6.2의 수렴 조건 \(\eta<1\)과 충분한 \(n_{\mathrm{refine}}\) 아래에서 잔차를 0으로 보낸다.

따라서 GuideFlow의 종단 feasibility는 **이론적 명제가 아니라 수치적 결과**다. 최적화 문제를 풀어 제약을 만족시키는 방식과 달리, 에너지 하강이 충분히 수렴했다는 조건에 의존한다. \(\varepsilon_{\max}\)를 작게 잡거나 \(n_{\mathrm{refine}}\)이 부족하면 Safety가 1에 못 미친다.

대신 per-sample 최적화 solver가 없어 추론 비용이 낮고, 배치 전체를 벡터화된 연산으로 처리할 수 있다.

---

## 7.1 구현 목표

- [x] 사전학습 Flow Matching \(v_t^\theta\)를 학습 없이 불러와 샘플링한다.
- [x] 제약 만족 앵커 vocabulary \(\mathcal{V}_a\)를 FPS로 만든다. 모든 앵커가 \(h\le 0\)을 만족한다.
- [x] CVF: 예측 속도장을 제약 만족 속도장으로 반사 보정한다. 크기 변화는 \(2\lambda\)로 유계다.
- [x] CF: \(k_c\)에서 한 번만 flow 상태를 제약 만족 경로로 재설정하고 계속 적분한다. 추론에서만 켠다.
- [x] RFE: Eq. (5) 스케줄로 에너지항을 켜고 Eq. (8)로 하강한다. \(\varepsilon_{\max}=0.5\)가 한 스텝 투영이다.
- [x] 세 모듈을 각각 끌 수 있고, 전부 끄면 무제약 FlowMatch와 정확히 같은 궤적이 나온다.
- [x] CFG: Eq. (12) 마스킹 학습과 Eq. (13) guidance scale. 기본값 off로 비교 프로토콜을 유지한다.
- [x] RAS: EP를 style condition으로 주입해 추론 시점에 공격성을 조절한다.
- [x] EBM 결합 학습 \(\mathcal{L}_{\mathrm{RFE}}=E_\theta(x^{(1)})-E_\theta(x_1)\). 기본값 off, 옵션 ablation.
- [x] Exp-01에서 같은 \(x_0\) 시드·같은 평가 점으로 Safety / Tube·Core 위반 / MMD / Radius MAE / 시간을 남긴다.
- [x] Default, \(n_{\mathrm{eval}}=4000\)에서 Safety \(\ge 0.99\)를 목표로 한다.
