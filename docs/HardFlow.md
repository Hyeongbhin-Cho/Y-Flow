# HardFlow 구현 연구 정리

Hard-Constrained Sampling for Flow-Matching Models via Trajectory Optimization  
(Zeyang Li, Kaveh Alim, Navid Azizan, MIT)

본 문서는 논문 구현을 위한 **연구 목적, 이론, 알고리즘**을 정리한다.  

---

## 1. 연구 목적

Flow Matching(및 ODE 샘플링 Diffusion)은 사전분포 \(p_0\)에서 데이터 분포 \(p_1\)로 샘플을 수송한다.  
실제 응용에서는 생성된 샘플이 **반드시** 지켜야 하는 조건이 있다.

- 로봇 궤적은 장애물과 충돌하면 안 된다.
- PDE 해는 물리 경계/상태 범위를 벗어나면 안 된다.
- 이미지 편집에서는 정체성(identity)이 과도하게 바뀌면 안 된다.

기존 방법의 한계:

| 계열 | 한계 |
|------|------|
| Soft guidance (penalty, gradient) | 제약을 강하게 권장할 뿐 **보장하지 못함** |
| Path-wise projection | 중간 상태까지 feasible set에 가둬 **탐색 공간을 과도하게 줄임** |
| Post-hoc projection | 최종 한 번만 투영하면 **분포 이동이 큼** |
| PMP 기반 training-free guidance | reward 유도에는 적합하나 **hard state constraint**에 취약 |

HardFlow의 목표:

1. **Training-free**: 사전학습된 \(v_t^\theta\)는 고정. inference에서만 궤적을 조정.
2. **Terminal-only hard constraint**: 중간 경로가 아니라 **최종 샘플 \(x_N\)만** \(h(x_N)\le 0\)을 만족.
3. **Cost + distribution consistency**: 최종 비용 \(C(x_N)\)을 줄이면서, 제어량 \(\|u\|\)를 작게 유지해 nominal 분포에서 크게 벗어나지 않음.
4. ODE 샘플링이면 Flow Matching뿐 아니라 DDIM 등 Diffusion에도 적용 가능.

---

## 2. 배경: Flow Matching

시간 \(t\in[0,1]\)에서 상태 \(X_t\in\mathbb{R}^d\), 주변 속도장 \(v_t\).  
Flow map은 ODE

$${
\frac{d}{d\tau}\Phi_{s\to\tau}(x)=v_\tau(\Phi_{s\to\tau}(x)),\qquad \Phi_{s\to s}(x)=x
}$$

를 따른다. \(x_0\sim p_0\)를 적분하면 \(x_1=\Phi_{0\to 1}(x_0)\sim p_1\).

조건 경로의 전형적인 affine 형태:

$${
X_t\mid Z=\alpha_t X_1+\beta_t X_0,\qquad Z=(X_0,X_1)
}$$

스케줄러 경계조건:

$${
\alpha_0=0,\ \alpha_1=1,\ \beta_0=1,\ \beta_1=0
}$$

논문 실험의 기본 스케줄:

$${
\alpha_t=t,\qquad \beta_t=1-t
}$$

조건 속도:

$${
v_{t\mid Z}(X_t\mid Z)=\dot\alpha_t X_1+\dot\beta_t X_0
}$$

\(Z\)는 가우시안일 필요가 없다. 일반적으로 **coupling** \(\pi_{0,1}\) 위의 쌍 \((X_0,X_1)\)이다.

---

## 3. 문제 정의

### 3.1 Nominal distribution

고정된 pretrained 모델 \(v_t^\theta\)와 초기분포 \(p_0\)에 대해, 제어 없이 적분한 최종 분포를 nominal distribution \(\bar\mu\)라 한다.

$${
\bar\mu=(\Phi^\theta_{0\to 1})_{\#}p_0
}$$

### 3.2 Hard constraint와 cost

- Hard constraint: \(h(x)\le 0\). 반드시 만족. 벡터면 성분별 만족.
- Equality \(g(x)=0\)은 \(g\le 0\)과 \(-g\le 0\)으로 분해.
- Cost \(C(x)\): 작을수록 좋지만 제약보다 우선순위가 낮음.
- 조정된 최종 분포 \(\tilde\mu\)는 \(\bar\mu\)와 가까워야 함.

### 3.3 Control input

속도장에 제어 \(u_t\)를 더한다.

$${
\dot x_t=v_t^\theta(x_t)+u_t
}$$

중간 상태 \(x_t\) 자체가 feasible일 필요는 없다.  
최종 \(x_1\)만 \(h(x_1)\le 0\)이면 된다.

---

## 4. 이론: 문제 변환 체인

### 4.1 Problem 1 — 연속시간 trajectory optimization

주어진 \(\bar x_0\sim p_0\),

$${
\begin{aligned}
\min_{\{x_t,u_t\}_{t\in[0,1]}}\quad
& C(x_1)+\lambda_{oc}\int_0^1\frac12\|u_t\|_2^2\,dt \\
\text{s.t.}\quad
& x_0=\bar x_0,\\
& \dot x_t=v_t^\theta(x_t)+u_t,\\
& h(x_1)\le 0.
\end{aligned}
}$$

의미:

- 항 \(C(x_1)\): 최종 샘플 품질
- 항 \(\|u_t\|^2\): 원래 sampler에서 벗어나는 정도(분포 이동) 억제
- 제약: 초기조건 + 동역학 + **터미널 hard constraint**

구현상 연속시간 문제를 직접 풀 수 없으므로 이산화한다.

### 4.2 Problem 2 — Forward Euler 이산화

시간격자 \(0=t_0<\cdots<t_N=1\), \(\Delta t_j=t_{j+1}-t_j\).

$${
\begin{aligned}
\min_{\{x_j\}_{j=0}^N,\{u_j\}_{j=0}^{N-1}}\quad
& C(x_N)+\lambda_{oc}\sum_{j=0}^{N-1}\frac12\|u_j\|_2^2\Delta t_j \\
\text{s.t.}\quad
& x_0=\bar x_0,\\
& x_{j+1}=x_j+v_{t_j}^\theta(x_j)\Delta t_j+u_j\Delta t_j,\\
& h(x_N)\le 0.
\end{aligned}
}$$

결정변수 개수는 대략 \(2Nd+d\).  
고차원(이미지) 또는 큰 \(N\)에서는 한 번에 풀기 어렵다.  
또한 터미널 제약이 neural dynamics를 통해 뒤로 전파되어 feasible set이 매우 복잡해진다.

### 4.3 Lemma 1 — Posterior mean (구현 핵심)

Affine path와 \(\Lambda_t=\alpha_t\dot\beta_t-\dot\alpha_t\beta_t\neq 0\)일 때,

$${
\mathcal{M}_t(x)=\mathbb{E}[X_1\mid X_t=x]=\frac{\dot\beta_t x-\beta_t v_t(x)}{\Lambda_t}
}$$

$${
\mathcal{W}_t(x)=\mathbb{E}[X_0\mid X_t=x]=\frac{-\dot\alpha_t x+\alpha_t v_t(x)}{\Lambda_t}
}$$

항등식:

$${
x=\alpha_t\mathcal{M}_t(x)+\beta_t\mathcal{W}_t(x)
}$$

구현에서는 \(v_t\) 대신 learned field \(v_t^\theta\)를 넣는다.

$${
\mathcal{M}_t^\theta(x)=\frac{\dot\beta_t x-\beta_t v_t^\theta(x)}{\Lambda_t}
}$$

표준 경계에서 \(\mathcal{M}_1^\theta(x)=x\).

\(\alpha_t=t\), \(\beta_t=1-t\)이면 \(\dot\alpha_t=1\), \(\dot\beta_t=-1\), \(\Lambda_t=-1\)이므로

$${
\mathcal{M}_t^\theta(x)=x+(1-t)v_t^\theta(x)
}$$

$${
\mathcal{W}_t^\theta(x)=x-t v_t^\theta(x)
}$$

즉 posterior mean은 “현재에서 남은 시간만큼 속도를 적분한 최종 예측”이다.

### 4.4 Problem 3 — MPC (receding horizon)

전체 horizon을 한 번에 풀지 않고, 매 스텝에서 **현재 제어만** 최적화한다.  
미래 제어는 0으로 가정하고, 터미널 상태 대신 posterior mean을 proxy로 쓴다.

\(i=0,\dots,N-1\):

$${
\begin{aligned}
u_i^*=\arg\min_{u_i}\quad
& C(\hat x_N)+\frac{\lambda_{oc}}{2}\|u_i\|_2^2\Delta t_i \\
\text{s.t.}\quad
& x_{i+1}=x_i+v_{t_i}^\theta(x_i)\Delta t_i+u_i\Delta t_i,\\
& \hat x_N=\mathcal{M}_{t_{i+1}}^\theta(x_{i+1}),\\
& h(\hat x_N)\le 0.
\end{aligned}
}$$

그 다음 \(x_{i+1}\)을 갱신하고 다음 스텝으로 간다.

근사 오차의 두 원인:

1. \(\mathcal{M}\)이 진짜 ODE 적분 결과가 아님
2. 미래 \(\{u_j\}_{j>i}\)를 무시

### 4.5 Problem 4 — 제어를 다음 상태로 치환

$${
\bar x_{i+1}=x_i+v_{t_i}^\theta(x_i)\Delta t_i
}$$

$${
u_i=\frac{x_{i+1}-\bar x_{i+1}}{\Delta t_i}
}$$

이므로 목적함수의 제어 항은 \(\|x_{i+1}-\bar x_{i+1}\|^2\)가 된다.

문제는 결정변수가 \(x_{i+1}\)인데 제약/비용은 \(\mathcal{M}(x_{i+1})\)에 걸린다는 점이다.  
feasible set

$${
\{x_{i+1}\mid h(\mathcal{M}_{t_{i+1}}^\theta(x_{i+1}))\le 0\}
}$$

은 신경망 때문에 매우 꼬여 있다.

### 4.6 Problem 5 — 역재매개변수화

결정변수를 \(\hat x_N\)으로 바꾼다.

$${
x_{i+1}=(\mathcal{M}_{t_{i+1}}^\theta)^{-1}(\hat x_N)
}$$

제약 \(h(\hat x_N)\le 0\)은 원래 집합 그대로라 solver가 다루기 쉽다.

역함수는 고정점 문제로 쓴다. \(y=\mathcal{M}_t(x)\)이면

$${
T_t^y(x)=\alpha_t y+\beta_t\mathcal{W}_t^\theta(x)
}$$

의 고정점이 \(x=(\mathcal{M}_t^\theta)^{-1}(y)\)이다.

$${
(\mathcal{M}_t^\theta)^{-1}(y)=\lim_{k\to\infty}(T_t^y)^{\circ k}(x^{(0)})
}$$

### 4.7 Problem 6 — 한 스텝 고정점 근사 (실제 구현)

초기값을 nominal next state \(\bar x_{i+1}\)로 두고 **한 번만** 적용:

$${
\mathcal{F}_{t_{i+1}}(y)=\alpha_{t_{i+1}} y+\beta_{t_{i+1}}\mathcal{W}_{t_{i+1}}^\theta(\bar x_{i+1})
}$$

이 맵은 \(y\)에 대해 affine이므로 목적함수를 \(\hat x_N\)만으로 쓸 수 있다.

$${
\|\mathcal{F}_{t_{i+1}}(\hat x_N)-\bar x_{i+1}\|_2^2=\alpha_{t_{i+1}}^2\|\hat x_N-\bar x_N\|_2^2
}$$

여기서 \(\bar x_N=\mathcal{M}_{t_{i+1}}^\theta(\bar x_{i+1})\).

최종 한 스텝 문제:

$${
\begin{aligned}
\hat x_N^*=\arg\min_{\hat x_N}\quad
& C(\hat x_N)+\frac{\lambda_{oc}}{2\Delta t_i}\alpha_{t_{i+1}}^2\|\hat x_N-\bar x_N\|_2^2 \\
\text{s.t.}\quad
& h(\hat x_N)\le 0.
\end{aligned}
}$$

다음 상태:

$${
x_{i+1}=\mathcal{F}_{t_{i+1}}(\hat x_N^*)
}$$

중요: 여러 근사를 했지만 **터미널 feasibility는 완화하지 않는다**.

---

## 5. 구현 알고리즘

### 5.1 입력

- 초기분포 \(p_0\), pretrained \(v_t^\theta\)
- 비용 \(C(\cdot)\), 제약 \(h(\cdot)\le 0\)
- regularization \(\lambda_{oc}>0\)
- 스텝 수 \(N\), 시간격자 \(\{t_i\}\)
- 스케줄 \((\alpha_t,\beta_t)\)

### 5.2 HardFlow 루프

1. \(\bar x_0\sim p_0\), \(x_0\leftarrow\bar x_0\)
2. for \(i=0,\dots,N-1\):
   1. \(\Delta t_i=t_{i+1}-t_i\)
   2. nominal next:
      $${
      \bar x_{i+1}=x_i+v_{t_i}^\theta(x_i)\Delta t_i
      }$$
   3. terminal prediction:
      $${
      \bar x_N=\frac{\dot\beta_{t_{i+1}}\bar x_{i+1}-\beta_{t_{i+1}}v_{t_{i+1}}^\theta(\bar x_{i+1})}{\alpha_{t_{i+1}}\dot\beta_{t_{i+1}}-\dot\alpha_{t_{i+1}}\beta_{t_{i+1}}}
      }$$
   4. constrained optimization으로 \(\hat x_N^*\) 계산 (위 Problem 6)
   5. 되돌리기:
      $${
      x_{i+1}=\alpha_{t_{i+1}}\hat x_N^*+\beta_{t_{i+1}}\frac{-\dot\alpha_{t_{i+1}}\bar x_{i+1}+\alpha_{t_{i+1}}v_{t_{i+1}}^\theta(\bar x_{i+1})}{\alpha_{t_{i+1}}\dot\beta_{t_{i+1}}-\dot\alpha_{t_{i+1}}\beta_{t_{i+1}}}
      }$$
3. 출력 \(x_N\)

\(u_i\)를 명시적으로 저장할 필요는 없다.  
필요하면

$${
u_i=\frac{x_{i+1}-\bar x_{i+1}}{\Delta t_i}
}$$

### 5.3 \(\alpha_t=t\), \(\beta_t=1-t\), uniform Euler일 때 단순화

\(\Delta t=1/N\)로 두면 구현이 짧아진다.

$${
\bar x_{i+1}=x_i+v_{t_i}^\theta(x_i)\Delta t
}$$

$${
\bar x_N=\bar x_{i+1}+(1-t_{i+1})v_{t_{i+1}}^\theta(\bar x_{i+1})
}$$

$${
\hat x_N^*=\arg\min_{\hat x_N}\ C(\hat x_N)+\frac{\lambda_{oc}}{2\Delta t}t_{i+1}^2\|\hat x_N-\bar x_N\|_2^2
\quad\text{s.t.}\quad h(\hat x_N)\le 0
}$$

$${
x_{i+1}=t_{i+1}\hat x_N^*+(1-t_{i+1})\big(\bar x_{i+1}-t_{i+1}v_{t_{i+1}}^\theta(\bar x_{i+1})\big)
}$$

마지막 스텝 \(t_N=1\)에서는 \(\alpha_1=1\), \(\beta_1=0\)이므로 \(x_N=\hat x_N^*\).  
따라서 \(h(x_N)\le 0\)이 그대로 보장된다.

### 5.4 내부 최적화 solver

| 제약/비용 구조 | 추천 |
|----------------|------|
| 이차 비용 + 선형/박스 제약 | QP, 가능하면 닫힌 해 |
| 일반 비선형 (로봇, maze, PDE) | IPOPT, SQP, interior-point |
| 신경망 제약 (LPIPS) + 고차원 | Augmented Lagrangian, 고정 iteration |

\(C\)와 \(h\)는 일반적으로 미분 가능해야 gradient-based solver를 쓸 수 있다.  
집합 \(\mathcal{S}=\{x\mid h(x)\le 0\}\)이 본질이고, \(h\le 0\)은 solver가 다루는 표준 표현이다.

### 5.5 실용 heuristic

이론적으로 posterior mean과 고정점 근사는 \(t\to 1\)에서 더 정확하다 (\(\beta_t\to 0\)).  
따라서 **초반 스텝은 제어를 끄고, 후반에 HardFlow를 켜는** 스케줄이 안정적이다.

구현 옵션:

- `activate_from_t`: 예) \(t\ge 0.5\)부터 최적화
- 초반에는 \(\lambda_{oc}\)를 매우 크게 (거의 nominal)
- 후반에는 제약을 엄격히

---

## 6. 이론적 보장 (구현 시 알아둘 점)

### 6.1 Proposition 1 — Terminal feasibility

feasible set이 비어 있지 않고 마지막 서브문제가 해를 가지면,

$${
h(x_N)\le 0
}$$

이 항상 성립한다. 경로 중간을 투영하지 않아도 최종 제약은 유지된다.

### 6.2 Theorem 1 — MPC 근사 오차

Problem 2(전체 horizon)와 Problem 3(MPC)의 목적함수 차이:

$${
0\le J(x_0,u^{P3})-J(x_0,u^{P2})\le 2\sum_{i=1}^{N-1}(L_C\varepsilon_i+\Gamma_i\Delta t_i)
}$$

- \(\varepsilon_i\): 한 스텝 posterior consistency error
- \(\Gamma_i\): 미래 제어 무시 + feasible set으로 보내는 제어 비용

### 6.3 Theorem 2

Problem 3, 4, 5는 변수 변환만 다르고 같은 궤적을 만든다.

### 6.4 Theorem 3 — 한 스텝 고정점 오차

\(r=|\beta_{t_{i+1}}|L_{\mathcal{W},i+1}<1\)이면 Problem 5와 6의 목적함수 차이는

$${
\big|J_i^{P5}(y)-J_i^{P6}(y)\big|
\le
\frac{\lambda_{oc}}{2\Delta t_i}\frac{r(2-r)}{(1-r)^2}\alpha_{t_{i+1}}^2\|y-\bar y\|_2
}$$

후반부일수록 \(r\)이 작아져 근사가 좋아진다.

---

---

## 7.1 구현 목표

- [x] 사전학습 Flow Matching $v_t^\theta$를 학습 없이 불러와 샘플링한다.
- [x] 매 스텝에서 예측된 최종 상태 $\hat x_N$에만 $h\le 0$과 비용 $C$를 적용한다. 중간 $x_t$는 feasible일 필요가 없다.
- [x] 제어량(nominal sampler와의 차이)을 작게 유지해 분포가 크게 밀리지 않게 한다.
- [x] 마지막 스텝에서 $x_N=\hat x_N^*$이면 $h(x_N)\le 0$이 성립한다.
- [x] 초반 스텝은 제어를 끄고, $t\ge t_{\mathrm{on}}$에서만 제약을 켠다.
- [x] Exp-01에서 같은 $x_0$ 시드·같은 평가 점으로 Safety / Tube·Core 위반 / MMD / 시간을 남긴다.
- [x] Default, $n_{\mathrm{eval}}=4000$에서 Safety $\ge 0.99$를 목표로 한다.
