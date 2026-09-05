# YFlow 구현 연구 정리

Physical Guidance Flow Matching (YFlow) + HardFlow-style terminal constraint

본 문서는 기존 YFlow를 HardFlow의 일부 아이디어로 개선한 **구현용 이론·알고리즘** 정리이다.  


---

## 1. 연구 목적

Flow Matching 생성 과정에서 **물리 법칙을 엄격히 만족**시키고 hallucination을 줄이는 것이 목표다.

기존 YFlow:

1. 현재 $x_t$와 속도 $v_t$로 clean target $\hat x_1$을 먼저 예측
2. 물리 연산자 $P(\cdot)$를 타깃 공간에 적용
3. 현재 상태와 가이드된 타깃 사이를 **선형 보간**하여 다음 스텝으로 이동

기존 YFlow의 한계:

- $P$는 단순 투영이라 **비용 $C$와 하드 제약 $h$를 동시에** 다루기 어렵다.
- 매 스텝 무조건 투영하면 초반 노이즈 구간에 과도한 왜곡이 생길 수 있다.
- LUCID 대비 bias는 단순하지만, 제약이 “연산자 한 번”으로만 표현된다.

HardFlow에서 가져올 점:

- 제약은 **예측된 최종 상태**에만 건다.
- $C(\hat x_1)$와 $h(\hat x_1)\le 0$을 한 스텝 최적화로 푼다.
- 초반에는 제약을 약하게, $t\to 1$ 또는 $L_P\le 1$이 안정될 때 강하게 건다.

유지할 YFlow 장점:

- HardFlow의 $\mathcal{M}^{-1}$ 고정점 반복 대신 **선형 보간**으로 다음 상태를 계산
- 물리 연산자 $P$의 1-Lipschitz / 정답 보존성 분석을 그대로 활용

---

## 2. 기존 YFlow 요약

### 2.1 물리 연산자 $P$

가정:

1. 1-Lipschitz
$${
\|P(x)-P(y)\|\le L_P\|x-y\|,\qquad L_P\le 1
}$$

2. 양의 동차성 (해당되는 경우)
$${
P(\alpha x)=\alpha P(x)\quad(\alpha\ge 0)
}$$

3. 정답 보존
$${
P(X_1)=X_1
}$$

### 2.2 기존 속도장

선형 경로 $X^{\mathrm{true}}(t)=(1-t)X_0+t X_1$에 오차 $e(t)$를 더한다.

$${
X_t=(1-t)X_0+t X_1+e_{\mathrm{YFlow}}(t)
}$$

타깃 예측:

$${
X_t+r_t V_t,\qquad r_t=1-t
}$$

이상적인 rectified path에서는

$${
X_t+r_t V_t \approx X_1+e_{\mathrm{YFlow}}(t)+\text{(noise)}
}$$

기존 가이던스:

$${
V_{\mathrm{YFlow}}(t)=P(X_t+r_t V_t)-X_t
}$$

다음 상태(이산):

$${
x_{t+\Delta t}=x_t+\Delta t\,V_{\mathrm{YFlow}}
}$$

또는 동등하게 타깃과 현재를 보간한다.

### 2.3 LUCID와의 차이 (유지)

| 항목 | YFlow | LUCID |
|------|-----|-------|
| 연산자 입력 | $X_t+r_t V_t\approx X_1+e$ | $r_t X_t+r_t V_t$ ( $X_0,X_1,t^2$가 얽힘) |
| 전개 중심 | $P(X_1)=X_1$ | $X_1$ 주변의 복잡한 섭동 |
| Bias | $\frac12 t^2(X_0-X_1)$ | $\exp(-\frac12 t^2 J)$가 섞인 적분 |

YFlow는 타깃을 $X_1$ 근처로 바로 정렬하므로 drift가 단순하다.  
Improved YFlow도 이 정렬 구조를 유지한다.

---

## 3. Improved YFlow 문제 정의

매 스텝에서 예측된 최종 상태 $\hat x_1$이 다음을 만족하도록 조정한다.

1. Hard constraint
$${
h(\hat x_1)\le 0
}$$
중간 $x_t$는 feasible일 필요 없음.

2. Cost
$${
C(\hat x_1)
}$$
작을수록 좋지만 제약보다 우선순위가 낮음.

3. 물리 연산자 정합 (optional)
$${
\hat x_1 \approx P(\hat x_1)\quad\text{또는}\quad \|\hat x_1-P(\hat x_1^{\mathrm{raw}})\| \text{ 제한}
}$$

4. Nominal에서 크게 벗어나지 않음
$${
\|\hat x_1-\hat x_1^{\mathrm{raw}}\|_2^2
}$$

다음 상태로의 복원은 HardFlow inverse가 아니라 **선형 보간**.

### 3.1 $h(x) \le 0$과 $P(x)$의 차이점 및 도메인별 예시

- **$h(x) \le 0$ (Hard Constraint / Feasible Set)**:
  - 시스템/도메인이 허용하는 **실행 가능 영역(Feasible Region)** $\mathcal{S} = \{x \mid h(x) \le 0\}$을 규정한다.
  - 경계(boundary) 및 안전 영역의 개념으로, 제약 조건을 만족하는 영역 내부라면 어디에 있든 동등하게 feasible한 것으로 간주된다.
- **$P(x)$ (Physical Projection Operator / Manifold Prior)**:
  - 임의의 상태를 물리 법칙이나 부분다양체(Submanifold) 상의 **이상적인 상태(Ideal Target Point)**로 사영하는 **점 대 점 매핑 연산자(Mapping Operator)**이다.
  - 최적화 서브문제에서 좋은 초기점을 제공하는 **Warm Start**이자, 엉뚱한 해로 튀지 않도록 잡아주는 **정규화 유도 항(Loss Regularizer)** 역할을 한다.

#### 도메인별 구체적 예시

| 도메인 | 물리 연산자 $P(x)$ (이상적 상태로의 사영) | Hard 제약 $h(x) \le 0$ (안전/허용 영역) |
| :--- | :--- | :--- |
| **Swiss Roll (Exp-01)** | 1D 중심 나선 곡선으로의 수직 투영 $P(x) = \Pi_{\mathcal{M}}(x)$ | 튜브 허용 반경 ($d_{\mathcal{M}}(x) - \tau \le 0$), 코어 회피 ($\rho_{\min} - \|x\|_2 \le 0$), 바운딩 박스 ($\|x\|_\infty - R \le 0$) |
| **로보틱스 / 궤적 계획** | 역기구학(IK) 해석적 투영 또는 매니폴드 사영 | 장애물 충돌 회피 ($r_{\mathrm{safe}} - d_{\mathrm{obs}}(x) \le 0$), 관절 각도/속도/가속도 한계 |
| **유체역학 / SciML (PDE)** | Helmholtz-Hodge 분해를 통한 발산 자유 속도장 투영 ($P(v) = v - \nabla \phi$) | 질량/밀도 비음수성 ($-\rho \le 0$), 물리 경계 조건 허용 오차 ($\|u_{\mathrm{bd}} - u_{\mathrm{target}}\| - \epsilon \le 0$) |
| **분자 / 단백질 생성** | 1-step force-field energy relaxation (물리적 완화) | 원자 간 steric clash 방지 ($r_{\mathrm{vdw}} - \|r_i - r_j\|_2 \le 0$) |

---

## 4. 한 스텝 공식

시간격자 $0=t_0<\cdots<t_N=1$.  
현재 상태 $x_i$, 시각 $t_i$.

### 4.1 Raw terminal prediction

YFlow 방식 (rectified / linear schedule):

$${
\hat x_1^{\mathrm{raw}}=x_i+(1-t_i)v_{t_i}^\theta(x_i)
}$$

HardFlow $\mathcal{M}$과 $\alpha_t=t$, $\beta_t=1-t$일 때 같은 식이다.

$${
\mathcal{M}_{t_i}^\theta(x_i)=x_i+(1-t_i)v_{t_i}^\theta(x_i)
}$$

즉 Improved YFlow의 raw target은 HardFlow posterior mean과 같다.

### 4.2 물리 연산자 warm start (선택)

단순 YFlow는 여기서 $P(\hat x_1^{\mathrm{raw}})$를 바로 쓴다.  
Improved YFlow는 이를 초기값으로만 쓴다.

$${
\hat x_1^{(0)}=P(\hat x_1^{\mathrm{raw}})
}$$

### 4.3 Terminal 최적화 (HardFlow 차용)

$${
\begin{aligned}
\hat x_1^*=\arg\min_{\hat x_1}\quad
& C(\hat x_1)+\frac{\lambda}{2}\|\hat x_1-\hat x_1^{\mathrm{raw}}\|_2^2
+\frac{\mu}{2}\|\hat x_1-P(\hat x_1^{\mathrm{raw}})\|_2^2 \\
\text{s.t.}\quad
& h(\hat x_1)\le 0.
\end{aligned}
}$$

하이퍼파라미터:

- $\lambda$: raw flow prediction에 붙는 정도 (HardFlow의 $\lambda_{oc}$에 해당)
- $\mu$: 물리 투영 근처에 머물게 하는 항. $\mu=0$이면 HardFlow형, $\lambda=0,\mu\to\infty$면 기존 YFlow형
- 제약만 필요하면 $C\equiv 0$

등식 물리 조건은 $h$에 넣어도 된다.

$${
g(\hat x_1)=0 \quad\Rightarrow\quad \pm g(\hat x_1)\le\epsilon
}$$

#### 4.3.1 이상적인 물리 연산자 $P$의 부재 시 대처 ($\mu = 0$ 설정)

자연어, 고해상도 자연 이미지처럼 데이터 매니폴드가 매우 복잡하고 고차원 비선형 공간에 있어 명시적인 물리 투영 연산자 $P(\cdot)$를 수학적으로 정의하기 어렵거나 계산 불가능한 경우가 있다.

이때는 **$\mu = 0$**으로 설정하여 $P$ 의존성을 완전히 제거한다:

1. **HardFlow 형태로의 자연스러운 퇴화(환원)**:
   $\mu = 0$으로 두면 최적화 목적함수는 다음과 같이 순수 제약 최적화 문제로 바뀐다:
   $${
   \hat x_1^* = \arg\min_{\hat x_1}\quad C(\hat x_1) + \frac{\lambda}{2}\|\hat x_1 - \hat x_1^{\mathrm{raw}}\|_2^2 \quad \text{s.t.} \quad h(\hat x_1) \le 0
   }$$
   이는 HardFlow의 Problem 6 목적함수와 완벽히 일치한다.
2. **Warm Start 및 스케줄링 간소화**:
   - Warm start는 $P(\hat x_1^{\mathrm{raw}})$ 대신 $\hat x_1^{(0)} = \hat x_1^{\mathrm{raw}}$(또는 feasible projection)를 그대로 사용한다.
   - $P$ 연산자가 없으므로 국소 립시츠 추적($\widehat L_P$) 단계를 생략하고, 시간 기반 스케줄($t \ge t_{\mathrm{on}}$)만으로 제약 활성화를 제어한다.
3. **도메인별 유연성**:
   - **물리적/기하학적 사전 지식 $P$가 존재하는 도메인(Swiss roll, 유체역학 등)**: $\mu > 0$으로 두어 warm start 및 정규화 효과로 최적화 속도와 수렴 안정성을 극대화한다.
   - **명시적 $P$가 없고 제약 $h$만 정의된 일반 도메인**: $\mu = 0$으로 두어 순수 terminal constraint optimization으로 안전성을 보장한다.

### 4.4 Lipschitz 상수 추적 및 적응형 스케줄 (Lipschitz Gating & Scheduling)

물리 투영 연산자 $P(\cdot)$의 안정성을 보장하기 위해, 매 스텝에서 **국소 립시츠 상수 $\widehat{L}_P$를 추적**하고 이를 기반으로 제약 강도 $\gamma(t)$를 조절하는 게이팅(gating)을 적용한다.

#### 4.4.1 립시츠 상수 추적의 필요성

1. **매니폴드 위에서의 비확장성**:
   매니폴드 근방의 정상적인 데이터 분포에서는 투영 연산자 $P$가 1-Lipschitz ($L_P \le 1$) 성질을 만족하여 수축/비확장 매핑으로 동작한다.
2. **초기 노이즈 및 팔 사이(gap)에서의 불안정성**:
   - $t < t_{\mathrm{on}}$인 초기 확산 구간이나 나선 팔 사이(arm gap)와 같은 결정 경계 부근에서는, 미세한 변위 $\epsilon$만으로도 투영 대상이 인접한 다른 나선 팔로 불연속하게 도약(jump)한다.
   - 이 경우 $\widehat{L}_P \gg 1$로 폭증하며, 이 불안정 구간에서 무리하게 투영 $P$를 강제하면 궤적이 급격히 꺾이거나 환각(hallucination) 및 엉뚱한 모드로의 왜곡이 발생한다.
3. **해결책 (Lipschitz Gating)**:
   $\widehat{L}_P \le 1 + \delta$로 안정성이 확인된 샘플 및 영역에서만 제약을 활성화하고, $\widehat{L}_P > 1 + \delta$인 불안정 영역에서는 순수 nominal flow의 방향을 보존한다.

#### 4.4.2 국소 립시츠 상수 $\widehat{L}_P$ 추정

각 점 $p \in \mathbb{R}^D$에 대해 기저 축 방향의 미세 섭동 벡터 $d \in \{\pm \mathbf{e}_1, \dots, \pm \mathbf{e}_D\}$과 미세 스텝 $\epsilon = 10^{-4}$을 적용하여 방향별 변화율의 최대치를 계산한다.

$${
\widehat{L}_P(p) = \max_{d \in \{\pm \mathbf{e}_1, \dots, \pm \mathbf{e}_D\}} \frac{\|P(p + \epsilon d) - P(p)\|_2}{\epsilon}
}$$

이 방식은 행렬 야코비안을 직접 계산하지 않고도 $2D$회의 $P$ 평가만으로 빠르고 정확하게 국소 립시츠 상수를 추정할 수 있다.

#### 4.4.3 적응형 제약 스케줄 $\gamma(t, \widehat{L}_P)$

시간 $t$와 추정 립시츠 상수 $\widehat{L}_P$에 따른 제약 게이트:

$${
\gamma(t, \widehat{L}_P) = 
\begin{cases}
0, & t < t_{\mathrm{on}}\ \text{또는}\ \widehat{L}_P > 1 + \delta \\
\gamma_{\max} \cdot \frac{t - t_{\mathrm{on}}}{1 - t_{\mathrm{on}}}, & \text{otherwise}
\end{cases}
}$$

동작 방식:

- **$\gamma = 0$ (순수 Flow 보존)**:
  $t < t_{\mathrm{on}}$이거나 $\widehat{L}_P > 1 + \delta$인 불안정 영역에서는 $\hat{x}_1^* \leftarrow \hat{x}_1^{\mathrm{raw}}$로 두어 순수 Flow Matching 궤적을 그대로 따른다.
- **$0 < \gamma < 1$ (점진적 가이던스)**:
  안정 영역에서는 $\mu_{\mathrm{eff}} = \mu \cdot \gamma$ 및 $\text{max\_iter} = \max(1, \lfloor \text{max\_iter} \cdot \gamma \rfloor)$로 soft하게 최적화하여 궤적의 급격한 불연속성을 방지한다.
- **마지막 스텝 ($t_{N-1} = 1 - \Delta t$)**:
  터미널 hard constraint ($h(x_N) \le 0$) 보장을 위해 모든 점에 엄격한 최적화를 적용한다.

### 4.5 다음 상태: 선형 보간 (YFlow 유지)

HardFlow:

$${
x_{i+1}=\alpha_{t_{i+1}}\hat x_N^*+\beta_{t_{i+1}}\mathcal{W}(\bar x_{i+1})
}$$

Improved YFlow:

$${
x_{i+1}=(1-\eta_i)x_i+\eta_i\hat x_1^*
}$$

권장 스텝:

$${
\eta_i=\frac{\Delta t_i}{1-t_i}
}$$

이면

$${
x_{i+1}=x_i+\Delta t_i\cdot\frac{\hat x_1^*-x_i}{1-t_i}
}$$

유효 속도:

$${
V_{\mathrm{imp}}(t_i)=\frac{\hat x_1^*-x_i}{1-t_i}
}$$

기존 YFlow의 $V_{\mathrm{YFlow}}=P(\cdot)-x_t$를 $\hat x_1^*-x_t$로 바꾼 것이다.

마지막 스텝 $t_N=1$ 직전에는 $\eta\to 1$이므로 $x_N=\hat x_1^*$.  
최적화가 성공하면 $h(x_N)\le 0$.

---

## 5. 구현 알고리즘

### 입력

- pretrained $v_t^\theta$
- 물리 연산자 $P$ (투영 또는 근사 투영)
- $C(\cdot)$, $h(\cdot)$
- $\lambda,\mu,\lambda_{oc}$ 대응 계수
- $N$, $\{t_i\}$, $t_{\mathrm{on}}$, Lipschitz threshold

### 루프

1. $x_0\sim p_0$
2. for $i=0,\dots,N-1$:
   1. $v\leftarrow v_{t_i}^\theta(x_i)$
   2. $\hat x_1^{\mathrm{raw}}\leftarrow x_i+(1-t_i)v$
   3. (선택) $\hat x_1^{(0)}\leftarrow P(\hat x_1^{\mathrm{raw}})$
   4. $\gamma\leftarrow \mathrm{Schedule}(t_i,\widehat L_P)$
   5. if $\gamma=0$: $\hat x_1^*\leftarrow\hat x_1^{\mathrm{raw}}$
      else: 4.3의 constrained optimization (초기값 $\hat x_1^{(0)}$ 또는 raw)
   6. $\eta\leftarrow \Delta t_i/(1-t_i)$ (마지막 스텝은 $\eta=1$)
   7. $x_{i+1}\leftarrow (1-\eta)x_i+\eta\hat x_1^*$
3. return $x_N$

### 의사코드

```text
x = sample_p0()
for i in 0..N-1:
    t, dt = t_grid[i], t_grid[i+1]-t_grid[i]
    v = model(x, t)
    x1_raw = x + (1-t)*v
    x1_phys = P(x1_raw)
    gamma = constraint_schedule(t, lipschitz_estimate(P, x1_raw))
    if gamma == 0:
        x1 = x1_raw
    else:
        x1 = solve(
            min  C(z) + (lambda/2)*||z-x1_raw||^2 + (mu/2)*||z-x1_phys||^2
            s.t. h(z) <= 0,
            z0 = x1_phys
        )
    eta = 1.0 if i == N-1 else dt/(1-t)
    x = (1-eta)*x + eta*x1
return x
```

---

## 6. 이론 스케치 (구현 관점)

### 6.1 Terminal feasibility

마지막 보간이 $\eta=1$이면 $x_N=\hat x_1^*$.  
서브문제가 실행 가능하면 $h(x_N)\le 0$.  
이는 HardFlow Proposition 1과 같은 논리이며, inverse map이 없어도 성립한다.

### 6.2 기존 YFlow bias와의 관계

제약을 켜지 않으면($\gamma=0$ 또는 $\hat x_1^*=P(\hat x_1^{\mathrm{raw}})$만 사용) 기존 YFlow와 동일하다.  
$L_P\le 1$, $P(X_1)=X_1$, $(J-I)e\approx 0$ 근사에서

$${
\mathrm{Bias}(e_{\mathrm{YFlow}}(t))\approx\frac12 t^2(X_0-X_1)
}$$

$${
\mathrm{Var}(e_{\mathrm{YFlow}}(t))
=\sigma_1^2 t\,\mathrm{Tr}(I)+\sigma_2^2\Big(t-t^2+\frac13 t^3\Big)\mathrm{Tr}(J^2)
}$$

최적화가 raw target에서 $\Delta=\hat x_1^*-\hat x_1^{\mathrm{raw}}$만큼 옮기면, 추가 drift는 $\Delta$의 보간으로 들어간다.  
$\lambda$가 크면 $\Delta$가 작아져 YFlow 분석에 가깝고, 제약이 빡세면 $\Delta$는 feasible set으로의 최소 이동량이 된다.

### 6.3 선형 보간 vs HardFlow $\mathcal{F}$

HardFlow $\mathcal{F}$는 $\mathcal{W}(\bar x_{i+1})$을 한 번 평가해 $X_0$ 추정을 섞는다.  
Improved YFlow는 $X_0$ 추정을 쓰지 않고

$${
x_{i+1}=x_i+\eta(\hat x_1^*-x_i)
}$$

만 사용한다.

장점:

- $\mathcal{M}^{-1}$ 고정점 / $\mathcal{W}$ 추가 forward가 없음
- 구현이 짧고 $P$ 분석과 맞추기 쉬움

단점:

- affine path의 정확한 역맵은 아님
- 스케줄이 $\alpha_t=t$가 아니면 보간 계수를 일반화해야 함

일반 스케줄:

$${
\hat x_1^{\mathrm{raw}}=\mathcal{M}_{t_i}^\theta(x_i)
}$$

$${
x_{i+1}=\alpha_{t_{i+1}}\hat x_1^*+\beta_{t_{i+1}}\mathcal{W}_{t_i}^\theta(x_i)
}$$

이 식을 쓰면 HardFlow Problem 6에 더 가까워진다.  
1차 구현은 linear schedule + 단순 보간으로 충분하다.

---

## 7. HardFlow / 기존 YFlow / Improved YFlow

| 항목 | 기존 YFlow | HardFlow | Improved YFlow |
|------|----------|----------|----------------|
| Target 예측 | $x_t+(1-t)v_t$ | $\mathcal{M}_t^\theta$ | 동일 (linear면 같음) |
| 제약 | $P(\hat x_1)$ | $h(\hat x_1)\le 0$ 최적화 | $h$ 최적화 + $P$ warm start ($\mu=0$ 시 HardFlow형) |
| 비용 $C$ | 없음 | 있음 | 있음 |
| 다음 상태 | 선형 보간 | $\mathcal{F}\approx T^y(\bar x_{i+1})$ | 선형 보간 (기본) |
| 적용 시점 | 매 스텝 | 후반 권장 | $t$ + $L_P$ 스케줄 ($P$ 없을 시 $t$ 스케줄) |
| Training | free | free | free |

---
---

## 8. 구현 목표

- [x] 사전학습 $v_t^\theta$로 raw target $\hat x_1^{\mathrm{raw}}=x_t+(1-t)v_t$를 계산한다.
- [x] 물리 연산자 $P$는 해 자체가 아니라 warm start다. $P$는 1-Lipschitz, $P(X_1)=X_1$을 가정한다. (이상적인 $P$가 없는 도메인은 $\mu=0$으로 두어 HardFlow형으로 유연하게 전환)
- [x] $\hat x_1$에서 $h(\hat x_1)\le 0$과 비용 $C$를 푼 뒤, 현재 상태와 **선형 보간**으로 $x_{t+\Delta t}$를 만든다. HardFlow inverse map은 쓰지 않는다.
- [x] $\gamma=0$이면 제약을 끈 원본 flow와 같은 궤적이 나온다.
- [x] $t$와 추정 Lipschitz $\widehat L_P$로 제약 강도를 스케줄한다. 초반 노이즈 구간에서 과도한 투영을 피한다.
- [x] Exp-01 Swiss roll에서 $P=\Pi_{\mathcal{M}}$, $C=d_{\mathcal{M}}^2$, $h$는 tube / core / box다.
- [x] HardFlow와 같은 시드·같은 평가 프로토콜로 Safety / MMD / 시간을 비교한다.
- [x] Default, $n_{\mathrm{eval}}=4000$에서 Safety $\ge 0.99$를 목표로 한다.
